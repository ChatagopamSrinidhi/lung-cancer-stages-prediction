"""
HuggingFace model integration module.
Fetches pretrained lung cancer / medical imaging models from HuggingFace Hub,
downloads them, and runs inference.
"""

import os
import json
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import threading

# Directory to cache downloaded models
HF_CACHE_DIR = os.path.join('model', 'hf_cache')
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# Curated list of lung cancer / medical imaging models from HuggingFace
CURATED_MODELS = [
    {
        'model_id': 'Anwarkh1/Lung_Cancer_Detection',
        'model_name': 'Lung Cancer Detection (ViT)',
        'description': 'Vision Transformer fine-tuned for lung cancer detection from CT scans. Classifies images into cancer types.',
        'pipeline_tag': 'image-classification',
    },
    {
        'model_id': 'dima806/lung_cancer_detection',
        'model_name': 'Lung Cancer Detection (EfficientNet)',
        'description': 'EfficientNet-based lung cancer classifier trained on histopathological images. High accuracy model.',
        'pipeline_tag': 'image-classification',
    },
    {
        'model_id': 'Falconsai/medical_image_classification',
        'model_name': 'Medical Image Classifier',
        'description': 'General medical image classification model capable of identifying various pathologies including lung conditions.',
        'pipeline_tag': 'image-classification',
    },
    {
        'model_id': 'microsoft/rad-dino',
        'model_name': 'RAD-DINO (Microsoft)',
        'description': 'Microsoft RAD-DINO: A biomedical vision foundation model trained on radiology images. Excellent for feature extraction from medical scans.',
        'pipeline_tag': 'image-classification',
    },
    {
        'model_id': 'google/vit-base-patch16-224',
        'model_name': 'ViT Base (Google)',
        'description': 'Google Vision Transformer pre-trained on ImageNet. Can be used as a base model for medical image classification with transfer learning.',
        'pipeline_tag': 'image-classification',
    },
]


def search_hf_models(query="lung cancer", limit=10):
    """Search HuggingFace Hub for models related to lung cancer detection."""
    try:
        url = "https://huggingface.co/api/models"
        params = {
            'search': query,
            'filter': 'image-classification',
            'sort': 'downloads',
            'direction': '-1',
            'limit': limit
        }
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            models = resp.json()
            results = []
            for m in models:
                results.append({
                    'model_id': m.get('modelId', ''),
                    'model_name': m.get('modelId', '').split('/')[-1].replace('-', ' ').replace('_', ' ').title(),
                    'description': f"Downloads: {m.get('downloads', 0):,} | Likes: {m.get('likes', 0)} | Pipeline: {m.get('pipeline_tag', 'N/A')}",
                    'downloads': m.get('downloads', 0),
                    'likes': m.get('likes', 0),
                    'pipeline_tag': m.get('pipeline_tag', 'image-classification'),
                })
            return results
        return []
    except Exception as e:
        print(f"[HF] Search error: {e}")
        return []


def get_curated_models():
    """Return the curated list of suitable models."""
    return CURATED_MODELS


def get_model_info(model_id):
    """Get detailed info about a specific HuggingFace model."""
    try:
        url = f"https://huggingface.co/api/models/{model_id}"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


_hf_pipelines = {}
_hf_load_lock = threading.Lock()


def load_hf_model(model_id):
    """Load a HuggingFace model using the transformers pipeline."""
    global _hf_pipelines
    if model_id in _hf_pipelines:
        return _hf_pipelines[model_id]

    with _hf_load_lock:
        if model_id in _hf_pipelines:
            return _hf_pipelines[model_id]
        try:
            from transformers import pipeline, AutoFeatureExtractor, AutoModelForImageClassification
            # Try loading as image classification pipeline
            try:
                pipe = pipeline(
                    "image-classification",
                    model=model_id,
                    cache_dir=HF_CACHE_DIR
                )
                _hf_pipelines[model_id] = pipe
                return pipe
            except Exception as e:
                print(f"[HF] Pipeline load failed for {model_id}: {e}")
                # Try loading model + extractor separately
                try:
                    extractor = AutoFeatureExtractor.from_pretrained(model_id, cache_dir=HF_CACHE_DIR)
                    model = AutoModelForImageClassification.from_pretrained(model_id, cache_dir=HF_CACHE_DIR)
                    pipe = pipeline("image-classification", model=model, feature_extractor=extractor)
                    _hf_pipelines[model_id] = pipe
                    return pipe
                except Exception as e2:
                    print(f"[HF] Model load failed: {e2}")
                    return None
        except ImportError:
            print("[HF] transformers library not installed")
            return None


# Label mapping for HuggingFace models to our cancer stages
STAGE_MAPPING = {
    # Common label patterns from HF models
    'normal': 'Normal',
    'benign': 'Normal',
    'no_cancer': 'Normal',
    'negative': 'Normal',
    'lung_n': 'Normal',
    'stage1': 'Stage1',
    'stage_1': 'Stage1',
    'stage 1': 'Stage1',
    'early': 'Stage1',
    'adenocarcinoma': 'Stage2',
    'lung_aca': 'Stage2',
    'malignant': 'Stage2',
    'positive': 'Stage2',
    'stage2': 'Stage2',
    'stage_2': 'Stage2',
    'stage 2': 'Stage2',
    'squamous': 'Stage3',
    'lung_scc': 'Stage3',
    'large_cell': 'Stage3',
    'stage3': 'Stage3',
    'stage_3': 'Stage3',
    'stage 3': 'Stage3',
    'advanced': 'Stage3',
    'small_cell': 'Stage3',
}

LABELS = ['Normal', 'Stage1', 'Stage2', 'Stage3']


def map_hf_label(label_str):
    """Map a HuggingFace model label to our stage labels."""
    label_lower = label_str.lower().strip()
    # Direct match
    if label_lower in STAGE_MAPPING:
        return STAGE_MAPPING[label_lower]
    # Partial match
    for key, value in STAGE_MAPPING.items():
        if key in label_lower or label_lower in key:
            return value
    # Default: try to extract stage number
    for i, stage in enumerate(LABELS):
        if stage.lower() in label_lower:
            return stage
    return 'Stage2'  # Default to Stage2 if unknown malignancy


def predict_with_hf_model(model_id, image_bytes):
    """
    Run prediction using a HuggingFace model.
    Returns: (predicted_label, confidence, probabilities_dict, raw_results)
    """
    pipe = load_hf_model(model_id)
    if pipe is None:
        raise RuntimeError(f"Could not load HuggingFace model: {model_id}")

    img = Image.open(BytesIO(image_bytes)).convert('RGB')

    # Run inference
    results = pipe(img)

    if not results:
        raise RuntimeError("No predictions returned from model")

    # Map results to our labels
    probs_dict = {label: 0.0 for label in LABELS}

    for r in results:
        mapped = map_hf_label(r['label'])
        probs_dict[mapped] = max(probs_dict[mapped], r['score'])

    # Normalize
    total = sum(probs_dict.values())
    if total > 0:
        probs_dict = {k: v / total for k, v in probs_dict.items()}

    # Find best
    best_label = max(probs_dict, key=probs_dict.get)
    confidence = probs_dict[best_label]

    return best_label, confidence, probs_dict, results


def predict_with_local_model(image_bytes):
    """Run prediction using the local CNN model."""
    import model_utils
    idx, probs = model_utils.predict_image_bytes(image_bytes)
    label = LABELS[idx] if idx < len(LABELS) else str(idx)
    confidence = max(probs)
    probs_dict = {LABELS[i]: p for i, p in enumerate(probs) if i < len(LABELS)}
    return label, confidence, probs_dict, None
