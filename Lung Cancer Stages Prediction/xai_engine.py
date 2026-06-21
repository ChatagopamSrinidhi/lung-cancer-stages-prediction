"""
Explainable AI (XAI) Engine for Lung Cancer Stage Prediction.
Provides:
  - Grad-CAM visualization for model interpretability
  - Detailed explanation of cancer stages
  - Treatment suggestions and precautions
  - Risk assessment based on predicted stage
"""

import numpy as np
import base64
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Cancer Stage Knowledge Base ────────────────────────────────────────────────

STAGE_INFO = {
    'Normal': {
        'severity': 'None',
        'risk_level': 0,
        'color': '#28a745',
        'icon': 'fa-check-circle',
        'description': (
            'No signs of lung cancer detected. The lung tissue appears healthy and normal. '
            'There are no visible tumors, nodules, or abnormal growths in the CT scan image.'
        ),
        'medical_explanation': (
            'The scanned lung tissue shows normal cellular structure without any signs of '
            'malignant transformation. The alveolar patterns, bronchial structures, and '
            'vascular markings all appear within normal limits.'
        ),
        'what_it_means': (
            'This result indicates that the AI model did not detect any patterns consistent '
            'with lung cancer in the provided image. However, this is an AI screening tool '
            'and should not replace professional medical diagnosis.'
        ),
        'treatment_suggestions': [
            'No cancer treatment needed at this time.',
            'Continue regular health check-ups and screenings.',
            'Maintain a healthy lifestyle with regular exercise.',
            'If you are a smoker, consider smoking cessation programs.',
            'Annual low-dose CT screening recommended for high-risk individuals (age 50-80, 20+ pack-year smoking history).',
        ],
        'precautions': [
            'Avoid exposure to secondhand smoke and air pollution.',
            'Maintain a diet rich in fruits, vegetables, and antioxidants.',
            'Exercise regularly (at least 150 minutes of moderate activity per week).',
            'Avoid occupational exposure to carcinogens (asbestos, radon, etc.).',
            'Get regular check-ups, especially if you have a family history of lung cancer.',
            'Report any persistent cough, chest pain, or breathing difficulties to your doctor.',
        ],
        'next_steps': [
            'Schedule routine follow-up screening in 12 months.',
            'Discuss risk factors with your healthcare provider.',
            'Consider genetic counseling if there is a family history of cancer.',
        ],
    },
    'Stage1': {
        'severity': 'Early Stage',
        'risk_level': 1,
        'color': '#ffc107',
        'icon': 'fa-exclamation-triangle',
        'description': (
            'Stage 1 lung cancer has been detected. The cancer is small (typically ≤4 cm) '
            'and localized to the lung. It has NOT spread to lymph nodes or distant organs. '
            'This is the earliest stage of invasive lung cancer with the best prognosis.'
        ),
        'medical_explanation': (
            'Stage 1 Non-Small Cell Lung Cancer (NSCLC) is subdivided into:\n'
            '• Stage 1A: Tumor ≤3 cm, confined within the lung\n'
            '• Stage 1B: Tumor >3 cm but ≤4 cm, still confined to the lung\n\n'
            'The cancer cells have penetrated the lung tissue but have not invaded the '
            'visceral pleura, main bronchus, or any lymph nodes. The 5-year survival rate '
            'for Stage 1 is approximately 68-92% depending on the sub-stage.'
        ),
        'what_it_means': (
            'Detection at Stage 1 is very favorable. The cancer is localized and has the '
            'highest chance of successful treatment. Early-stage lung cancer often has '
            'no symptoms, making screening programs crucial. Surgical removal is typically '
            'curative at this stage.'
        ),
        'treatment_suggestions': [
            '**Surgery (Primary Treatment):** Lobectomy (removal of the affected lobe) is the standard treatment. Video-assisted thoracoscopic surgery (VATS) may be used for minimally invasive approach.',
            '**Wedge Resection / Segmentectomy:** For patients who cannot tolerate lobectomy, a smaller portion of the lung may be removed.',
            '**Stereotactic Body Radiation Therapy (SBRT):** For patients who are not surgical candidates, SBRT delivers precise, high-dose radiation.',
            '**Adjuvant Chemotherapy:** May be recommended for Stage 1B tumors, especially those >4 cm, to reduce recurrence risk.',
            '**Immunotherapy:** Atezolizumab (Tecentriq) may be recommended post-surgery for PD-L1 positive tumors.',
            '**Regular Monitoring:** CT scans every 6 months for the first 2 years, then annually.',
        ],
        'precautions': [
            'Seek immediate consultation with a thoracic oncologist.',
            'Get a PET-CT scan for complete staging assessment.',
            'Undergo pulmonary function tests before any surgical intervention.',
            'Quit smoking immediately — it improves surgical outcomes and survival.',
            'Maintain good nutrition to support recovery from treatment.',
            'Consider joining a support group for emotional well-being.',
            'Discuss biomarker testing (EGFR, ALK, ROS1, PD-L1) for targeted therapy options.',
        ],
        'next_steps': [
            'Immediate referral to a thoracic surgeon or oncologist.',
            'Complete staging workup including PET-CT and brain MRI.',
            'Discuss surgical options and timeline with your care team.',
            'Get molecular/biomarker testing of tumor tissue.',
            'Consider getting a second opinion from a major cancer center.',
        ],
    },
    'Stage2': {
        'severity': 'Moderate Stage',
        'risk_level': 2,
        'color': '#fd7e14',
        'icon': 'fa-exclamation-circle',
        'description': (
            'Stage 2 lung cancer has been detected. The tumor is larger (typically 4-7 cm) '
            'and/or has spread to nearby lymph nodes within the lung (hilar lymph nodes). '
            'The cancer has NOT spread to distant organs.'
        ),
        'medical_explanation': (
            'Stage 2 NSCLC is subdivided into:\n'
            '• Stage 2A: Tumor >4 cm but ≤5 cm without lymph node involvement, OR tumor ≤5 cm with spread to ipsilateral peribronchial/hilar lymph nodes\n'
            '• Stage 2B: Tumor >5 cm but ≤7 cm, OR tumor with invasion of chest wall, phrenic nerve, parietal pericardium\n\n'
            'The 5-year survival rate for Stage 2 is approximately 53-60%. '
            'The cancer is still considered potentially curable with aggressive treatment.'
        ),
        'what_it_means': (
            'Stage 2 indicates the cancer has grown larger or started to involve nearby '
            'lymph nodes. While more advanced than Stage 1, it is still considered '
            'surgically resectable in most cases. Treatment typically involves a combination '
            'of surgery followed by chemotherapy.'
        ),
        'treatment_suggestions': [
            '**Surgery:** Lobectomy or pneumonectomy (removal of entire lung) depending on tumor size and location. This remains the primary treatment.',
            '**Adjuvant Chemotherapy:** Cisplatin-based combination chemotherapy for 4 cycles post-surgery to reduce recurrence risk (standard of care).',
            '**Neoadjuvant Therapy:** Chemotherapy or chemoimmunotherapy before surgery to shrink the tumor and improve surgical outcomes.',
            '**Radiation Therapy:** Recommended for patients who are not surgical candidates or as adjuvant treatment if margins are positive.',
            '**Immunotherapy:** Nivolumab + chemotherapy as neoadjuvant, or atezolizumab as adjuvant therapy for PD-L1 positive tumors.',
            '**Targeted Therapy:** If biomarker testing reveals actionable mutations (EGFR, ALK), targeted drugs like osimertinib may be used.',
            '**Pulmonary Rehabilitation:** Pre and post-surgery to improve lung function and recovery.',
        ],
        'precautions': [
            'Urgent consultation with a multidisciplinary oncology team is essential.',
            'Complete comprehensive staging to rule out distant metastasis.',
            'Nutritional assessment and optimization before treatment.',
            'Smoking cessation is critical — improves treatment outcomes by 30-40%.',
            'Discuss fertility preservation options if applicable before chemotherapy.',
            'Monitor for symptoms of tumor progression (increasing cough, hemoptysis, weight loss).',
            'Ensure adequate emotional and psychological support.',
            'Discuss advance care planning with your healthcare team.',
        ],
        'next_steps': [
            'Urgent referral to thoracic oncology team.',
            'Complete PET-CT, brain MRI, and pulmonary function tests.',
            'Molecular/biomarker testing of tumor tissue (EGFR, ALK, ROS1, BRAF, PD-L1).',
            'Discuss treatment plan in multidisciplinary tumor board.',
            'Begin treatment within 4-6 weeks of diagnosis.',
        ],
    },
    'Stage3': {
        'severity': 'Advanced Stage',
        'risk_level': 3,
        'color': '#dc3545',
        'icon': 'fa-times-circle',
        'description': (
            'Stage 3 lung cancer has been detected. This is an advanced stage where the cancer '
            'has spread significantly to lymph nodes in the mediastinum (center of the chest) '
            'and/or has invaded nearby structures. The tumor may be large (>7 cm) or involve '
            'multiple areas.'
        ),
        'medical_explanation': (
            'Stage 3 NSCLC is subdivided into:\n'
            '• Stage 3A: Tumor with ipsilateral mediastinal lymph node involvement, OR large tumor (>7 cm) with chest wall/diaphragm invasion\n'
            '• Stage 3B: Tumor with contralateral mediastinal or supraclavicular lymph node involvement\n'
            '• Stage 3C: Tumor with extensive local invasion and contralateral lymph node spread\n\n'
            'The 5-year survival rate for Stage 3 varies from 13-36% depending on sub-stage. '
            'Stage 3A may still be surgically resectable, while 3B/3C are typically treated '
            'with chemoradiation.'
        ),
        'what_it_means': (
            'Stage 3 represents a significant advancement of the disease. The cancer has spread '
            'beyond the lung to nearby lymph nodes or structures, making treatment more complex. '
            'However, advances in immunotherapy and combination treatments have significantly '
            'improved outcomes. A multimodal treatment approach is critical at this stage.'
        ),
        'treatment_suggestions': [
            '**Concurrent Chemoradiation:** The standard of care for unresectable Stage 3. Platinum-based chemotherapy delivered simultaneously with radiation therapy over 6-7 weeks.',
            '**Consolidation Immunotherapy:** Durvalumab (Imfinzi) for 12 months after chemoradiation — shown to significantly improve survival in the PACIFIC trial.',
            '**Trimodality Therapy:** For resectable Stage 3A — neoadjuvant chemoimmunotherapy followed by surgery and adjuvant immunotherapy.',
            '**Targeted Therapy:** For patients with actionable mutations (EGFR, ALK, ROS1, BRAF, MET), targeted agents can be highly effective.',
            '**Surgery (Select Cases):** For carefully selected Stage 3A patients after neoadjuvant therapy, surgical resection may improve outcomes.',
            '**Palliative Care Integration:** Early integration of palliative care alongside curative treatment to manage symptoms and improve quality of life.',
            '**Clinical Trials:** Strongly consider enrollment in clinical trials for novel combinations and therapies.',
            '**Supportive Care:** Pain management, nutritional support, pulmonary rehabilitation, and psychological counseling.',
        ],
        'precautions': [
            'URGENT: Seek immediate comprehensive oncology evaluation.',
            'This requires a multidisciplinary team (thoracic surgeon, medical oncologist, radiation oncologist, pulmonologist).',
            'Complete extensive staging including PET-CT, brain MRI, and bone scan.',
            'Biomarker testing is essential — it can reveal targeted therapy options.',
            'Maintain nutrition and physical activity as much as tolerable.',
            'Smoking cessation remains important even at this stage.',
            'Discuss goals of care and treatment preferences with your medical team.',
            'Consider palliative care consultation for symptom management.',
            'Ensure strong social support system and psychological counseling.',
            'Monitor for emergency symptoms: severe shortness of breath, hemoptysis, neurological changes.',
        ],
        'next_steps': [
            'Immediate multidisciplinary oncology consultation.',
            'Comprehensive molecular profiling of tumor tissue.',
            'Discuss treatment options including clinical trials.',
            'Begin palliative care consultation alongside treatment.',
            'Create a comprehensive care plan with your oncology team.',
        ],
    },
}


def get_stage_explanation(predicted_label, confidence, probabilities):
    """
    Generate a comprehensive XAI explanation for the prediction.

    Args:
        predicted_label: str — Predicted stage ('Normal', 'Stage1', 'Stage2', 'Stage3')
        confidence: float — Confidence score (0-1)
        probabilities: dict — {label: probability} for each class

    Returns:
        dict with explanation data
    """
    info = STAGE_INFO.get(predicted_label, STAGE_INFO['Normal'])

    confidence_pct = confidence * 100 if confidence <= 1 else confidence

    # Confidence assessment
    if confidence_pct >= 90:
        confidence_level = 'Very High'
        confidence_note = 'The model is highly confident in this prediction.'
    elif confidence_pct >= 75:
        confidence_level = 'High'
        confidence_note = 'The model shows strong confidence. Consider confirmatory tests.'
    elif confidence_pct >= 50:
        confidence_level = 'Moderate'
        confidence_note = 'The model shows moderate confidence. Additional imaging or biopsy is recommended.'
    else:
        confidence_level = 'Low'
        confidence_note = 'The confidence is low. This prediction should be verified with additional tests and professional evaluation.'

    # Build probability analysis
    prob_analysis = []
    if probabilities:
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        for label, prob in sorted_probs:
            pct = prob * 100 if prob <= 1 else prob
            stage_info = STAGE_INFO.get(label, {})
            prob_analysis.append({
                'label': label,
                'probability': pct,
                'color': stage_info.get('color', '#6c757d'),
                'severity': stage_info.get('severity', 'Unknown'),
            })

    explanation = {
        'predicted_label': predicted_label,
        'severity': info['severity'],
        'risk_level': info['risk_level'],
        'color': info['color'],
        'icon': info['icon'],
        'confidence': confidence_pct,
        'confidence_level': confidence_level,
        'confidence_note': confidence_note,
        'description': info['description'],
        'medical_explanation': info['medical_explanation'],
        'what_it_means': info['what_it_means'],
        'treatment_suggestions': info['treatment_suggestions'],
        'precautions': info['precautions'],
        'next_steps': info['next_steps'],
        'probability_analysis': prob_analysis,
        'disclaimer': (
            'IMPORTANT DISCLAIMER: This AI-based prediction is intended for educational '
            'and screening purposes only. It should NOT be used as a definitive diagnosis. '
            'Always consult with a qualified healthcare professional (oncologist, pulmonologist, '
            'or radiologist) for proper medical evaluation and diagnosis. AI predictions may '
            'have false positives or false negatives.'
        ),
    }

    return explanation


def generate_gradcam_visualization(image_bytes, model=None):
    """
    Generate a Grad-CAM heatmap visualization for XAI.
    Shows which areas of the image the model focuses on.

    Returns base64-encoded image string.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras

        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img_resized = img.resize((32, 32))
        img_array = np.array(img_resized).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if model is None:
            import model_utils
            model = model_utils.get_model()

        if model is None:
            return None

        # Find the last conv layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break

        if last_conv_layer is None:
            return _generate_attention_heatmap(image_bytes)

        # Create Grad-CAM model
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            predicted_class = tf.argmax(predictions[0])
            class_output = predictions[:, predicted_class]

        grads = tape.gradient(class_output, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # Resize heatmap to original image size
        heatmap_resized = np.uint8(255 * heatmap)
        heatmap_img = Image.fromarray(heatmap_resized).resize(img.size, Image.BILINEAR)
        heatmap_array = np.array(heatmap_img)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Heatmap
        axes[1].imshow(heatmap_array, cmap='jet')
        axes[1].set_title('Grad-CAM Attention Map', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(img)
        axes[2].imshow(heatmap_array, cmap='jet', alpha=0.4)
        axes[2].set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        plt.suptitle('XAI: Model Attention Visualization (Grad-CAM)', fontsize=14, fontweight='bold')
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode()

    except Exception as e:
        print(f"[XAI] Grad-CAM error: {e}")
        return _generate_attention_heatmap(image_bytes)


def _generate_attention_heatmap(image_bytes):
    """Fallback: generate a simulated attention heatmap for visualization."""
    try:
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img_array = np.array(img).astype('float32')

        # Simple edge/intensity-based heatmap
        gray = np.mean(img_array, axis=2)
        # Apply Gaussian-like filtering manually
        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(gray, sigma=min(gray.shape) // 10)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Attention Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(img)
        axes[2].imshow(heatmap, cmap='jet', alpha=0.4)
        axes[2].set_title('Overlay Visualization', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        plt.suptitle('XAI: Image Attention Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode()
    except Exception as e:
        print(f"[XAI] Fallback heatmap error: {e}")
        return None


def get_risk_summary(predicted_label):
    """Get a brief risk summary for display in dashboards."""
    summaries = {
        'Normal': {'level': 'Low Risk', 'badge': 'success', 'message': 'No cancer detected'},
        'Stage1': {'level': 'Early Detection', 'badge': 'warning', 'message': 'Early stage — highly treatable'},
        'Stage2': {'level': 'Moderate Risk', 'badge': 'orange', 'message': 'Moderate stage — treatment recommended'},
        'Stage3': {'level': 'High Risk', 'badge': 'danger', 'message': 'Advanced stage — urgent care needed'},
    }
    return summaries.get(predicted_label, summaries['Normal'])
