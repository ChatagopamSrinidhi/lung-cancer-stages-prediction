import os
import pickle
import threading
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # use non-GUI backend
import matplotlib.pyplot as plt
import base64

# Lazy import TensorFlow/Keras at call time to speed startup
_keras = None
_model = None
_model_lock = threading.Lock()

def _ensure_keras():
    global _keras
    if _keras is None:
        # prefer tensorflow.keras for compatibility
        from tensorflow import keras
        _keras = keras
    return _keras

def load_history():
    path = os.path.join('model','history.pckl')
    if os.path.exists(path):
        with open(path,'rb') as f:
            return pickle.load(f)
    return None

def load_dataset():
    X_path = os.path.join('model','X.txt.npy')
    Y_path = os.path.join('model','Y.txt.npy')
    if not os.path.exists(X_path) or not os.path.exists(Y_path):
        raise FileNotFoundError('Preprocessed dataset not found in model/ (X.txt.npy / Y.txt.npy)')
    X = np.load(X_path)
    Y = np.load(Y_path)
    X = X.astype('float32')/255.0
    # one-hot if needed
    from numpy import unique
    if Y.ndim==1 or (Y.ndim==2 and Y.shape[1]==1):
        unique_classes = unique(Y)
        num_classes = len(unique_classes)
        keras = _ensure_keras()
        Y = keras.utils.to_categorical(Y, num_classes=num_classes)
    return X, Y

def build_model(input_shape, num_classes):
    keras = _ensure_keras()
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_model():
    """Return cached model; do not load until requested."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                # Try loading saved full model first
                keras = _ensure_keras()
                model_path = os.path.join('model','model_weights.weights.h5')
                if os.path.exists(model_path):
                    try:
                        _model = keras.models.load_model(model_path)
                        return _model
                    except Exception:
                        # fallback: reconstruct architecture and load weights
                        try:
                            X, Y = load_dataset()
                            _model = build_model(input_shape=X.shape[1:], num_classes=Y.shape[1])
                            _model.load_weights(model_path)
                            return _model
                        except Exception:
                            _model = None
                            return None
                else:
                    return None
    return _model

def evaluate_on_test_split():
    X, Y = load_dataset()
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = get_model()
    if model is None:
        raise RuntimeError('Model not available')
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    return loss, acc

def predict_image_bytes(image_bytes):
    # Accept raw image bytes, return predicted class index and probabilities
    keras = _ensure_keras()
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    img = img.resize((32,32))
    arr = np.asarray(img).astype('float32')/255.0
    arr = arr.reshape(1,32,32,3)
    model = get_model()
    if model is None:
        raise RuntimeError('Model not available')
    preds = model.predict(arr)
    idx = int(np.argmax(preds, axis=1)[0])
    return idx, preds.tolist()[0]

# Training wrapper (runs in thread)
_training_status = {'running': False, 'progress': '', 'result': None}

def _train_thread(epochs=20, batch_size=32):
    global _training_status
    try:
        _training_status['running'] = True
        _training_status['progress'] = 'Loading dataset'
        X, Y = load_dataset()
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        input_shape = X_train.shape[1:]
        num_classes = y_train.shape[1]
        _training_status['progress'] = 'Building model'
        model = build_model(input_shape, num_classes)
        weights_path = os.path.join('model','model_weights.weights.h5')
        checkpoint = _ensure_keras().callbacks.ModelCheckpoint(weights_path, save_best_only=True, save_weights_only=True)
        _training_status['progress'] = 'Training'
        hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[checkpoint], verbose=1)
        # save history
        with open(os.path.join('model','history.pckl'),'wb') as f:
            pickle.dump(hist.history, f)
        # cache the trained model
        global _model
        with _model_lock:
            _model = model
        _training_status['progress'] = 'Completed'
        _training_status['result'] = {'history_keys': list(hist.history.keys())}
    except Exception as e:
        _training_status['progress'] = f'Error: {e}'
    finally:
        _training_status['running'] = False

def start_training(epochs=20, batch_size=32):
    if _training_status['running']:
        return False
    t = threading.Thread(target=_train_thread, args=(epochs,batch_size), daemon=True)
    t.start()
    return True

def training_status():
    return dict(_training_status)

def preprocess_from_folder(dataset_path):
    """Preprocess all images from Dataset/ folder and save to model/"""
    import cv2
    X = []
    Y = []
    all_labels = ['Normal', 'Stage1', 'Stage2', 'Stage3']
    
    # Create case-insensitive mapping
    label_map = {label.lower(): label for label in all_labels}
    
    # First pass: find which labels actually exist in the dataset
    found_labels = set()
    for root, dirs, files in os.walk(dataset_path):
        dir_name = os.path.basename(root)
        dir_name_lower = dir_name.lower()
        if dir_name_lower in label_map:
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    if 'Thumbs.db' not in file:
                        found_labels.add(label_map[dir_name_lower])
    
    labels = sorted(list(found_labels)) if found_labels else all_labels[:3]
    
    # Second pass: load and encode images
    for root, dirs, files in os.walk(dataset_path):
        dir_name = os.path.basename(root)
        dir_name_lower = dir_name.lower()
        # Only process directories that match our labels (case-insensitive)
        if dir_name_lower not in label_map:
            continue
        label_key = label_map[dir_name_lower]
        if label_key not in labels:
            continue
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                if 'Thumbs.db' not in file:
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (32, 32))
                        img = np.array(img)
                        X.append(img)
                        Y.append(labels.index(label_key))
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    np.save('model/X.txt', X)
    np.save('model/Y.txt', Y)
    return len(X)

def plot_history_as_base64():
    """Generate training history plot and return as base64 string"""
    history = load_history()
    if not history:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    if 'accuracy' in history:
        ax1.plot(history['accuracy'], label='Train', marker='o', linewidth=2)
        ax1.plot(history.get('val_accuracy', []), label='Val', marker='s', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    if 'loss' in history:
        ax2.plot(history['loss'], label='Train', marker='o', linewidth=2)
        ax2.plot(history.get('val_loss', []), label='Val', marker='s', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()

def get_confusion_matrix_base64():
    """Evaluate model on test set and return confusion matrix as base64 image"""
    try:
        from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
        X, Y = load_dataset()
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        model = get_model()
        if model is None:
            return None
        
        preds = model.predict(X_test)
        preds = np.argmax(preds, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)
        
        cm = sklearn_confusion_matrix(y_test_labels, preds)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        # Use dynamic labels based on confusion matrix shape
        all_labels = ['Normal', 'Stage1', 'Stage2', 'Stage3']
        num_classes = cm.shape[0]
        labels = all_labels[:num_classes]
        
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, weight='bold')
        
        for i in range(num_classes):
            for j in range(num_classes):
                text = ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i,j]>cm.max()/2 else 'black', fontsize=12, weight='bold')
        
        plt.colorbar(im, ax=ax)
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode()
    except Exception as e:
        return None
