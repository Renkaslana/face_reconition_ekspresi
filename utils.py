# Utility Functions for Emotion Detection System
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from sklearn.utils import class_weight
import logging
from datetime import datetime
from pathlib import Path
import config

# Setup Logging
def setup_logging(name='emotion_detection'):
    """Setup logging configuration"""
    log_file = config.LOGS_DIR / f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

logger = setup_logging()

# Focal Loss Implementation
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss untuk mengatasi class imbalance dalam multi-class classification.
    Fokus pada 'hard examples' (contoh yang sulit diprediksi).
    
    Args:
        gamma (float): Parameter fokus. Nilai > 0 mengurangi bobot 'easy examples'.
        alpha (float): Parameter penyeimbang untuk class imbalance.
        
    Returns:
        function: Fungsi loss yang siap digunakan.
    """
    alpha_tensor = tf.constant(alpha, dtype=tf.float32)
    gamma_tensor = tf.constant(gamma, dtype=tf.float32)

    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        cross_entropy = -y_true * tf.math.log(y_pred)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - pt, gamma_tensor)
        alpha_weight = y_true * alpha_tensor + (1 - y_true) * (1.0 - alpha_tensor)
        loss = focal_weight * alpha_weight * cross_entropy
        
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    return focal_loss_fixed

# Calculate Class Weights
def calculate_class_weights(train_path, power=1.5, img_size=(48, 48)):
    """
    Menghitung class weights menggunakan tf.keras.utils.image_dataset_from_directory
    dengan optional exponential scaling.
    """
    logger.info("Calculating class weights...")
    
    # Set TF log level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        color_mode='grayscale',
        shuffle=False,
        interpolation='bilinear',
        batch_size=32
    )
    
    y_classes = np.concatenate([y.numpy() for x, y in train_ds], axis=0)
    class_labels = train_ds.class_names
    
    base_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_classes),
        y=y_classes
    )
    
    base_weights = dict(enumerate(base_weights_array))
    
    if power != 1.0:
        class_weights = {k: v ** power for k, v in base_weights.items()}
        max_weight = max(class_weights.values())
        class_weights = {k: (v / max_weight) * 10 for k, v in class_weights.items()}
    else:
        class_weights = base_weights
    
    logger.info(f"Class weights calculated for {len(y_classes)} samples")
    for idx, name in enumerate(class_labels):
        weight_val = class_weights.get(idx, 1.0)
        logger.info(f"  [{idx}] {name:<12}: weight={weight_val:>5.2f}")
    
    return class_weights

# Check if model exists
def model_exists(model_path=None):
    """
    Check if trained model exists
    """
    if model_path is None:
        model_path = config.MODEL_PATH
    
    exists = Path(model_path).exists()
    if exists:
        file_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Model found: {model_path} ({file_size:.2f} MB)")
    else:
        logger.info(f"Model not found: {model_path}")
    
    return exists

# Load model
def load_model(model_path=None):
    """
    Load trained model with custom objects
    """
    if model_path is None:
        model_path = config.MODEL_PATH
    
    logger.info(f"Loading model from: {model_path}")
    
    try:
        model = keras.models.load_model(
            model_path,
            custom_objects={'focal_loss_fixed': focal_loss(config.FOCAL_LOSS_GAMMA, config.FOCAL_LOSS_ALPHA)}
        )
        logger.info("Model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# Preprocess face for prediction
def preprocess_face(face_img, target_size=(48, 48)):
    """
    Preprocess face image for model prediction
    
    Args:
        face_img: Grayscale face image (numpy array)
        target_size: Target size for model input
        
    Returns:
        Preprocessed face image ready for prediction
    """
    import cv2
    
    # Resize to target size
    face_resized = cv2.resize(face_img, target_size)
    
    # Normalize to [0, 1]
    face_normalized = face_resized.astype('float32') / 255.0
    
    # Reshape for model input (batch_size, height, width, channels)
    face_input = face_normalized.reshape(1, target_size[0], target_size[1], 1)
    
    return face_input

# Format prediction results
def format_prediction(predictions, confidence_threshold=0.0):
    """
    Format prediction results with emotion label and confidence
    
    Args:
        predictions: Model prediction output (probabilities)
        confidence_threshold: Minimum confidence to return result
        
    Returns:
        dict: {'emotion': str, 'confidence': float, 'probabilities': dict}
    """
    idx = np.argmax(predictions)
    confidence = predictions[0][idx]
    
    if confidence < confidence_threshold:
        return None
    
    emotion = config.EMOTION_LABELS[idx]
    color = config.EMOTION_COLORS[idx]
    
    # All probabilities
    probabilities = {
        config.EMOTION_LABELS[i]: float(predictions[0][i]) 
        for i in range(len(config.EMOTION_LABELS))
    }
    
    return {
        'emotion': emotion,
        'confidence': float(confidence),
        'color': color,
        'index': int(idx),
        'probabilities': probabilities
    }

# Print system info
def print_system_info():
    """
    Print system and environment information
    """
    import platform
    import sys
    
    print("="*70)
    print("🎭 EMOTION DETECTION SYSTEM - MODERN & OPTIMIZED")
    print("="*70)
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"💻 OS: {platform.system()} {platform.release()}")
    print(f"🧠 TensorFlow: {tf.__version__}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"⚡ GPU: {len(gpus)} GPU(s) detected")
    else:
        print("🐌 GPU: Running on CPU")
    
    print("="*70)
    print()
