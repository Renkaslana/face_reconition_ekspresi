#!/usr/bin/env python3
"""
Modern Training Script for Emotion Detection
- Auto-detects existing models (no unnecessary retraining)
- Progress tracking and logging
- Class weight balancing
- Advanced augmentation
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *

import config
import utils

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

logger = utils.logger


def create_data_generators(batch_size=None):
    """
    Create training and validation data generators with augmentation
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    logger.info("Creating data generators with augmentation...")
    
    # Training augmentation - aggressive
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        **config.AUGMENTATION_CONFIG
    )
    
    # Validation - only rescale
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training flow
    train_gen = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMG_SIZE,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=True
    )
    
    # Validation flow
    val_gen = val_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=config.IMG_SIZE,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False
    )
    
    logger.info(f"Data generators created:")
    logger.info(f"  • Classes: {train_gen.num_classes}")
    logger.info(f"  • Train samples: {train_gen.samples:,}")
    logger.info(f"  • Validation samples: {val_gen.samples:,}")
    
    return train_gen, val_gen


def create_model(num_classes=None, input_shape=None):
    """
    Create optimized CNN model for emotion detection
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if input_shape is None:
        input_shape = (*config.IMG_SIZE, 1)
    
    logger.info(f"Creating model with {num_classes} classes...")
    
    model = Sequential([
        # Input layer
        Input(shape=input_shape, name='Input_48x48_Grayscale'),
        
        # Block 1: Feature Extraction (64 filters)
        Conv2D(64, (3, 3), padding='same', activation='relu', name='Conv_1_1_64'),
        BatchNormalization(name='BN_1_1'),
        Conv2D(64, (3, 3), padding='same', activation='relu', name='Conv_1_2_64'),
        BatchNormalization(name='BN_1_2'),
        MaxPooling2D((2, 2), name='MaxPool_1'),
        Dropout(0.25, name='Dropout_1'),
        
        # Block 2: Feature Extraction (128 filters)
        Conv2D(128, (3, 3), padding='same', activation='relu', name='Conv_2_1_128'),
        BatchNormalization(name='BN_2_1'),
        Conv2D(128, (3, 3), padding='same', activation='relu', name='Conv_2_2_128'),
        BatchNormalization(name='BN_2_2'),
        MaxPooling2D((2, 2), name='MaxPool_2'),
        Dropout(0.25, name='Dropout_2'),
        
        # Block 3: Feature Extraction (256 filters)
        Conv2D(256, (3, 3), padding='same', activation='relu', name='Conv_3_1_256'),
        BatchNormalization(name='BN_3_1'),
        Conv2D(256, (3, 3), padding='same', activation='relu', name='Conv_3_2_256'),
        BatchNormalization(name='BN_3_2'),
        MaxPooling2D((2, 2), name='MaxPool_3'),
        Dropout(0.25, name='Dropout_3'),
        
        # Block 4: High-Level Features (512 filters)
        Conv2D(512, (3, 3), padding='same', activation='relu', name='Conv_4_1_512'),
        BatchNormalization(name='BN_4_1'),
        GlobalAveragePooling2D(name='GlobalAvgPool'),
        Dropout(0.5, name='Dropout_4'),
        
        # Classification Head
        Dense(512, activation='relu', name='Dense_1_512'),
        BatchNormalization(name='BN_Dense_1'),
        Dropout(0.5, name='Dropout_Dense_1'),
        
        Dense(256, activation='relu', name='Dense_2_256'),
        BatchNormalization(name='BN_Dense_2'),
        Dropout(0.4, name='Dropout_Dense_2'),
        
        # Output layer
        Dense(num_classes, activation='softmax', name='Output_Softmax')
    ], name='Optimized_Emotion_CNN')
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss=utils.focal_loss(config.FOCAL_LOSS_GAMMA, config.FOCAL_LOSS_ALPHA),
        metrics=['accuracy']
    )
    
    logger.info(f"Model created: {model.count_params():,} parameters ({model.count_params() * 4 / 1024 / 1024:.1f} MB)")
    
    return model


def train_model(force_retrain=False, epochs=None, batch_size=None):
    """
    Train emotion detection model with auto-caching
    
    Args:
        force_retrain: Force retraining even if model exists
        epochs: Number of training epochs (default from config)
        batch_size: Batch size for training (default from config)
    """
    utils.print_system_info()
    
    # Check if model already exists
    if not force_retrain and utils.model_exists():
        logger.info("✅ Trained model found! Skipping training.")
        logger.info("   Use --force-retrain to train from scratch.")
        return utils.load_model()
    
    if force_retrain and utils.model_exists():
        logger.warning("⚠️  Force retrain enabled. Existing model will be overwritten!")
    
    # Set defaults
    if epochs is None:
        epochs = config.EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    logger.info(f"🚀 Starting training: {epochs} epochs, batch size {batch_size}")
    
    # Calculate class weights
    class_weights = utils.calculate_class_weights(
        config.TRAIN_DIR,
        power=config.CLASS_WEIGHT_POWER
    )
    
    # Create data generators
    train_gen, val_gen = create_data_generators(batch_size)
    
    # Create model
    model = create_model()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            config.MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train
    logger.info("🎯 Training started...")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("✅ Training completed!")
    logger.info(f"   Model saved to: {config.MODEL_PATH}")
    
    # Evaluate
    logger.info("📊 Evaluating model...")
    val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
    logger.info(f"   Validation Loss: {val_loss:.4f}")
    logger.info(f"   Validation Accuracy: {val_accuracy*100:.2f}%")
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Train Emotion Detection Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force retrain even if model exists'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=config.EPOCHS,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=config.BATCH_SIZE,
        help='Batch size for training'
    )
    
    args = parser.parse_args()
    
    try:
        train_model(
            force_retrain=args.force_retrain,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
