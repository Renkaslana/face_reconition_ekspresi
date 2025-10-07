# Configuration for Emotion Detection System
import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'
SCREENSHOTS_DIR = BASE_DIR / 'screenshots'

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
SCREENSHOTS_DIR.mkdir(exist_ok=True)

# Dataset Configuration
TRAIN_DIR = str(BASE_DIR / 'train')
TEST_DIR = str(BASE_DIR / 'test')

# Model Configuration
MODEL_NAME = 'best_emotion_model.keras'
MODEL_PATH = str(BASE_DIR / MODEL_NAME)
IMG_SIZE = (48, 48)
NUM_CLASSES = 7

# Emotion Labels (Indonesian)
EMOTION_LABELS = [
    'Marah',      # angry
    'Jijik',      # disgusted
    'Takut',      # fearful
    'Senang',     # happy
    'Netral',     # neutral
    'Sedih',      # sad
    'Terkejut'    # surprised
]

# Colors for visualization (BGR format)
EMOTION_COLORS = [
    (0, 0, 255),      # Marah - Red
    (0, 255, 255),    # Jijik - Yellow
    (255, 0, 255),    # Takut - Magenta
    (0, 255, 0),      # Senang - Green
    (200, 200, 200),  # Netral - Gray
    (255, 0, 0),      # Sedih - Blue
    (0, 165, 255)     # Terkejut - Orange
]

# Training Configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
CLASS_WEIGHT_POWER = 1.5

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    'rotation_range': 25,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest',
    'brightness_range': [0.8, 1.2]
}

# Focal Loss Parameters
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = 0.25

# Detection Configuration
DETECTION_CONFIDENCE = 0.7  # Minimum confidence for MediaPipe face detection
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FPS = 30

# MediaPipe Configuration
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.7
MEDIAPIPE_MODEL_SELECTION = 0  # 0 for close-range, 1 for full-range

# Display Configuration
FONT_SCALE = 0.7
FONT_THICKNESS = 2
BOX_THICKNESS = 3
SHOW_FPS = True
SHOW_CONFIDENCE = True

# Logging Configuration
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
