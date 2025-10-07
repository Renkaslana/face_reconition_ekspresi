# 📝 Quick Reference Cheat Sheet

Referensi cepat untuk semua perintah dan fitur sistem Emotion Detection.

## ⚡ Quick Commands

### 🎯 Most Common
```bash
# Test system
python test_system.py

# Train (auto-skip if exists)
python train.py

# Webcam detection
python detect.py --mode webcam

# Interactive menu
python run_demo.py
```

### 🔧 Training Commands
```bash
# Normal training (skip if model exists)
python train.py

# Force retrain from scratch
python train.py --force-retrain

# Custom epochs
python train.py --epochs 100

# Custom batch size
python train.py --batch-size 64

# Combine options
python train.py --force-retrain --epochs 100 --batch-size 64
```

### 📹 Detection Commands
```bash
# Webcam (default camera)
python detect.py --mode webcam

# Webcam (specific camera)
python detect.py --mode webcam --camera 1

# Process image
python detect.py --mode image --input photo.jpg

# Process image and save
python detect.py --mode image --input photo.jpg --output result.jpg

# Process video
python detect.py --mode video --input video.mp4 --output result.mp4

# Process video without display (faster)
python detect.py --mode video --input video.mp4 --output result.mp4 --no-display

# Use custom model
python detect.py --mode webcam --model /path/to/model.keras
```

## ⌨️ Keyboard Controls

### Webcam Mode
| Key | Action |
|-----|--------|
| `Q` | Quit/Exit |
| `S` | Save Screenshot |
| `ESC` | Alternative quit |

## 📁 File Locations

### Important Files
```
config.py              # All settings
utils.py               # Utility functions
train.py               # Training script
detect.py              # Detection script
run_demo.py            # Interactive menu
test_system.py         # System validation
```

### Generated Files
```
best_emotion_model.keras       # Trained model (31 MB)
logs/emotion_detection_*.log   # Log files
screenshots/screenshot_*.jpg   # Saved screenshots
```

### Data Folders
```
train/        # Training data (28,709 images)
test/         # Test data (7,178 images)
models/       # Model versions
```

## ⚙️ Configuration Quick Edit

Edit `config.py` to customize:

### Model Settings
```python
NUM_CLASSES = 7              # Number of emotions
IMG_SIZE = (48, 48)          # Input image size
MODEL_NAME = 'best_emotion_model.keras'
```

### Training Settings
```python
BATCH_SIZE = 32              # Training batch size
EPOCHS = 50                  # Max training epochs
LEARNING_RATE = 0.001        # Learning rate
```

### Detection Settings
```python
FRAME_WIDTH = 1280           # Webcam width
FRAME_HEIGHT = 720           # Webcam height
FRAME_FPS = 30               # Target FPS
```

### MediaPipe Settings
```python
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.7  # Face detection threshold
MEDIAPIPE_MODEL_SELECTION = 0             # 0=close, 1=full range
```

### Display Settings
```python
SHOW_FPS = True              # Show FPS counter
SHOW_CONFIDENCE = True       # Show confidence %
FONT_SCALE = 0.7             # Text size
BOX_THICKNESS = 3            # Bounding box thickness
```

## 🎨 Emotion Labels & Colors

| Emotion | Indonesian | Color | RGB |
|---------|-----------|-------|-----|
| Angry | Marah | 🔴 Red | (0, 0, 255) |
| Disgusted | Jijik | 💛 Yellow | (0, 255, 255) |
| Fearful | Takut | 💜 Magenta | (255, 0, 255) |
| Happy | Senang | 💚 Green | (0, 255, 0) |
| Neutral | Netral | ⚪ Gray | (200, 200, 200) |
| Sad | Sedih | 🔵 Blue | (255, 0, 0) |
| Surprised | Terkejut | 🟠 Orange | (0, 165, 255) |

## 🔍 Debugging Commands

### Check System Status
```bash
python test_system.py
```

### View Logs
```bash
# Latest log
ls -lt logs/ | head -5

# View log content
cat logs/emotion_detection_*.log

# Follow log in real-time
tail -f logs/emotion_detection_*.log
```

### Check Model
```bash
# Check if model exists
ls -lh best_emotion_model.keras

# Check model details (Python)
python -c "from utils import load_model; m = load_model(); m.summary()"
```

### Check Dataset
```bash
# Count training images
find train/ -type f -name "*.jpg" -o -name "*.png" | wc -l

# Count test images
find test/ -type f -name "*.jpg" -o -name "*.png" | wc -l

# List classes
ls train/
```

## 🐛 Troubleshooting Quick Fixes

### Problem: Model not found
```bash
python train.py
```

### Problem: Camera not opening
```bash
# Try camera 1 instead of 0
python detect.py --mode webcam --camera 1

# List available cameras (Linux)
ls /dev/video*

# Test camera with OpenCV
python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"
```

### Problem: Face not detected
```python
# Edit config.py, lower confidence:
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5  # was 0.7
```

### Problem: Low FPS
```python
# Edit config.py:
FRAME_WIDTH = 640            # was 1280
FRAME_HEIGHT = 480           # was 720
```

### Problem: Out of memory
```bash
# Use smaller batch size
python train.py --batch-size 16  # was 32
```

### Problem: Wrong output shape
```bash
# Retrain with correct classes
python train.py --force-retrain
```

## 📊 Performance Tuning

### For Speed (Lower Quality)
```python
# config.py
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.6
```

### For Quality (Slower)
```python
# config.py
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.8
```

### For Battery Save
```python
# config.py
FRAME_FPS = 15  # Lower FPS
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
```

## 🧪 Testing Checklist

- [ ] Run `python test_system.py` - All pass?
- [ ] Model exists and loads?
- [ ] Camera opens successfully?
- [ ] Face detection works?
- [ ] Emotion prediction works?
- [ ] FPS > 20?
- [ ] Screenshot saves correctly?

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| README.md | Full documentation |
| QUICKSTART.md | Quick start guide |
| SUMMARY.md | Project overview |
| IMPROVEMENTS.md | What's improved |
| ARCHITECTURE.md | System architecture |
| CHEATSHEET.md | This file |

## 💡 Pro Tips

### Tip 1: Faster Development
```bash
# Use --no-display for faster video processing
python detect.py --mode video --input video.mp4 --no-display
```

### Tip 2: Multiple Experiments
```python
# Save different model versions
# In config.py:
MODEL_NAME = 'model_v2.keras'  # Change version
```

### Tip 3: Batch Processing
```bash
# Process multiple images
for img in *.jpg; do
    python detect.py --mode image --input "$img" --output "result_$img"
done
```

### Tip 4: Monitor Training
```bash
# In another terminal, watch logs
tail -f logs/emotion_detection_*.log
```

### Tip 5: Quick Model Info
```bash
python -c "from utils import *; m = load_model(); print(f'Params: {m.count_params():,}')"
```

## 🔗 Import Shortcuts

### In Python Script
```python
# Import everything
import config
import utils
from utils import load_model, preprocess_face

# Quick test
model = load_model()
print(f"Model loaded: {model.count_params():,} params")
```

### In Jupyter Notebook
```python
import sys
sys.path.append('/app')

import config
from utils import load_model
from detect import EmotionDetector

# Quick detection test
detector = EmotionDetector()
# Use detector.process_image() or detector.run_webcam()
```

## ⚡ One-Liners

### Check Python/TF versions
```bash
python --version && python -c "import tensorflow as tf; print(f'TF: {tf.__version__}')"
```

### Count model parameters
```bash
python -c "from utils import load_model; print(f'{load_model().count_params():,}')"
```

### Test MediaPipe
```bash
python -c "import mediapipe as mp; print('MediaPipe OK')"
```

### List emotion classes
```bash
python -c "import config; print('\n'.join(config.EMOTION_LABELS))"
```

### Check GPU
```bash
python -c "import tensorflow as tf; print(f'GPU: {len(tf.config.list_physical_devices(\"GPU\"))} device(s)')"
```

## 📦 Dependency Quick Reference

### Core (Required)
```bash
pip install tensorflow opencv-python mediapipe numpy scikit-learn
```

### Full (Recommended)
```bash
pip install -r requirements.txt
```

### Version Check
```bash
pip list | grep -E "tensorflow|opencv|mediapipe|numpy|scikit"
```

## 🎯 Common Workflows

### Workflow 1: First Time Setup
```bash
1. pip install -r requirements.txt
2. python test_system.py
3. python train.py
4. python detect.py --mode webcam
```

### Workflow 2: Daily Use
```bash
1. python detect.py --mode webcam
   (model auto-loads, no training needed)
```

### Workflow 3: Process Video
```bash
1. python detect.py --mode video --input video.mp4 --output result.mp4 --no-display
2. Check result.mp4
```

### Workflow 4: Retrain
```bash
1. python train.py --force-retrain --epochs 100
2. python detect.py --mode webcam
```

## 🎓 Learning Resources

### Understanding the Code
1. Start with `config.py` - understand settings
2. Read `utils.py` - utility functions
3. Review `train.py` - training pipeline
4. Study `detect.py` - detection pipeline

### Documentation Order
1. QUICKSTART.md - Get started quickly
2. README.md - Understand features
3. ARCHITECTURE.md - System design
4. IMPROVEMENTS.md - What changed
5. This file - Quick reference

---

**Quick Tip**: Keep this file open while working! 📌

**Updated**: 2025-10-07
