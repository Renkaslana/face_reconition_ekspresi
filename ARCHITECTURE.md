# 🏗️ System Architecture

Dokumentasi arsitektur lengkap sistem Modern Emotion Detection.

## 📊 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  run_demo.py │  │  detect.py   │  │   train.py   │      │
│  │  (Menu GUI)  │  │  (Detection) │  │  (Training)  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │               │
└─────────┼──────────────────┼──────────────────┼──────────────┘
          │                  │                  │
          │                  │                  │
┌─────────┼──────────────────┼──────────────────┼──────────────┐
│         │       BUSINESS LOGIC LAYER          │               │
├─────────┼──────────────────┼──────────────────┼──────────────┤
│         │                  │                  │               │
│         └─────────┬────────┴─────────┬────────┘              │
│                   │                  │                        │
│            ┌──────▼──────┐    ┌─────▼──────┐                │
│            │   utils.py  │    │ config.py  │                │
│            │ (Functions) │    │ (Settings) │                │
│            └──────┬──────┘    └─────┬──────┘                │
│                   │                  │                        │
└───────────────────┼──────────────────┼───────────────────────┘
                    │                  │
                    │                  │
┌───────────────────┼──────────────────┼───────────────────────┐
│                   │   CORE LAYER     │                        │
├───────────────────┼──────────────────┼───────────────────────┤
│                   │                  │                        │
│  ┌────────────────▼───┐  ┌──────────▼───────┐               │
│  │   TensorFlow/Keras │  │  MediaPipe Face  │               │
│  │   (CNN Model)      │  │    Detection     │               │
│  └────────────────────┘  └──────────────────┘               │
│                                                               │
│  ┌────────────────────┐  ┌──────────────────┐               │
│  │   OpenCV           │  │  NumPy/sklearn   │               │
│  │   (Image Process)  │  │  (Math/ML Utils) │               │
│  └────────────────────┘  └──────────────────┘               │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow

### 1️⃣ Training Flow
```
Dataset (train/) 
    ↓
[ImageDataGenerator]
    ↓ (Augmentation)
[Training Images] → [CNN Model] ← [Focal Loss + Class Weights]
    ↓
[Model Evaluation]
    ↓
[Save Model] → best_emotion_model.keras
```

### 2️⃣ Detection Flow (Webcam)
```
[Webcam/Camera]
    ↓
[Capture Frame]
    ↓
[MediaPipe Face Detection] ← (High Accuracy)
    ↓
[Face Regions Detected]
    ↓
[Preprocess Each Face] → (Resize 48x48, Normalize)
    ↓
[CNN Model Prediction]
    ↓
[Emotion + Confidence]
    ↓
[Draw Bounding Box + Label]
    ↓
[Display Frame] → (with FPS counter)
    ↓
[Loop back to Capture Frame]
```

## 📦 Module Dependencies

```
config.py (Configuration)
    ↓ (imported by)
    ├── utils.py
    ├── train.py
    └── detect.py

utils.py (Utilities)
    ↓ (imported by)
    ├── train.py
    └── detect.py

train.py (Training)
    ↓ (uses)
    ├── config.py
    ├── utils.py
    └── TensorFlow/Keras

detect.py (Detection)
    ↓ (uses)
    ├── config.py
    ├── utils.py
    ├── MediaPipe
    └── OpenCV
```

## 🧩 Component Details

### config.py
```
Purpose: Centralized configuration
Contents:
  ├── Dataset paths
  ├── Model configuration
  ├── Training hyperparameters
  ├── Detection settings
  └── UI settings
```

### utils.py
```
Purpose: Shared utility functions
Functions:
  ├── focal_loss() - Custom loss function
  ├── calculate_class_weights() - Balance classes
  ├── model_exists() - Check model cache
  ├── load_model() - Load trained model
  ├── preprocess_face() - Prepare face for prediction
  └── format_prediction() - Format results
```

### train.py
```
Purpose: Model training with caching
Flow:
  1. Check if model exists → Skip if yes
  2. Load and augment data
  3. Calculate class weights
  4. Create CNN model
  5. Train with callbacks
  6. Save best model
  7. Evaluate and log results
```

### detect.py
```
Purpose: Real-time emotion detection
Components:
  ├── EmotionDetector class
  │   ├── __init__() - Load model + MediaPipe
  │   ├── detect_faces() - Find faces (MediaPipe)
  │   ├── predict_emotion() - Predict emotion
  │   ├── process_frame() - Full pipeline
  │   ├── run_webcam() - Webcam mode
  │   ├── process_image() - Image mode
  │   └── process_video() - Video mode
  └── main() - CLI interface
```

## 🎯 CNN Model Architecture

```
Input: 48x48x1 (Grayscale)
    ↓
┌─────────────────────────────────────┐
│ Block 1: Feature Extraction         │
│  Conv2D(64) → BN → Conv2D(64) → BN  │
│  → MaxPool → Dropout(0.25)          │
└─────────────────────────────────────┘
    ↓ (24x24x64)
┌─────────────────────────────────────┐
│ Block 2: Feature Extraction         │
│  Conv2D(128) → BN → Conv2D(128) → BN│
│  → MaxPool → Dropout(0.25)          │
└─────────────────────────────────────┘
    ↓ (12x12x128)
┌─────────────────────────────────────┐
│ Block 3: Feature Extraction         │
│  Conv2D(256) → BN → Conv2D(256) → BN│
│  → MaxPool → Dropout(0.25)          │
└─────────────────────────────────────┘
    ↓ (6x6x256)
┌─────────────────────────────────────┐
│ Block 4: High-Level Features        │
│  Conv2D(512) → BN                   │
│  → GlobalAvgPool → Dropout(0.5)     │
└─────────────────────────────────────┘
    ↓ (512)
┌─────────────────────────────────────┐
│ Classification Head                 │
│  Dense(512) → BN → Dropout(0.5)     │
│  → Dense(256) → BN → Dropout(0.4)   │
│  → Dense(7) → Softmax               │
└─────────────────────────────────────┘
    ↓
Output: [7 probabilities] (Emotions)
```

## 🔍 MediaPipe Face Detection Pipeline

```
Input Frame (BGR)
    ↓
[Convert to RGB]
    ↓
[MediaPipe BlazeFace Model]
    ↓ (Neural Network)
[Face Detections]
    ↓ (for each detection)
[Bounding Box Coordinates]
    ├── x, y (top-left)
    ├── width
    └── height
    ↓
[Convert to Absolute Coordinates]
    ↓
[Validate & Clip to Frame]
    ↓
[Return Face Regions]
```

## 🚀 Execution Flow

### Scenario 1: First Time Training
```
1. User runs: python train.py
2. Check model exists? → NO
3. Calculate class weights
4. Create data generators
5. Create CNN model
6. Start training (50 epochs)
7. Save best model
8. Done! (Model cached)
```

### Scenario 2: Already Trained
```
1. User runs: python train.py
2. Check model exists? → YES
3. Load model from cache
4. Skip training
5. Done! (30x faster)
```

### Scenario 3: Real-time Detection
```
1. User runs: python detect.py --mode webcam
2. Load trained model
3. Initialize MediaPipe
4. Open webcam
5. LOOP:
   a. Capture frame
   b. Detect faces (MediaPipe)
   c. For each face:
      - Extract region
      - Preprocess (48x48)
      - Predict emotion
      - Draw box + label
   d. Calculate FPS
   e. Display frame
   f. Check key press (Q=quit, S=save)
6. Cleanup & exit
```

## 💾 File System Structure

```
/app/
│
├── 🐍 Python Scripts
│   ├── config.py         (1 file,  2.2 KB)
│   ├── utils.py          (1 file,  6.8 KB)
│   ├── train.py          (1 file,  8.2 KB)
│   ├── detect.py         (1 file, 15.0 KB)
│   ├── run_demo.py       (1 file,  6.0 KB)
│   └── test_system.py    (1 file,  7.5 KB)
│
├── 📚 Documentation
│   ├── README.md         (1 file,  6.5 KB)
│   ├── QUICKSTART.md     (1 file,  2.5 KB)
│   ├── IMPROVEMENTS.md   (1 file,  9.7 KB)
│   ├── SUMMARY.md        (1 file,  8.3 KB)
│   └── ARCHITECTURE.md   (this file)
│
├── 📦 Dependencies
│   └── requirements.txt  (1 file,  0.3 KB)
│
├── 💾 Model & Data
│   ├── best_emotion_model.keras  (31 MB)
│   ├── train/            (28,709 images, 7 classes)
│   └── test/             (7,178 images, 7 classes)
│
└── 📂 Generated
    ├── models/           (for model versions)
    ├── logs/             (training & detection logs)
    └── screenshots/      (saved screenshots)
```

## 🎛️ Configuration Hierarchy

```
config.py (Default Settings)
    ↓
Environment Variables (Optional)
    ↓
Command Line Arguments (Override)
    ↓
Final Configuration Used
```

Example:
```bash
# Use default from config.py
python train.py

# Override with CLI args
python train.py --epochs 100 --batch-size 64

# Environment variable (if supported)
export EMOTION_EPOCHS=100
python train.py
```

## 🔐 Error Handling Strategy

```
Level 1: Input Validation
  ├── Check file exists
  ├── Validate parameters
  └── Verify dependencies

Level 2: Graceful Degradation
  ├── Skip invalid frames
  ├── Continue on single failure
  └── Log errors for review

Level 3: Recovery
  ├── Retry operations
  ├── Fallback mechanisms
  └── Safe cleanup

Level 4: User Feedback
  ├── Clear error messages
  ├── Actionable suggestions
  └── Log file reference
```

## 📊 Performance Optimization

### Training Optimizations
- ✅ Batch Normalization (stabilize training)
- ✅ Data Augmentation (improve generalization)
- ✅ Early Stopping (prevent overfitting)
- ✅ Learning Rate Scheduling (better convergence)
- ✅ Class Weights (handle imbalance)

### Detection Optimizations
- ✅ Model Caching (no reload every frame)
- ✅ MediaPipe (GPU-accelerated face detection)
- ✅ Batch Prediction (when possible)
- ✅ Frame Skipping (if needed for real-time)
- ✅ Efficient Preprocessing (vectorized operations)

## 🧪 Testing Strategy

```
test_system.py
    ├── Import Tests (packages available?)
    ├── Config Tests (settings valid?)
    ├── Utils Tests (functions work?)
    ├── Model Tests (can load?)
    ├── MediaPipe Tests (initialized?)
    ├── Dataset Tests (data exists?)
    └── Integration Tests (end-to-end?)
```

## 🎓 Design Patterns Used

1. **Separation of Concerns**
   - Config, Utils, Train, Detect separated

2. **Dependency Injection**
   - Pass config to functions, not hardcode

3. **Factory Pattern**
   - Create model, data generators via functions

4. **Singleton Pattern**
   - Logger instance shared

5. **Strategy Pattern**
   - Different detection modes (webcam/image/video)

## 🔄 Version Control

```
Git Structure:
  ├── .git/              (version control)
  ├── .gitignore         (exclude model, data, logs)
  └── All source files   (tracked)

Excluded from Git:
  ├── best_emotion_model.keras  (too large)
  ├── train/             (dataset)
  ├── test/              (dataset)
  ├── logs/              (runtime logs)
  └── screenshots/       (user data)
```

## 🎯 Key Architectural Decisions

### ✅ Why Modular Structure?
- Easier to maintain
- Better code reusability
- Easier to test
- Clear responsibilities

### ✅ Why MediaPipe over Haar Cascade?
- Much higher accuracy (95% vs 75%)
- Modern deep learning-based
- GPU-accelerated
- Better multi-angle support

### ✅ Why Auto-Caching?
- Save time (30x faster startup)
- Better user experience
- Efficient resource usage
- Optional force retrain

### ✅ Why CLI + Interactive?
- Flexibility (power users + beginners)
- Scriptable (automation possible)
- User-friendly (menu for beginners)
- Professional (production-ready)

## 📈 Scalability Considerations

### Current System
- ✅ Single model, single camera
- ✅ Real-time processing
- ✅ Local deployment

### Future Extensions
- 🔮 Multiple cameras support
- 🔮 Distributed processing
- 🔮 Cloud deployment
- 🔮 API endpoint
- 🔮 Mobile app integration

## 🎨 UI/UX Design Philosophy

1. **Simplicity First**
   - One-command execution
   - Clear error messages
   - Intuitive controls

2. **Feedback Loop**
   - FPS counter (performance)
   - Confidence display (trust)
   - Visual feedback (colors)

3. **Progressive Disclosure**
   - Basic: run_demo.py
   - Advanced: CLI with args
   - Expert: Edit config.py

---

**Architecture Status**: ✅ Production Ready

**Terakhir Diperbarui**: 2025-10-07

Made with ❤️ following software engineering best practices

### Contributor
- **Muhammad Affif**
- **Muhamad Fahren Andrean Rangkuti**
- **Putri Areka Sandra**