# 🎭 Modern Emotion Detection System

Sistem deteksi emosi real-time yang modern dan akurat menggunakan Deep Learning dan MediaPipe.

## ✨ Fitur Utama

- 🚀 **Auto-Caching**: Model otomatis di-cache, tidak perlu training ulang
- 🎯 **MediaPipe Face Detection**: Teknologi terbaru dari Google, lebih akurat dari Haar Cascade
- 🧠 **7 Emotion Classes**: Marah, Jijik, Takut, Senang, Netral, Sedih, Terkejut
- ⚡ **Real-time Performance**: Optimized untuk FPS tinggi (>20 FPS)
- 📹 **Multi-mode**: Webcam, image, dan video processing
- 🎨 **Modern UI**: Color-coded emotions dengan confidence display
- 📊 **Class Balancing**: Focal Loss + Enhanced Class Weights

## 🛠️ Instalasi

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Persiapkan Dataset

Pastikan struktur folder seperti ini:
```
/app/
├── train/           # Training data
│   ├── angry/
│   ├── disgusted/
│   ├── fearful/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprised/
├── test/            # Test/validation data
│   └── (sama seperti train)
```

## 🚀 Cara Penggunaan

### Mode 1: Training Model (Hanya Sekali)

Model akan otomatis di-cache. Training hanya perlu dilakukan sekali!

```bash
# Training otomatis (skip jika model sudah ada)
python train.py

# Force retrain (jika ingin training ulang)
python train.py --force-retrain

# Custom epochs dan batch size
python train.py --epochs 100 --batch-size 64
```

**Output:**
- Model disimpan di: `best_emotion_model.keras`
- Logs disimpan di: `logs/`

### Mode 2: Real-time Detection (Webcam)

```bash
# Deteksi real-time dari webcam
python detect.py --mode webcam

# Gunakan camera lain (misal camera eksternal)
python detect.py --mode webcam --camera 1
```

**Kontrol:**
- `Q` - Quit/keluar
- `S` - Screenshot (disimpan di folder `screenshots/`)

### Mode 3: Process Image

```bash
# Process single image
python detect.py --mode image --input foto.jpg --output hasil.jpg

# Process tanpa save, hanya tampilkan
python detect.py --mode image --input foto.jpg
```

### Mode 4: Process Video

```bash
# Process video file
python detect.py --mode video --input video.mp4 --output hasil.mp4

# Process tanpa display window (faster)
python detect.py --mode video --input video.mp4 --output hasil.mp4 --no-display
```

## 📁 Struktur File

```
/app/
├── config.py              # Configuration settings
├── utils.py               # Utility functions
├── train.py               # Training script
├── detect.py              # Detection script
├── requirements.txt       # Dependencies
├── README.md              # Documentation (file ini)
│
├── models/                # Model storage
│   └── (empty)
├── logs/                  # Training logs
│   └── (auto-generated)
├── screenshots/           # Saved screenshots
│   └── (auto-generated)
│
├── best_emotion_model.keras  # Trained model (auto-generated)
├── train/                    # Training dataset
└── test/                     # Validation dataset
```

## ⚙️ Konfigurasi

Edit `config.py` untuk mengubah settings:

```python
# Model settings
NUM_CLASSES = 7
IMG_SIZE = (48, 48)

# Training settings
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Detection settings
DETECTION_CONFIDENCE = 0.7
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Display settings
SHOW_FPS = True
SHOW_CONFIDENCE = True
```

## 🎯 Accuracy & Performance

**Model Architecture:**
- Deep CNN dengan 4 convolutional blocks
- Batch Normalization untuk stabilitas
- Dropout untuk regularization
- Global Average Pooling
- ~2.7M parameters (~10.4 MB)

**Expected Performance:**
- Validation Accuracy: >65% (balanced across all classes)
- Real-time FPS: >20 FPS (CPU), >60 FPS (GPU)
- Face Detection: MediaPipe (lebih akurat dari Haar Cascade)

## 🔄 Perbedaan dari Versi Lama

| Aspek | Versi Lama | Versi Baru (Modern) |
|-------|-----------|---------------------|
| **Format** | Jupyter Notebook | Python Scripts |
| **Training** | Selalu training ulang | Auto-caching, skip jika ada |
| **Face Detection** | Haar Cascade (kurang akurat) | MediaPipe (sangat akurat) |
| **Emotion Classes** | 5 (tidak match model) | 7 (lengkap) |
| **Interface** | Cell by cell | CLI dengan arguments |
| **Modularity** | Monolithic | Modular (config, utils, train, detect) |
| **Error Handling** | Minimal | Comprehensive dengan logging |
| **Flexibility** | Hardcoded | Configurable |

## 🐛 Troubleshooting

### Problem: "Model not found"
**Solution:** Jalankan training terlebih dahulu:
```bash
python train.py
```

### Problem: "Failed to open camera"
**Solution:** 
- Pastikan webcam terhubung
- Coba camera ID lain: `--camera 1`
- Check camera permissions

### Problem: "Out of memory"
**Solution:**
- Kurangi batch size: `--batch-size 16`
- Kurangi frame resolution di `config.py`

### Problem: "Face tidak terdeteksi"
**Solution:**
- Pastikan pencahayaan cukup
- Wajah menghadap kamera
- Turunkan `MEDIAPIPE_MIN_DETECTION_CONFIDENCE` di `config.py`

## 📚 Dependencies

- **TensorFlow** ≥2.13.0 - Deep learning framework
- **OpenCV** ≥4.8.0 - Computer vision
- **MediaPipe** ≥0.10.0 - Face detection (Google)
- **NumPy** ≥1.24.0 - Numerical computing
- **scikit-learn** ≥1.3.0 - Class weights calculation

## 📝 Logs

Semua aktivitas di-log ke folder `logs/`:
- Training logs: `logs/emotion_detection_YYYYMMDD_HHMMSS.log`
- Detection logs: Sama

## 🎨 Emotion Colors

| Emotion | Color | RGB |
|---------|-------|-----|
| Marah (Angry) | 🔴 Red | (0, 0, 255) |
| Jijik (Disgusted) | 💛 Yellow | (0, 255, 255) |
| Takut (Fearful) | 💜 Magenta | (255, 0, 255) |
| Senang (Happy) | 💚 Green | (0, 255, 0) |
| Netral (Neutral) | ⚪ Gray | (200, 200, 200) |
| Sedih (Sad) | 🔵 Blue | (255, 0, 0) |
| Terkejut (Surprised) | 🟠 Orange | (0, 165, 255) |

## 🤝 Contributing

Sistem ini adalah versi modern dan improved. Jika ada bug atau suggestion:
1. Check logs di folder `logs/`
2. Pastikan dependencies ter-install dengan benar
3. Review konfigurasi di `config.py`

## 📄 License

Silakan digunakan untuk keperluan edukasi dan penelitian.

## 🎓 Credits

- **Deep Learning**: TensorFlow/Keras
- **Face Detection**: MediaPipe (Google)
- **Dataset**: FER2013 (assumed)
- **Architecture**: Custom CNN with Focal Loss

---

**Made with ❤️ using Modern Python & MediaPipe**

Untuk pertanyaan atau issue, check logs atau review configuration!

### Contributor
- **Muhammad Affif**
- **Muhamad Fahren Andrean Rangkuti**
- **Putri Areka Sandra**