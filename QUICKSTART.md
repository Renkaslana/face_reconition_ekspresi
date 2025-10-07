# 🚀 Quick Start Guide

Panduan cepat untuk mulai menggunakan sistem Emotion Detection modern!

## ⚡ Cara Tercepat (Recommended)

### Step 1: Cek System
```bash
python test_system.py
```

### Step 2: Jalankan Demo Interaktif
```bash
python run_demo.py
```

Pilih opsi yang Anda inginkan dari menu!

## 📹 Langsung Pakai Webcam

```bash
# Langsung jalankan deteksi real-time
python detect.py --mode webcam
```

**Kontrol:**
- `Q` = Quit/Keluar
- `S` = Screenshot

## 🔄 Update Model (Jika Perlu)

Jika model lama tidak kompatibel (output shape salah), retrain:

```bash
# Training ulang dengan 7 emotion classes
python train.py --force-retrain --epochs 50
```

Ini akan memakan waktu (tergantung CPU/GPU), tapi hanya perlu dilakukan **SEKALI**!

## ✅ Verifikasi

Setelah training, test lagi:
```bash
python test_system.py
```

Pastikan output shape model adalah `(None, 7)`, bukan `(None, 5)`.

## 🎯 Fitur Utama yang Diperbaiki

| Fitur | Sebelum | Sesudah |
|-------|---------|---------|
| **Training** | Selalu training ulang | ✅ Auto-skip jika ada |
| **Face Detection** | Haar Cascade (kurang akurat) | ✅ MediaPipe (sangat akurat) |
| **Emotion Classes** | 5 classes (salah) | ✅ 7 classes (benar) |
| **Format** | Jupyter Notebook | ✅ Python Scripts |
| **Interface** | Manual cell by cell | ✅ CLI & Interactive menu |

## 📁 File Penting

- `train.py` - Training script
- `detect.py` - Detection script  
- `run_demo.py` - Interactive demo
- `config.py` - Pengaturan
- `test_system.py` - System validation

## 🐛 Troubleshooting Cepat

### Model output shape salah (5 instead of 7)?
```bash
python train.py --force-retrain
```

### Camera tidak terbuka?
```bash
# Coba camera ID berbeda
python detect.py --mode webcam --camera 1
```

### Face tidak terdeteksi?
Edit `config.py`, turunkan nilai:
```python
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5  # dari 0.7
```

## 💡 Tips

1. **Pencahayaan bagus** = Deteksi lebih akurat
2. **Wajah menghadap kamera** = Face detection lebih mudah
3. **Jarak ideal** = 50cm - 1m dari camera
4. **Training sekali saja** = Model akan di-cache otomatis

## 📊 Expected Performance

- **Accuracy**: >65% (balanced)
- **FPS**: >20 FPS (CPU), >60 FPS (GPU)
- **Training Time**: 30-60 menit (CPU), 5-10 menit (GPU)
- **Face Detection**: MediaPipe (lebih akurat dari Haar Cascade)

## 🎓 More Help

- Full documentation: `README.md`
- Configuration: `config.py`
- Logs: Check folder `logs/`

---

**Selamat mencoba! 🎉**
