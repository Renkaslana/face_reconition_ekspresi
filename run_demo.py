#!/usr/bin/env python3
"""
Quick Demo Launcher for Emotion Detection
Simple interface untuk memulai deteksi emosi
"""

import os
import sys
import argparse

def print_banner():
    """Print welcome banner"""
    print("=" * 70)
    print("🎭 EMOTION DETECTION - QUICK DEMO LAUNCHER")
    print("=" * 70)
    print()

def print_menu():
    """Print main menu"""
    print("📋 Pilih Mode:")
    print()
    print("  1. 📹 Webcam Detection (Real-time)")
    print("  2. 🖼️  Image Detection (Process gambar)")
    print("  3. 🎬 Video Detection (Process video)")
    print("  4. 🎯 Train Model (Training dari awal)")
    print("  5. ℹ️  Show System Info")
    print("  6. ❌ Exit")
    print()

def run_webcam():
    """Run webcam detection"""
    print("\n🚀 Starting webcam detection...")
    print("   Tekan Q untuk keluar, S untuk screenshot\n")
    os.system("python detect.py --mode webcam")

def run_image():
    """Run image detection"""
    image_path = input("\n📁 Masukkan path gambar: ").strip()
    
    if not os.path.exists(image_path):
        print(f"❌ File tidak ditemukan: {image_path}")
        return
    
    save = input("💾 Save hasil? (y/n): ").strip().lower()
    
    if save == 'y':
        output_path = input("📁 Output path (kosongkan untuk auto): ").strip()
        if not output_path:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_result{ext}"
        
        os.system(f'python detect.py --mode image --input "{image_path}" --output "{output_path}"')
    else:
        os.system(f'python detect.py --mode image --input "{image_path}"')

def run_video():
    """Run video detection"""
    video_path = input("\n📁 Masukkan path video: ").strip()
    
    if not os.path.exists(video_path):
        print(f"❌ File tidak ditemukan: {video_path}")
        return
    
    save = input("💾 Save hasil? (y/n): ").strip().lower()
    
    if save == 'y':
        output_path = input("📁 Output path (kosongkan untuk auto): ").strip()
        if not output_path:
            base, ext = os.path.splitext(video_path)
            output_path = f"{base}_result{ext}"
        
        display = input("👁️  Show preview saat processing? (y/n): ").strip().lower()
        
        if display == 'y':
            os.system(f'python detect.py --mode video --input "{video_path}" --output "{output_path}"')
        else:
            os.system(f'python detect.py --mode video --input "{video_path}" --output "{output_path}" --no-display')
    else:
        os.system(f'python detect.py --mode video --input "{video_path}"')

def run_training():
    """Run model training"""
    print("\n⚠️  Training Info:")
    print("   - Model sudah ada akan di-skip (auto-caching)")
    print("   - Training займет waktu (tergantung dataset)")
    print()
    
    confirm = input("🤔 Lanjutkan training? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("❌ Training dibatalkan")
        return
    
    force = input("🔥 Force retrain (training ulang)? (y/n): ").strip().lower()
    
    if force == 'y':
        os.system("python train.py --force-retrain")
    else:
        os.system("python train.py")

def show_system_info():
    """Show system information"""
    print("\n" + "=" * 70)
    print("📊 SYSTEM INFORMATION")
    print("=" * 70)
    
    import platform
    import tensorflow as tf
    try:
        import cv2
        cv_version = cv2.__version__
    except:
        cv_version = "Not installed"
    
    try:
        import mediapipe as mp
        mp_version = mp.__version__
    except:
        mp_version = "Not installed"
    
    print(f"\n🐍 Python: {platform.python_version()}")
    print(f"💻 OS: {platform.system()} {platform.release()}")
    print(f"🧠 TensorFlow: {tf.__version__}")
    print(f"👁️  OpenCV: {cv_version}")
    print(f"🎯 MediaPipe: {mp_version}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"⚡ GPU: {len(gpus)} GPU(s) detected")
    else:
        print("🐌 GPU: Running on CPU")
    
    # Check model
    if os.path.exists('best_emotion_model.keras'):
        size_mb = os.path.getsize('best_emotion_model.keras') / (1024 * 1024)
        print(f"\n✅ Model: Found ({size_mb:.2f} MB)")
    else:
        print("\n❌ Model: Not found (run training first)")
    
    # Check dataset
    train_exists = os.path.exists('train')
    test_exists = os.path.exists('test')
    
    print(f"\n📁 Dataset:")
    print(f"   Train: {'✅ Found' if train_exists else '❌ Not found'}")
    print(f"   Test:  {'✅ Found' if test_exists else '❌ Not found'}")
    
    print("\n" + "=" * 70)
    input("\nPress Enter to continue...")

def main():
    """Main menu loop"""
    parser = argparse.ArgumentParser(description='Quick Demo Launcher')
    parser.add_argument('--auto', choices=['webcam', 'train'], help='Auto-run mode')
    args = parser.parse_args()
    
    # Auto mode
    if args.auto == 'webcam':
        run_webcam()
        return
    elif args.auto == 'train':
        run_training()
        return
    
    # Interactive mode
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input("Pilih opsi (1-6): ").strip()
            
            if choice == '1':
                run_webcam()
            elif choice == '2':
                run_image()
            elif choice == '3':
                run_video()
            elif choice == '4':
                run_training()
            elif choice == '5':
                show_system_info()
            elif choice == '6':
                print("\n👋 Terima kasih! Sampai jumpa!")
                break
            else:
                print("\n❌ Pilihan tidak valid. Silakan pilih 1-6.\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Program dihentikan. Sampai jumpa!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == '__main__':
    main()
