#!/usr/bin/env python3
"""
System Test Script
Validate all components are working correctly
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

def print_header(text):
    """Print test section header"""
    print("\n" + "=" * 70)
    print(f"🧪 {text}")
    print("=" * 70)

def test_imports():
    """Test if all required packages are importable"""
    print_header("Testing Imports")
    
    packages = {
        'numpy': 'NumPy',
        'tensorflow': 'TensorFlow',
        'cv2': 'OpenCV',
        'mediapipe': 'MediaPipe',
        'sklearn': 'scikit-learn'
    }
    
    results = {}
    for pkg, name in packages.items():
        try:
            __import__(pkg)
            print(f"✅ {name:<20} - OK")
            results[pkg] = True
        except ImportError as e:
            print(f"❌ {name:<20} - FAILED: {e}")
            results[pkg] = False
    
    return all(results.values())

def test_config():
    """Test configuration file"""
    print_header("Testing Configuration")
    
    try:
        import config
        print(f"✅ Config imported successfully")
        print(f"   - NUM_CLASSES: {config.NUM_CLASSES}")
        print(f"   - IMG_SIZE: {config.IMG_SIZE}")
        print(f"   - MODEL_PATH: {config.MODEL_PATH}")
        return True
    except Exception as e:
        print(f"❌ Config failed: {e}")
        return False

def test_utils():
    """Test utility functions"""
    print_header("Testing Utils")
    
    try:
        import utils
        print(f"✅ Utils imported successfully")
        
        # Test focal loss
        focal = utils.focal_loss()
        print(f"✅ Focal loss function created")
        
        # Test model exists check
        exists = utils.model_exists()
        print(f"✅ Model exists check: {exists}")
        
        return True
    except Exception as e:
        print(f"❌ Utils failed: {e}")
        return False

def test_model_loading():
    """Test model loading"""
    print_header("Testing Model Loading")
    
    try:
        import utils
        
        if not utils.model_exists():
            print("⚠️  Model not found. Skipping model loading test.")
            print("   Run 'python train.py' to create model first.")
            return True
        
        model = utils.load_model()
        print(f"✅ Model loaded successfully")
        print(f"   - Parameters: {model.count_params():,}")
        print(f"   - Input shape: {model.input_shape}")
        print(f"   - Output shape: {model.output_shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe face detection"""
    print_header("Testing MediaPipe Face Detection")
    
    try:
        import mediapipe as mp
        import numpy as np
        import cv2
        
        # Initialize MediaPipe
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        
        print(f"✅ MediaPipe initialized successfully")
        
        # Create dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb_image = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)
        
        # Test detection
        results = face_detection.process(rgb_image)
        print(f"✅ Face detection process successful")
        
        face_detection.close()
        
        return True
    except Exception as e:
        print(f"❌ MediaPipe test failed: {e}")
        return False

def test_dataset():
    """Test dataset availability"""
    print_header("Testing Dataset")
    
    train_dir = 'train'
    test_dir = 'test'
    
    results = {}
    
    # Check train directory
    if os.path.exists(train_dir):
        subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        train_count = sum([len(os.listdir(os.path.join(train_dir, d))) for d in subdirs])
        print(f"✅ Train directory found")
        print(f"   - Classes: {len(subdirs)}")
        print(f"   - Images: {train_count:,}")
        results['train'] = True
    else:
        print(f"❌ Train directory not found")
        results['train'] = False
    
    # Check test directory
    if os.path.exists(test_dir):
        subdirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
        test_count = sum([len(os.listdir(os.path.join(test_dir, d))) for d in subdirs])
        print(f"✅ Test directory found")
        print(f"   - Classes: {len(subdirs)}")
        print(f"   - Images: {test_count:,}")
        results['test'] = True
    else:
        print(f"❌ Test directory not found")
        results['test'] = False
    
    return all(results.values())

def test_directories():
    """Test required directories"""
    print_header("Testing Directories")
    
    dirs = ['models', 'logs', 'screenshots']
    
    for dir_name in dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"⚠️  {dir_name}/ directory not found (will be created)")
            os.makedirs(dir_name, exist_ok=True)
    
    return True

def test_scripts():
    """Test if main scripts are present"""
    print_header("Testing Scripts")
    
    scripts = {
        'train.py': 'Training script',
        'detect.py': 'Detection script',
        'config.py': 'Configuration',
        'utils.py': 'Utilities',
        'run_demo.py': 'Demo launcher'
    }
    
    results = {}
    for script, desc in scripts.items():
        if os.path.exists(script):
            print(f"✅ {script:<15} - {desc}")
            results[script] = True
        else:
            print(f"❌ {script:<15} - {desc} NOT FOUND")
            results[script] = False
    
    return all(results.values())

def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("🎭 EMOTION DETECTION SYSTEM - VALIDATION")
    print("=" * 70)
    
    tests = [
        ('Imports', test_imports),
        ('Configuration', test_config),
        ('Utils', test_utils),
        ('Directories', test_directories),
        ('Scripts', test_scripts),
        ('Dataset', test_dataset),
        ('Model Loading', test_model_loading),
        ('MediaPipe', test_mediapipe),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} - {status}")
    
    print("\n" + "-" * 70)
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready to use!")
        print("\n📋 Next Steps:")
        print("   1. python train.py        - Train model (if not exists)")
        print("   2. python detect.py       - Run detection")
        print("   3. python run_demo.py     - Interactive demo")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
    
    print("=" * 70)
    
    return passed == total

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
