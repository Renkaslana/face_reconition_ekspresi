#!/bin/bash
# Quick Demo Script - One-liner untuk demo

echo "========================================"
echo "🎭 EMOTION DETECTION - QUICK DEMO"
echo "========================================"
echo ""
echo "Checking system..."
python test_system.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ System OK! Starting webcam detection..."
    echo ""
    sleep 2
    python detect.py --mode webcam
else
    echo ""
    echo "❌ System check failed. Please fix issues above."
fi
