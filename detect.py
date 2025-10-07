#!/usr/bin/env python3
"""
Modern Real-time Emotion Detection with MediaPipe
- MediaPipe Face Detection (Google's latest technology)
- Multi-face support
- FPS optimization
- Better accuracy
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import mediapipe as mp
import time
from datetime import datetime

import config
import utils

logger = utils.logger


class EmotionDetector:
    """
    Modern Emotion Detector using MediaPipe for face detection
    """
    
    def __init__(self, model_path=None):
        """
        Initialize emotion detector
        
        Args:
            model_path: Path to trained model (default from config)
        """
        logger.info("Initializing Emotion Detector...")
        
        # Load emotion detection model
        if model_path is None:
            model_path = config.MODEL_PATH
        
        if not utils.model_exists(model_path):
            logger.error(f"Model not found: {model_path}")
            logger.error("Please run 'python train.py' first to train the model.")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = utils.load_model(model_path)
        
        # Initialize MediaPipe Face Detection
        logger.info("Initializing MediaPipe Face Detection...")
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=config.MEDIAPIPE_MODEL_SELECTION,
            min_detection_confidence=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE
        )
        
        logger.info("✅ Emotion Detector initialized successfully!")
    
    def detect_faces(self, frame):
        """
        Detect faces in frame using MediaPipe
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            List of face bounding boxes [(x, y, w, h), ...]
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            h, w, _ = frame.shape
            
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                
                # Convert to absolute coordinates
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure coordinates are within frame
                x = max(0, x)
                y = max(0, y)
                width = min(w - x, width)
                height = min(h - y, height)
                
                faces.append((x, y, width, height))
        
        return faces
    
    def predict_emotion(self, face_img):
        """
        Predict emotion from face image
        
        Args:
            face_img: Grayscale face image
            
        Returns:
            Prediction result dict or None
        """
        # Preprocess face
        face_input = utils.preprocess_face(face_img, config.IMG_SIZE)
        
        # Predict
        predictions = self.model.predict(face_input, verbose=0)
        
        # Format result
        result = utils.format_prediction(predictions)
        
        return result
    
    def process_frame(self, frame):
        """
        Process single frame: detect faces and predict emotions
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Processed frame with annotations
        """
        # Detect faces
        faces = self.detect_faces(frame)
        
        # Convert to grayscale for emotion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face region
            face = gray[y:y+h, x:x+w]
            
            if face.size == 0:
                continue
            
            # Predict emotion
            result = self.predict_emotion(face)
            
            if result is None:
                continue
            
            emotion = result['emotion']
            confidence = result['confidence']
            color = result['color']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, config.BOX_THICKNESS)
            
            # Prepare text
            if config.SHOW_CONFIDENCE:
                text = f'{emotion}: {confidence*100:.1f}%'
            else:
                text = emotion
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE,
                config.FONT_THICKNESS
            )
            
            # Draw text background
            cv2.rectangle(
                frame,
                (x, y - text_height - 10),
                (x + text_width + 10, y),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                frame,
                text,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE,
                (255, 255, 255),
                config.FONT_THICKNESS
            )
        
        return frame
    
    def run_webcam(self, camera_id=0, show_fps=True):
        """
        Run real-time emotion detection from webcam
        
        Args:
            camera_id: Camera device ID (default 0)
            show_fps: Show FPS counter
        """
        logger.info(f"Starting webcam detection (camera {camera_id})...")
        
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.FRAME_FPS)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return
        
        logger.info("📹 Webcam detection started!")
        logger.info("   Controls:")
        logger.info("   - Press 'Q' to quit")
        logger.info("   - Press 'S' to save screenshot")
        print()
        
        # FPS calculation
        fps = 0
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                # Flip frame horizontally (mirror effect)
                frame = cv2.flip(frame, 1)
                
                # Process frame
                frame = self.process_frame(frame)
                
                # Calculate FPS
                frame_count += 1
                if frame_count >= 10:
                    end_time = time.time()
                    fps = frame_count / (end_time - start_time)
                    frame_count = 0
                    start_time = time.time()
                
                # Display FPS
                if show_fps and config.SHOW_FPS:
                    cv2.putText(
                        frame,
                        f'FPS: {int(fps)}',
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        config.FONT_SCALE,
                        (0, 255, 0),
                        config.FONT_THICKNESS
                    )
                
                # Display frame
                cv2.imshow('Emotion Detection - Press Q to Quit', frame)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    logger.info("Quit requested by user")
                    break
                elif key == ord('s') or key == ord('S'):
                    # Save screenshot
                    filename = config.SCREENSHOTS_DIR / f'screenshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
                    cv2.imwrite(str(filename), frame)
                    logger.info(f"📸 Screenshot saved: {filename}")
        
        except KeyboardInterrupt:
            logger.info("Detection interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.face_detection.close()
            logger.info("✅ Webcam detection stopped")
    
    def process_image(self, image_path, output_path=None, show=True):
        """
        Process single image file
        
        Args:
            image_path: Path to input image
            output_path: Path to save output (optional)
            show: Show result in window
        """
        logger.info(f"Processing image: {image_path}")
        
        # Read image
        frame = cv2.imread(str(image_path))
        
        if frame is None:
            logger.error(f"Failed to read image: {image_path}")
            return
        
        # Process frame
        result_frame = self.process_frame(frame)
        
        # Save output
        if output_path:
            cv2.imwrite(str(output_path), result_frame)
            logger.info(f"✅ Result saved: {output_path}")
        
        # Show result
        if show:
            cv2.imshow('Emotion Detection - Press any key to close', result_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return result_frame
    
    def process_video(self, video_path, output_path=None, show=True):
        """
        Process video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            show: Show video while processing
        """
        logger.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if output path specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Process frame
                result_frame = self.process_frame(frame)
                
                # Write frame
                if writer:
                    writer.write(result_frame)
                
                # Show frame
                if show:
                    cv2.imshow('Processing Video - Press Q to stop', result_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Processing stopped by user")
                        break
                
                frame_count += 1
                
                # Progress update
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%)")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self.face_detection.close()
            
            elapsed_time = time.time() - start_time
            avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            logger.info(f"✅ Video processing completed!")
            logger.info(f"   Processed {frame_count} frames in {elapsed_time:.2f}s ({avg_fps:.1f} fps)")
            
            if output_path:
                logger.info(f"   Output saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Real-time Emotion Detection with MediaPipe',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--mode',
        choices=['webcam', 'image', 'video'],
        default='webcam',
        help='Detection mode'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input file path (for image/video mode)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (for image/video mode)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (for webcam mode)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display output window'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to model file (default from config)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize detector
        utils.print_system_info()
        detector = EmotionDetector(model_path=args.model)
        
        # Run detection based on mode
        if args.mode == 'webcam':
            detector.run_webcam(camera_id=args.camera, show_fps=True)
        
        elif args.mode == 'image':
            if not args.input:
                logger.error("--input required for image mode")
                return
            detector.process_image(
                args.input,
                output_path=args.output,
                show=not args.no_display
            )
        
        elif args.mode == 'video':
            if not args.input:
                logger.error("--input required for video mode")
                return
            detector.process_video(
                args.input,
                output_path=args.output,
                show=not args.no_display
            )
    
    except KeyboardInterrupt:
        logger.info("Detection interrupted by user")
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
