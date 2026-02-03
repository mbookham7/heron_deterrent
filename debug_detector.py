#!/usr/bin/env python3
# ============================================================
# FILE: debug_detector.py
# Debug script to test model loading and inference
# ============================================================

import cv2
import numpy as np
import logging
from pathlib import Path
from ai.model_loader import ModelLoader
from ai.detector import ObjectDetector
from utils.config_loader import Config

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_model():
    """Debug the model loading and inference"""
    logger.info("=" * 60)
    logger.info("HERON DETECTOR DEBUG")
    logger.info("=" * 60)
    
    # Load config
    config = Config("config.yaml")
    model_path = config.get('ai.model_path')
    confidence_threshold = config.get('ai.confidence_threshold', 0.6)
    use_edge_tpu = config.get('ai.use_edge_tpu', False)
    
    logger.info(f"Config - Model: {model_path}")
    logger.info(f"Config - Confidence Threshold: {confidence_threshold}")
    logger.info(f"Config - Use Edge TPU: {use_edge_tpu}")
    
    # Check if model exists
    model_file = Path(model_path)
    if not model_file.exists():
        logger.error(f"Model file not found: {model_path}")
        return
    
    logger.info(f"Model file exists: {model_file.stat().st_size} bytes")
    
    # Load model
    logger.info("\nLoading model...")
    model_loader = ModelLoader(model_path, use_edge_tpu)
    
    interpreter = model_loader.get_interpreter()
    input_details = model_loader.get_input_details()
    output_details = model_loader.get_output_details()
    
    if interpreter is None:
        logger.error("Failed to load interpreter")
        return
    
    logger.info(f"Interpreter loaded successfully")
    logger.info(f"Input details: {input_details}")
    logger.info(f"Output details: {output_details}")
    
    # Create detector
    logger.info("\nCreating detector...")
    detector = ObjectDetector(model_loader, confidence_threshold)
    
    # Test with a blank frame
    logger.info("\nTesting with blank (black) frame...")
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detection = detector.infer(blank_frame)
    logger.info(f"Blank frame detection: {detection}")
    
    # Test with random frame
    logger.info("\nTesting with random noise frame...")
    noise_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    detection = detector.infer(noise_frame)
    logger.info(f"Noise frame detection: {detection}")
    
    # Test with a webcam frame if available
    logger.info("\nAttempting to read from camera...")
    camera_device_id = config.get('system.camera_device_id', 0)
    cap = cv2.VideoCapture(camera_device_id)
    
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            logger.info(f"Camera frame shape: {frame.shape}")
            detection = detector.infer(frame)
            logger.info(f"Camera frame detection: {detection}")
        else:
            logger.warning("Could not read from camera")
    else:
        logger.warning(f"Camera device {camera_device_id} not available")
    
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSIS COMPLETE")
    logger.info("=" * 60)
    
    # Print recommendations
    print("\n\n=== RECOMMENDATIONS ===")
    print("\n1. Check if the model is actually trained to detect herons:")
    print(f"   - Verify the model file: {model_path}")
    print(f"   - Check if it's a YOLO model (v5 or v8)")
    print(f"   - Verify the class labels in detector.py")
    
    print("\n2. If detections are returning confidence 0.0:")
    print(f"   - Current threshold: {confidence_threshold}")
    print("   - Try lowering the threshold in config.yaml")
    print("   - Or verify the model can detect objects at all")
    
    print("\n3. Check model output format:")
    print("   - Look at output_details shape above")
    print("   - Ensure it matches YOLOv5 or YOLOv8 format")
    
    print("\n4. If you're getting 'unknown' detections:")
    print("   - The model may be detecting something but with low confidence")
    print("   - Check the class labels in ObjectDetector.__init__")
    print("   - Verify your model's class indices match")

if __name__ == "__main__":
    debug_model()
