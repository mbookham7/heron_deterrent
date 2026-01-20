# ============================================================
# FILE: ai/detector.py
# ============================================================

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class Detection:
    def __init__(self, label: str, confidence: float, bbox: Optional[Tuple] = None):
        self.label = label
        self.confidence = confidence
        self.bbox = bbox  # (x, y, w, h)
    
    def __repr__(self):
        return f"Detection(label={self.label}, confidence={self.confidence:.2f})"

class ObjectDetector:
    def __init__(self, model_loader, confidence_threshold: float = 0.6):
        self.interpreter = model_loader.get_interpreter()
        self.input_details = model_loader.get_input_details()
        self.output_details = model_loader.get_output_details()
        self.confidence_threshold = confidence_threshold
        
        # YOLO class labels (customize based on your model)
        self.labels = {
            0: "heron",
            1: "cat",
            2: "dog",
            3: "bird",
            4: "person"
        }
        
        logger.info(f"Object detector initialized with threshold: {confidence_threshold}")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        input_shape = self.input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        
        # Resize and normalize
        processed = cv2.resize(frame, (width, height))
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        processed = np.expand_dims(processed, axis=0)
        
        # Normalize to [0, 1] if model expects float32
        if self.input_details[0]['dtype'] == np.float32:
            processed = processed.astype(np.float32) / 255.0
        
        return processed
    
    def infer(self, frame: np.ndarray) -> Detection:
        try:
            # Preprocess
            input_data = self.preprocess_frame(frame)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # Get outputs (simplified YOLO output parsing)
            # Actual implementation depends on your model's output format
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Parse detections
            detection = self._parse_output(output_data)
            
            return detection
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return Detection(label="unknown", confidence=0.0)
    
    def _parse_output(self, output_data: np.ndarray) -> Detection:
        # Simplified detection parsing
        # Adjust based on your actual YOLO model output format
        
        try:
            # Example: output shape might be [1, num_detections, 6]
            # where each detection is [x, y, w, h, confidence, class_id]
            
            detections = output_data[0]  # Remove batch dimension
            
            # Find detection with highest confidence
            best_detection = None
            best_confidence = 0.0
            
            for det in detections:
                if len(det) >= 6:
                    confidence = float(det[4])
                    if confidence > best_confidence and confidence >= self.confidence_threshold:
                        best_confidence = confidence
                        class_id = int(det[5])
                        label = self.labels.get(class_id, "unknown")
                        bbox = tuple(det[0:4])
                        best_detection = Detection(label, confidence, bbox)
            
            if best_detection:
                logger.info(f"Detected: {best_detection}")
                return best_detection
            else:
                return Detection(label="unknown", confidence=0.0)
                
        except Exception as e:
            logger.error(f"Output parsing error: {e}")
            return Detection(label="unknown", confidence=0.0)
    
    def aggregate_detections(self, detections: List[Detection]) -> Detection:
        # Find heron with highest confidence
        heron_detections = [d for d in detections if d.label == "heron"]
        if heron_detections:
            return max(heron_detections, key=lambda d: d.confidence)
        
        # Find most common non-unknown detection
        non_unknown = [d for d in detections if d.label != "unknown"]
        if non_unknown:
            # Return most confident
            return max(non_unknown, key=lambda d: d.confidence)
        
        return Detection(label="unknown", confidence=0.0)