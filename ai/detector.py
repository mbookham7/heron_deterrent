# ============================================================
# FILE: ai/detector_yolov8.py
# Enhanced YOLO detector specifically for YOLOv8 TFLite models
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
        self.bbox = bbox  # (x, y, w, h) normalized 0-1
    
    def __repr__(self):
        return f"Detection(label={self.label}, confidence={self.confidence:.2f})"

class ObjectDetector:
    """
    Enhanced YOLO detector for TFLite models with proper output parsing
    Handles YOLOv5 and YOLOv8 TFLite output formats
    """
    
    def __init__(self, model_loader, confidence_threshold: float = 0.6, iou_threshold: float = 0.45):
        self.interpreter = model_loader.get_interpreter()
        self.input_details = model_loader.get_input_details()
        self.output_details = model_loader.get_output_details()
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # YOLO class labels (customize based on your model)
        self.labels = {
            0: "heron",
            1: "cat",
            2: "dog",
            3: "bird",
            4: "person"
        }
        
        # Determine input size
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]
        
        # Check if quantized
        self.is_quantized = self.input_details[0]['dtype'] in [np.uint8, np.int8]
        
        logger.info(f"YOLO Detector initialized:")
        logger.info(f"  - Input size: {self.input_width}x{self.input_height}")
        logger.info(f"  - Quantized: {self.is_quantized}")
        logger.info(f"  - Confidence threshold: {confidence_threshold}")
        logger.info(f"  - Number of outputs: {len(self.output_details)}")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for YOLO input"""
        # Resize
        processed = cv2.resize(frame, (self.input_width, self.input_height))
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        # Add batch dimension
        processed = np.expand_dims(processed, axis=0)
        
        # Normalize based on quantization
        if self.is_quantized:
            # Keep as uint8
            processed = processed.astype(np.uint8)
        else:
            # Normalize to [0, 1]
            processed = processed.astype(np.float32) / 255.0
        
        return processed
    
    def dequantize(self, data: np.ndarray, output_detail: dict) -> np.ndarray:
        """Dequantize output data if needed"""
        if data.dtype in [np.uint8, np.int8]:
            quant_params = output_detail.get('quantization_parameters', {})
            scale = quant_params.get('scales', [1.0])[0]
            zero_point = quant_params.get('zero_points', [0])[0]
            
            if scale != 1.0 or zero_point != 0:
                data = (data.astype(np.float32) - zero_point) * scale
                logger.debug(f"Dequantized: scale={scale}, zero_point={zero_point}")
        
        return data
    
    def parse_yolov8_output(self, outputs: List[np.ndarray]) -> List[Detection]:
        """
        Parse YOLOv8 TFLite output format
        
        YOLOv8 typically outputs shape: [1, 84, 8400] or similar
        Where:
        - 84 = 4 (bbox) + 80 (classes for COCO) or 4 + num_classes
        - 8400 = number of anchor boxes (varies by input size)
        
        Format is transposed: [batch, features, num_boxes]
        Need to transpose to: [num_boxes, features]
        """
        detections = []
        
        try:
            # Get the main output (usually the largest one)
            output = outputs[0]
            
            # Dequantize if needed
            output = self.dequantize(output, self.output_details[0])
            
            logger.debug(f"Output shape: {output.shape}, dtype: {output.dtype}")
            
            # Handle 1D output
            if len(output.shape) == 1:
                logger.debug(f"1D output detected with shape {output.shape}, cannot parse YOLOv8 format")
                return detections
            
            # Remove batch dimension
            if output.shape[0] == 1:
                output = output[0]
            
            # YOLOv8 format: [num_features, num_boxes]
            # We need [num_boxes, num_features]
            if len(output.shape) == 2 and output.shape[0] < output.shape[1]:
                output = output.T  # Transpose
                logger.debug(f"Transposed to: {output.shape}")
            
            # Now output should be [num_boxes, num_features]
            num_boxes = output.shape[0]
            num_features = output.shape[1]
            
            logger.debug(f"Processing {num_boxes} boxes with {num_features} features")
            
            # First 4 features are bbox, rest are class scores
            num_classes = num_features - 4
            
            for i in range(num_boxes):
                # Extract bbox (center_x, center_y, width, height)
                bbox = output[i, :4]
                
                # Extract class scores
                class_scores = output[i, 4:]
                
                # Get best class
                class_id = np.argmax(class_scores)
                confidence = float(class_scores[class_id])
                
                # Filter by confidence
                if confidence >= self.confidence_threshold:
                    # Get label
                    label = self.labels.get(class_id, "unknown")
                    
                    # Convert bbox from center format to corner format
                    # (center_x, center_y, w, h) -> (x, y, w, h)
                    x_center, y_center, w, h = bbox
                    x = float(x_center - w / 2)
                    y = float(y_center - h / 2)
                    
                    detection = Detection(
                        label=label,
                        confidence=confidence,
                        bbox=(x, y, float(w), float(h))
                    )
                    detections.append(detection)
            
            logger.info(f"Found {len(detections)} detections above threshold")
            
        except Exception as e:
            logger.error(f"Error parsing YOLOv8 output: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return detections
    
    def parse_yolov5_output(self, outputs: List[np.ndarray]) -> List[Detection]:
        """
        Parse YOLOv5 TFLite output format
        
        YOLOv5 typically outputs: [1, num_boxes, 85]
        Where 85 = 4 (bbox) + 1 (objectness) + 80 (classes)
        """
        detections = []
        
        try:
            output = outputs[0]
            output = self.dequantize(output, self.output_details[0])
            
            logger.debug(f"Output shape: {output.shape}")
            
            # Handle shape variations
            # Skip batch dimension if present and ensure we have 2D output
            while len(output.shape) > 2:
                if output.shape[0] == 1:
                    output = output[0]
                else:
                    break
            
            # If we somehow still have 1D, reshape it
            if len(output.shape) == 1:
                logger.debug(f"1D output detected, cannot parse YOLOv5 format")
                return detections
            
            # output is now [num_boxes, 85]
            for i in range(output.shape[0]):
                # Format: [x, y, w, h, objectness, class_scores...]
                bbox = output[i, :4]
                objectness = output[i, 4]
                class_scores = output[i, 5:]
                
                # Get best class
                class_id = np.argmax(class_scores)
                class_confidence = float(class_scores[class_id])
                
                # Combined confidence
                confidence = objectness * class_confidence
                
                if confidence >= self.confidence_threshold:
                    label = self.labels.get(class_id, "unknown")
                    
                    detection = Detection(
                        label=label,
                        confidence=confidence,
                        bbox=tuple(bbox.astype(float))
                    )
                    detections.append(detection)
            
            logger.info(f"Found {len(detections)} detections above threshold")
            
        except Exception as e:
            logger.error(f"Error parsing YOLOv5 output: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return detections
    
    def non_max_suppression(self, detections: List[Detection]) -> List[Detection]:
        """Apply Non-Maximum Suppression to remove overlapping boxes"""
        if len(detections) == 0:
            return []
        
        # Convert to numpy arrays for processing
        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])
        
        # Get indices of boxes sorted by score (highest first)
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while len(indices) > 0:
            # Pick the box with highest score
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Compute IoU with remaining boxes
            current_box = boxes[current]
            other_boxes = boxes[indices[1:]]
            
            ious = self.compute_iou(current_box, other_boxes)
            
            # Keep only boxes with IoU less than threshold
            indices = indices[1:][ious < self.iou_threshold]
        
        return [detections[i] for i in keep]
    
    def compute_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute Intersection over Union"""
        # box: [x, y, w, h]
        # boxes: [n, 4]
        
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2])
        y2 = np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        area_box = box[2] * box[3]
        area_boxes = boxes[:, 2] * boxes[:, 3]
        
        union = area_box + area_boxes - intersection
        
        return intersection / (union + 1e-6)
    
    def infer(self, frame: np.ndarray) -> Detection:
        """Run inference and return best detection"""
        try:
            # Preprocess
            input_data = self.preprocess_frame(frame)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # Get all outputs
            outputs = []
            for output_detail in self.output_details:
                output_data = self.interpreter.get_tensor(output_detail['index'])
                outputs.append(output_data)
            
            # Try to determine output format and parse
            detections = []
            
            # Check output shape to determine format
            main_output = outputs[0]
            
            logger.debug(f"Main output shape: {main_output.shape}")
            
            if len(main_output.shape) == 1:
                logger.warning("Output is 1D - cannot parse. Check model output format.")
                return Detection(label="unknown", confidence=0.0)
            elif len(main_output.shape) >= 2:
                # YOLOv8 format: [1, num_features, num_boxes] or [num_features, num_boxes]
                if main_output.shape[-2] < main_output.shape[-1]:
                    detections = self.parse_yolov8_output(outputs)
                # YOLOv5 format: [1, num_boxes, num_features] or [num_boxes, num_features]
                else:
                    detections = self.parse_yolov5_output(outputs)
            
            # Apply NMS
            detections = self.non_max_suppression(detections)
            
            # Return best detection
            if detections:
                best = max(detections, key=lambda d: d.confidence)
                logger.info(f"Best detection: {best}")
                return best
            else:
                return Detection(label="unknown", confidence=0.0)
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return Detection(label="unknown", confidence=0.0)
    
    def aggregate_detections(self, detections: List[Detection]) -> Detection:
        """Aggregate multiple detections"""
        # Find heron with highest confidence
        heron_detections = [d for d in detections if d.label == "heron"]
        if heron_detections:
            return max(heron_detections, key=lambda d: d.confidence)
        
        # Find most confident non-unknown detection
        non_unknown = [d for d in detections if d.label != "unknown"]
        if non_unknown:
            return max(non_unknown, key=lambda d: d.confidence)
        
        return Detection(label="unknown", confidence=0.0)