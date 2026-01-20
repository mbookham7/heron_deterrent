# ============================================================
# FILE: capture/motion_detector.py
# ============================================================

import cv2
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class MotionDetector:
    def __init__(self, sensitivity: int = 5000):
        self.sensitivity = sensitivity
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
        logger.info(f"Motion detector initialized with sensitivity: {sensitivity}")
    
    def detect(self, frame: np.ndarray) -> bool:
        if frame is None:
            return False
        
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Remove shadows (value 127 in MOG2)
        fg_mask = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)[1]
        
        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Calculate total motion area
        motion_area = sum(cv2.contourArea(c) for c in contours)
        
        is_motion = motion_area > self.sensitivity
        
        if is_motion:
            logger.debug(f"Motion detected: area={motion_area}")
        
        return is_motion