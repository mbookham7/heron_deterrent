# ============================================================
# FILE: capture/camera.py
# ============================================================

import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class Camera:
    def __init__(self, device_id: int = 0, resolution: Tuple[int, int] = (640, 480)):
        self.device_id = device_id
        self.resolution = resolution
        self.cap = None
        self._initialize()
    
    def _initialize(self):
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.device_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        logger.info(f"Camera initialized: {self.resolution[0]}x{self.resolution[1]}")
    
    def read(self) -> Optional[np.ndarray]:
        if not self.cap or not self.cap.isOpened():
            logger.error("Camera not available")
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to read frame")
            return None
        
        return frame
    
    def is_connected(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
    
    def restart(self):
        logger.info("Restarting camera...")
        self.release()
        self._initialize()
    
    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def __del__(self):
        self.release()