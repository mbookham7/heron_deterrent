# ============================================================
# FILE: storage/media_store.py
# ============================================================

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class MediaStore:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ['heron', 'other', 'unknown', 'labeled']:
            (self.base_path / subdir).mkdir(exist_ok=True)
    
    def save_video_clip(self, frames: List[np.ndarray], label: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{label}_{timestamp}.mp4"
        
        # Determine subdirectory
        if label == "heron":
            subdir = "heron"
        elif label == "unknown":
            subdir = "unknown"
        else:
            subdir = "other"
        
        filepath = self.base_path / subdir / filename
        
        try:
            if len(frames) == 0:
                logger.warning("No frames to save")
                return str(filepath)
            
            # Get frame dimensions
            height, width = frames[0].shape[:2]
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(filepath), fourcc, 10.0, (width, height))
            
            for frame in frames:
                out.write(frame)
            
            out.release()
            logger.info(f"Saved video: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save video: {e}")
            return ""
    
    def save_labeled_image(self, image: np.ndarray, label: str, 
                          bboxes: List[Tuple], detection_id: int) -> Tuple[str, str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = f"labeled_{detection_id}_{timestamp}"
        
        # Save image
        image_dir = self.base_path / "labeled" / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{base_name}.jpg"
        cv2.imwrite(str(image_path), image)
        
        # Save YOLO format label
        label_dir = self.base_path / "labeled" / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        label_path = label_dir / f"{base_name}.txt"
        
        height, width = image.shape[:2]
        
        with open(label_path, 'w') as f:
            for bbox in bboxes:
                x, y, w, h = bbox
                # Convert to YOLO format (normalized center coordinates)
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                norm_w = w / width
                norm_h = h / height
                
                # Class ID (0 for heron in this example)
                class_id = 0 if label == "heron" else 1
                
                f.write(f"{class_id} {x_center} {y_center} {norm_w} {norm_h}\n")
        
        logger.info(f"Saved labeled data: {base_name}")
        return str(image_path), str(label_path)
    
    def cleanup_old_media(self, days: int):
        cutoff = datetime.now().timestamp() - (days * 86400)
        
        for subdir in ['heron', 'other', 'unknown']:
            path = self.base_path / subdir
            if path.exists():
                for file in path.glob('*.mp4'):
                    if file.stat().st_mtime < cutoff:
                        file.unlink()
                        logger.info(f"Deleted old file: {file}")
    
    def get_file_path(self, filename: str) -> Optional[Path]:
        for subdir in ['heron', 'other', 'unknown', 'labeled/images']:
            filepath = self.base_path / subdir / filename
            if filepath.exists():
                return filepath
        return None