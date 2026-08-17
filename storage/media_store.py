# ============================================================
# FILE: storage/media_store.py
# ============================================================

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Every directory holding files that count towards the size cap.
MEDIA_SUBDIRS = ('heron', 'other', 'unknown', 'labeled/images', 'labeled/labels')


class MediaStore:
    def __init__(self, base_path: str, fps: float = 5.0):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        # Write clips at the rate they were captured, so they play at real speed.
        self.fps = fps

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
            out = cv2.VideoWriter(str(filepath), fourcc, self.fps, (width, height))
            
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
                
                # The model is single-class, so only heron boxes are valid
                # annotations. Anything else is written as a background image
                # (an empty label file), which is what YOLO expects for a
                # negative example — never as a fictitious class 1.
                if label == "heron":
                    f.write(f"0 {x_center} {y_center} {norm_w} {norm_h}\n")

        logger.info(f"Saved labeled data: {base_name}"
                    + ("" if label == "heron" else " (as a background/negative example)"))
        return str(image_path), str(label_path)

    def _media_files(self) -> List[Path]:
        """Every file counting towards retention and the size cap."""
        files = []
        for subdir in MEDIA_SUBDIRS:
            path = self.base_path / subdir
            if not path.exists():
                continue
            files.extend(f for f in path.iterdir() if f.is_file())
        return files

    def total_bytes(self) -> int:
        total = 0
        for f in self._media_files():
            try:
                total += f.stat().st_size
            except OSError:
                pass  # deleted underneath us; ignore
        return total

    def cleanup_old_media(self, days: int) -> int:
        """Delete media older than `days`. Covers labeled/ too, which the
        previous version skipped entirely, letting it grow without limit."""
        cutoff = datetime.now().timestamp() - (days * 86400)
        removed = 0
        for f in self._media_files():
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed += 1
            except OSError:
                pass
        if removed:
            logger.info(f"Retention: deleted {removed} file(s) older than {days} days")
        return removed

    def enforce_size_cap(self, max_gb: float) -> int:
        """Delete oldest media until the total is under `max_gb`.

        Retention alone cannot protect the card: at 100 events/day a 2 GB
        partition fills in about four months, long before a 365-day cutoff
        removes anything. This is the backstop that actually bounds growth.
        """
        max_bytes = int(max_gb * (1024 ** 3))
        files = []
        for f in self._media_files():
            try:
                files.append((f.stat().st_mtime, f.stat().st_size, f))
            except OSError:
                pass

        total = sum(size for _, size, _ in files)
        if total <= max_bytes:
            return 0

        files.sort(key=lambda item: item[0])  # oldest first
        removed = 0
        for _, size, f in files:
            if total <= max_bytes:
                break
            try:
                f.unlink()
                total -= size
                removed += 1
            except OSError:
                pass

        logger.warning(
            f"Size cap: deleted {removed} oldest file(s) to bring media under "
            f"{max_gb:.1f} GB (now {total / 1024 ** 3:.2f} GB)"
        )
        return removed


    def get_file_path(self, filename: str) -> Optional[Path]:
        for subdir in ['heron', 'other', 'unknown', 'labeled/images']:
            filepath = self.base_path / subdir / filename
            if filepath.exists():
                return filepath
        return None