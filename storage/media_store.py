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

# Ephemeral clip storage: subject to age retention and the hard size cap.
# Deleting the oldest clip is fine — a new one will show up on the next event.
EPHEMERAL_MEDIA_SUBDIRS = ('heron', 'other', 'unknown')

# Curated training data: NEVER deleted by retention or the size cap.
# Two failure modes make automated deletion under labeled/ actively poisonous:
#   1. The size cap sorts by mtime and stops as soon as it is under the ceiling,
#      so it deletes the 100 KB .jpg first and leaves the 50 B .txt orphaned.
#   2. Worse in the other direction: if a .jpg outlives its label, YOLO sees the
#      image with no annotation and treats it as a background (negative) example.
#      A hand-labelled heron silently becomes "no heron here" — the exact
#      poisoning the manual labelling exists to prevent.
# The labeled/ directory has its own soft warning (labeled_bytes + max_labeled_gb)
# so growth is visible without ever being unsafe.
CURATED_MEDIA_SUBDIRS = ('labeled/images', 'labeled/labels')

assert not (set(EPHEMERAL_MEDIA_SUBDIRS) & set(CURATED_MEDIA_SUBDIRS)), (
    "A subdir listed in both sets would be deletable — labeled/ must not appear "
    "in the ephemeral tuple."
)


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

    def _files_under(self, subdirs) -> List[Path]:
        files = []
        for subdir in subdirs:
            path = self.base_path / subdir
            if not path.exists():
                continue
            files.extend(f for f in path.iterdir() if f.is_file())
        return files

    def _ephemeral_files(self) -> List[Path]:
        """Files that retention and the size cap may delete.

        Explicitly excludes curated training data — see CURATED_MEDIA_SUBDIRS.
        """
        return self._files_under(EPHEMERAL_MEDIA_SUBDIRS)

    def _curated_files(self) -> List[Path]:
        return self._files_under(CURATED_MEDIA_SUBDIRS)

    def _bytes_of(self, files) -> int:
        total = 0
        for f in files:
            try:
                total += f.stat().st_size
            except OSError:
                pass  # deleted underneath us; ignore
        return total

    def total_bytes(self) -> int:
        """Bytes used across ALL media, ephemeral and curated.

        The low-disk warning is about the partition as a whole, so it counts
        labeled/ even though the cleanup paths refuse to touch it.
        """
        return self._bytes_of(self._ephemeral_files()) + self._bytes_of(self._curated_files())

    def labeled_bytes(self) -> int:
        return self._bytes_of(self._curated_files())

    def cleanup_old_media(self, days: int) -> int:
        """Delete ephemeral media older than `days`.

        labeled/ is deliberately excluded — see CURATED_MEDIA_SUBDIRS for why
        automated deletion of hand-labelled training data is unsafe.
        """
        cutoff = datetime.now().timestamp() - (days * 86400)
        removed = 0
        for f in self._ephemeral_files():
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
        """Delete oldest ephemeral media until the total is under `max_gb`.

        Retention alone cannot protect the card: at 100 events/day a 2 GB
        partition fills in about four months, long before a 365-day cutoff
        removes anything. This is the backstop that actually bounds growth.

        labeled/ is excluded — see CURATED_MEDIA_SUBDIRS. Use warn_labeled_size()
        for a visible-but-non-destructive check on that directory.
        """
        max_bytes = int(max_gb * (1024 ** 3))
        files = []
        for f in self._ephemeral_files():
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

    def warn_labeled_size(self, max_labeled_gb: float) -> bool:
        """Warn if the curated training data is over `max_labeled_gb`.

        Returns True if a warning was emitted. Never deletes anything: growth
        under labeled/ is a signal to move data off the card, not to lose it.
        """
        used_bytes = self.labeled_bytes()
        max_bytes = int(max_labeled_gb * (1024 ** 3))
        if used_bytes <= max_bytes:
            return False
        logger.warning(
            f"labeled/ is using {used_bytes / 1024 ** 3:.2f} GB, over the "
            f"{max_labeled_gb:.1f} GB soft limit. Nothing will be deleted — "
            f"move it off the card: "
            f"rsync -a {self.base_path}/labeled/ <backup>:/labeled/"
        )
        return True


    def get_file_path(self, filename: str) -> Optional[Path]:
        for subdir in ['heron', 'other', 'unknown', 'labeled/images']:
            filepath = self.base_path / subdir / filename
            if filepath.exists():
                return filepath
        return None