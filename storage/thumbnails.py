# ============================================================
# FILE: storage/thumbnails.py
#
# Small JPEGs of one clip frame, for the dashboard's triage contact sheet.
#
# Cost, measured on the Pi 4 with 37 real 640x480 clips: a full frame from
# /api/detection/<id>/frame/<i> is ~22 KB and ~86 ms of CPU; a 320-wide
# thumbnail of frame 0 is a few KB and ~34 ms, once — after that it is a file
# read. Frame 0 also needs no seek; a middle frame costs ~3x as much because
# the decoder walks from the keyframe.
#
# Two rules keep it from competing with detection on the same board:
#   * one generation at a time, process-wide. A browser asks for a whole grid
#     at once; serialising means a cold contact sheet uses at most one of the
#     four cores, never all of them.
#   * the cache lives beside the database (data/thumbs/), not under media/:
#     it is not footage, must not count against the clip pools, and must not
#     end up in an offload pull.
# ============================================================

import logging
import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


class ThumbnailCache:
    WIDTH = 320          # tiles are ~160 CSS px; 320 stays sharp on a 2x screen
    QUALITY = 70

    def __init__(self, cache_dir: str, media_dir: str):
        # Absolute, because the config's paths are relative ('./data/') and
        # Flask's send_file resolves a relative path against the app's own
        # package directory, not the working directory.
        self.cache_dir = Path(cache_dir).resolve()
        self.media_dir = Path(media_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def path_for(self, video_path: str, frame: int) -> Path:
        # Keyed by the clip's own filename, which embeds a timestamp, not by
        # detection id: ids restart if the database is ever recreated.
        return self.cache_dir / f"{Path(video_path).stem}.f{frame}.jpg"

    def get(self, video_path: str, frame: int = 0) -> Tuple[Optional[Path], bool]:
        """(thumbnail path, whether it was generated now), or (None, False) if
        the clip cannot be read."""
        path = self.path_for(video_path, frame)
        if path.exists():
            return path, False
        with self._lock:
            if path.exists():               # another request made it meanwhile
                return path, False
            image = self._frame(video_path, frame)
            if image is None:
                return None, False
            h, w = image.shape[:2]
            small = cv2.resize(image, (self.WIDTH, round(h * self.WIDTH / w)),
                               interpolation=cv2.INTER_AREA)
            ok, buffer = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, self.QUALITY])
            if not ok:
                return None, False
            tmp = path.with_suffix('.tmp')
            tmp.write_bytes(buffer.tobytes())
            tmp.replace(path)               # never serve a half-written file
            return path, True

    @staticmethod
    def _frame(video_path: str, frame: int):
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                return None
            if frame > 0:
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(frame, count - 1)))
            ok, image = cap.read()
            return image if ok else None
        finally:
            cap.release()

    def prune(self) -> int:
        """Delete thumbnails whose clip no longer exists (retention, the size
        caps, an offload with --remove-source-files). Returns how many."""
        live = {p.stem for p in self.media_dir.rglob('*.mp4')}
        removed = 0
        for thumb in self.cache_dir.glob('*.jpg'):
            if thumb.name.rsplit('.f', 1)[0] not in live:
                thumb.unlink(missing_ok=True)
                removed += 1
        if removed:
            logger.info(f"Thumbnails: pruned {removed} whose clip is gone")
        return removed
