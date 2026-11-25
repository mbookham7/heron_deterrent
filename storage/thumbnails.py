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
# Normally the detector writes each thumbnail when it saves the clip (put()),
# from frame 0 still in memory: a resize and a JPEG encode, a few ms, with no
# video decode at all. get() decoding the clip is the fallback, for clips saved
# before this existed or whose put() failed. Measured on the Pi, a cold contact
# sheet decoded this way slowed inference from ~385 to ~560 ms for ~2 s.
#
# Two rules keep that fallback from competing with detection:
#   * one generation at a time, process-wide. A browser asks for a whole grid
#     at once; serialising means a cold contact sheet uses at most one of the
#     four cores, never all of them.
#   * the cache lives beside the database (data/thumbs/), not under media/:
#     it is not footage, must not count against the clip pools, and must not
#     end up in an offload pull.
# ============================================================

import logging
import os
import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


def _yield_to_detection():
    """Drop THIS thread to the lowest scheduling priority before decoding.

    Measured on the Pi: with generation merely serialised, a cold contact
    sheet of 38 clips raised the detector's inference from a median 386 ms to
    662 ms while it loaded. The model runs on four threads that wait for one
    another, so losing even part of one core stalls all four. Linux applies
    niceness per thread, and Flask serves each request on its own thread, so
    this lowers only the thumbnail work — never detection. Irreversible for
    the thread (raising priority again needs privileges), which is fine: the
    thread ends with the request. No-op where unsupported.
    """
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
    except (AttributeError, OSError):
        pass


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
        _yield_to_detection()
        with self._lock:
            if path.exists():               # another request made it meanwhile
                return path, False
            image = self._frame(video_path, frame)
            if image is None or not self._write(path, image):
                return None, False
            return path, True

    def put(self, video_path: str, image, frame: int = 0) -> Optional[Path]:
        """Write the thumbnail from a frame already in memory, at clip-save
        time. Runs on the detection thread, so it neither lowers the thread's
        priority nor waits for get()'s lock, and it never raises: a missing
        thumbnail only means get() makes it later."""
        if not video_path or image is None:
            return None
        path = self.path_for(video_path, frame)
        try:
            return path if self._write(path, image) else None
        except Exception as e:
            logger.warning(f"Thumbnail for {Path(video_path).name} not written: {e}")
            return None

    def _write(self, path: Path, image) -> bool:
        h, w = image.shape[:2]
        small = cv2.resize(image, (self.WIDTH, round(h * self.WIDTH / w)),
                           interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, self.QUALITY])
        if not ok:
            return False
        # Per-thread temp name: put() and get() do not share a lock.
        tmp = path.with_name(f"{path.name}.{threading.get_native_id()}.tmp")
        tmp.write_bytes(buffer.tobytes())
        tmp.replace(path)                   # never serve a half-written file
        return True

    @staticmethod
    def _frame(video_path: str, frame: int):
        # One decoder thread. FFmpeg otherwise starts one per core, so "one
        # generation at a time" still occupied several cores.
        cap = cv2.VideoCapture(str(video_path), cv2.CAP_ANY, [cv2.CAP_PROP_N_THREADS, 1])
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
