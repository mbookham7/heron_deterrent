# ============================================================
# FILE: capture/camera.py
#
# cv2.VideoCapture is not thread-safe, and the dashboard's live
# stream reads the same camera as the detection pipeline, so every
# access is serialised behind one lock.
#
# A USB brownout usually leaves isOpened() returning True while
# every read() fails, so health is judged on consecutive read
# failures rather than on isOpened() alone.
# ============================================================

import logging
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Consecutive failed reads before the camera is considered gone.
# At 5 fps this is about two seconds of silence.
MAX_READ_FAILURES = 10

RECONNECT_DELAY_START = 1.0
RECONNECT_DELAY_MAX = 60.0


class Camera:
    def __init__(self, device_id: int = 0, resolution: Tuple[int, int] = (640, 480)):
        self.device_id = device_id
        self.resolution = resolution
        self.cap = None
        self._lock = threading.Lock()
        self._read_failures = 0
        self._reconnect_delay = RECONNECT_DELAY_START
        with self._lock:
            self._open()

    def _open(self):
        """Open the device. Caller must hold the lock."""
        cap = cv2.VideoCapture(self.device_id)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Could not open camera device {self.device_id} "
                f"(config.yaml -> system.camera_device_id).\n"
                f"  Is it plugged in?     v4l2-ctl --list-devices\n"
                f"  Wrong device number?  ls /dev/video*\n"
                f"  In use by something?  sudo fuser -v /dev/video{self.device_id}"
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap = cap
        self._read_failures = 0
        logger.info(f"Camera {self.device_id} open at {self.resolution[0]}x{self.resolution[1]}")

    def _close(self):
        """Release the device. Caller must hold the lock."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self.cap:
                return None

            ok, frame = self.cap.read()
            if not ok or frame is None:
                self._read_failures += 1
                if self._read_failures == MAX_READ_FAILURES:
                    # Log the transition once, not on every failed read.
                    logger.error(
                        f"Camera {self.device_id}: {self._read_failures} consecutive read "
                        f"failures — treating as disconnected. Often USB power; "
                        f"a powered hub usually fixes it."
                    )
                return None

            self._read_failures = 0
            return frame

    def is_connected(self) -> bool:
        """True only if the device is open AND actually yielding frames."""
        with self._lock:
            return (self.cap is not None
                    and self.cap.isOpened()
                    and self._read_failures < MAX_READ_FAILURES)

    def restart(self) -> bool:
        """Reconnect with exponential backoff. Returns True if the camera is back.

        The backoff sleep happens outside the lock so readers are not blocked
        for up to a minute while we wait.
        """
        with self._lock:
            self._close()

        delay = self._reconnect_delay
        logger.info(f"Reconnecting camera {self.device_id} in {delay:.0f}s...")
        time.sleep(delay)

        try:
            with self._lock:
                self._open()
        except RuntimeError as e:
            self._reconnect_delay = min(self._reconnect_delay * 2, RECONNECT_DELAY_MAX)
            logger.warning(
                f"Camera reconnect failed, next attempt in {self._reconnect_delay:.0f}s: "
                f"{str(e).splitlines()[0]}"
            )
            return False

        self._reconnect_delay = RECONNECT_DELAY_START
        logger.info("Camera reconnected")
        return True

    def release(self):
        with self._lock:
            self._close()

    def __del__(self):
        # Interpreter shutdown may already have torn down the lock.
        try:
            self.release()
        except Exception:
            pass
