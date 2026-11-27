# ============================================================
# FILE: storage/durable.py
#
# Make a just-written file survive a power cut.
#
# Found on the Pi (2026-11-26): detection 50's clip was 0 bytes, but its
# database row was intact. The power went soon after the clip was saved.
# ffmpeg had written the clip into the kernel's page cache and exited;
# SQLite then committed the row with fsync, as it always does. The power
# went before the kernel flushed the clip (it may wait up to ~30 s), so the
# row survived and pointed at an empty file — a clip in triage that can
# never be shown or labelled.
#
# The fix is ordering: flush the file, and the directory entry that names
# it, BEFORE anything that refers to it is committed. A new file needs its
# directory flushed too, or after a crash the name itself may be missing.
#
# Cost, measured on the Pi's SD card (2026-11-27, 20 real clips of
# 104-426 KB): median 22 ms, max 33 ms per clip, file and directory
# together. It runs after the deterrent has fired and after the clip is
# encoded (which takes seconds), so it never delays the sound.
# ============================================================

import logging
import os
from pathlib import Path
from typing import Iterable, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return                      # e.g. Windows: directories cannot be opened
    try:
        os.fsync(fd)
    except OSError:
        pass                        # some filesystems refuse fsync on a directory
    finally:
        os.close(fd)


def make_durable(*paths: PathLike) -> None:
    """fsync each file, then each distinct directory holding them. Raises
    OSError if a FILE cannot be flushed — the caller decides what that
    means; a directory that refuses is tolerated."""
    files = [Path(p) for p in paths]
    for path in files:
        with open(path, 'rb') as f:
            os.fsync(f.fileno())
    for directory in dict.fromkeys(p.parent for p in files):
        _fsync_dir(directory)
