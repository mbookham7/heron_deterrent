# ============================================================
# FILE: storage/video_encoder.py
#
# Why this module exists
# ----------------------
# Clips used to be written with cv2.VideoWriter(*'mp4v'), which produces
# MPEG-4 Part 2 (the DivX/Xvid-era codec) inside an .mp4 container. That is a
# valid file — the browser parses it, reads the duration and draws the playback
# controls — and then cannot decode a single frame, because no browser has ever
# supported MPEG-4 Part 2 in <video>. The gallery showed working controls over a
# black rectangle, with no error anywhere.
#
# The failure was silent in both directions, which is the real lesson here:
#
#   * cv2.VideoWriter returns isOpened() == True for mp4v and writes a
#     well-formed file, so nothing downstream could tell something was wrong.
#   * Asking OpenCV for a codec it cannot provide does NOT raise. On the Coral,
#     VideoWriter_fourcc(*'MJPG') silently falls back to mp4v, and every H.264
#     spelling ('avc1'/'H264'/'x264') fails to open and leaves a 0-byte file
#     behind. A writer that "succeeds" into an unplayable file is exactly what
#     this module refuses to do.
#
# Why ffmpeg rather than cv2
# --------------------------
# Measured on the Coral Dev Board (Mendel 5, pip-built OpenCV 4.5.4):
#
#   * H.264 via cv2 is IMPOSSIBLE. The only h264 encoder registered in that
#     build is h264_v4l2m2m (the i.MX8 hardware encoder), which fails with
#     "Could not find a valid device". There is no libx264 in the build.
#   * VP9 via cv2 works but costs ~1014 ms/frame — 25 s to encode a 5 s clip,
#     the same order as inference (~942 ms/frame). It would starve the
#     detection semaphore.
#   * libx264 via an ffmpeg subprocess costs ~39 ms/frame at preset veryfast:
#     0.99 s for a 5 s clip, and a smaller file than the mp4v it replaces.
#
# So frames are piped as raw BGR to ffmpeg. Deliberate encoder choices:
#
#   -pix_fmt yuv420p    Some browsers refuse yuv444p/yuv422p outright.
#   -movflags +faststart Moves the moov atom to the front. Without it the
#                       index sits at the end of the file and playback cannot
#                       start until the whole clip has downloaded.
#   -profile:v baseline Keeps the codecs= string deterministic (avc1.42E01E)
#   -level 3.0          so the UI can declare it precisely and let the browser
#                       reject an unsupported file up front instead of
#                       failing quietly after the fact.
# ============================================================

import logging
import shutil
import struct
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Sample-entry fourccs that browsers can actually decode in an HTML5 <video>.
# MPEG-4 Part 2 ('mp4v') is deliberately absent — that is the whole bug.
BROWSER_PLAYABLE_CODECS = frozenset({'avc1', 'avc3', 'vp08', 'vp09', 'av01'})

# Matches -profile:v baseline -level 3.0 below. Declared in the <video> tag so
# an unsupported file is rejected before playback rather than after.
H264_MIME_TYPE = 'video/mp4; codecs="avc1.42E01E"'

FFMPEG_PRESET = 'veryfast'   # fastest AND smallest of the presets measured

_MISSING_FFMPEG_HINT = (
    "ffmpeg is not installed, so clips cannot be written in a format browsers "
    "can play. Install it:  sudo apt-get install -y ffmpeg"
)


def _discard(path: Path) -> None:
    """Remove a partial output. Mendel ships Python 3.7, which has no
    Path.unlink(missing_ok=...) — that arrived in 3.8."""
    try:
        path.unlink()
    except OSError:
        pass


class VideoEncoderUnavailable(RuntimeError):
    """Raised at startup when no browser-playable encoder is available.

    Deliberately fatal rather than a fallback to mp4v: silently writing
    unplayable clips is the failure this module exists to prevent.
    """


def ffmpeg_path() -> Optional[str]:
    return shutil.which('ffmpeg')


def encoder_status() -> Tuple[bool, str]:
    """(usable, human-readable detail). Shared by startup and preflight."""
    exe = ffmpeg_path()
    if not exe:
        return False, _MISSING_FFMPEG_HINT

    try:
        proc = subprocess.run([exe, '-hide_banner', '-encoders'],
                              capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"{exe} did not respond to -encoders: {e}"

    if 'libx264' not in proc.stdout:
        return False, (
            f"{exe} has no libx264 encoder, so it cannot produce H.264. "
            f"Install a build that includes it: sudo apt-get install -y ffmpeg"
        )

    return True, f"{exe} with libx264 (preset {FFMPEG_PRESET})"


def require_encoder() -> str:
    """Assert a browser-playable encoder exists, or raise loudly.

    Call once at startup. Never falls back — an unplayable clip that looks
    like a success is worse than refusing to start.
    """
    ok, detail = encoder_status()
    if not ok:
        raise VideoEncoderUnavailable(detail)
    return detail


def encode_h264(frames: List[np.ndarray], out_path: str, fps: float,
                preset: str = FFMPEG_PRESET, timeout: int = 120) -> None:
    """Encode BGR frames to browser-playable H.264. Raises on any failure.

    Never leaves a partial file behind: a failed encode unlinks its output so
    the gallery cannot pick up a truncated clip and render another black frame.
    """
    if not frames:
        raise ValueError("no frames to encode")

    exe = ffmpeg_path()
    if not exe:
        raise VideoEncoderUnavailable(_MISSING_FFMPEG_HINT)

    height, width = frames[0].shape[:2]
    out = Path(out_path)

    cmd = [
        exe, '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}', '-r', str(fps),
        '-i', '-',
        '-an',
        '-c:v', 'libx264', '-preset', preset,
        '-profile:v', 'baseline', '-level', '3.0',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        str(out),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            # A frame of the wrong size would desynchronise the raw stream and
            # produce a skewed clip rather than an error, so reject it here.
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"frame size {frame.shape[1]}x{frame.shape[0]} does not match "
                    f"the first frame's {width}x{height}"
                )
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read().decode('utf-8', 'replace').strip()
        rc = proc.wait(timeout=timeout)
    except BrokenPipeError:
        stderr = proc.stderr.read().decode('utf-8', 'replace').strip()
        rc = proc.wait(timeout=timeout)
        if rc == 0:
            rc = 1
    except Exception:
        proc.kill()
        proc.wait(timeout=10)
        _discard(out)
        raise
    finally:
        for stream in (proc.stdin, proc.stderr):
            try:
                if stream and not stream.closed:
                    stream.close()
            except OSError:
                pass

    if rc != 0:
        _discard(out)
        raise RuntimeError(f"ffmpeg exited {rc}: {stderr or '(no stderr)'}")

    if not out.exists() or out.stat().st_size == 0:
        _discard(out)
        raise RuntimeError("ffmpeg reported success but produced no output")


def probe_mp4_codec(path) -> Optional[str]:
    """Return the video sample-entry fourcc of an MP4, or None.

    Parses the container directly instead of shelling out to ffprobe, so tests
    can assert on the codec on a developer machine that has no ffmpeg at all —
    which is precisely where a regression would otherwise slip through.
    """
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None

    def walk(buf: bytes, want: bytes) -> Optional[bytes]:
        i = 0
        while i + 8 <= len(buf):
            size = struct.unpack('>I', buf[i:i + 4])[0]
            box_type = buf[i + 4:i + 8]
            header = 8
            if size == 1:                      # 64-bit extended size
                if i + 16 > len(buf):
                    break
                size = struct.unpack('>Q', buf[i + 8:i + 16])[0]
                header = 16
            elif size == 0:                    # box runs to end of file
                size = len(buf) - i
            if size < header or i + size > len(buf) + 8:
                break
            payload = buf[i + header:i + size]
            if box_type == want:
                return payload
            # Only these boxes can contain stsd; recursing into mdat would be
            # both pointless and slow.
            if box_type in (b'moov', b'trak', b'mdia', b'minf', b'stbl'):
                found = walk(payload, want)
                if found is not None:
                    return found
            i += size
        return None

    stsd = walk(data, b'stsd')
    if not stsd or len(stsd) < 16:
        return None
    # stsd: 4B version/flags, 4B entry_count, then [4B size][4B format]...
    return stsd[12:16].decode('ascii', 'replace')


def is_browser_playable(path) -> bool:
    return probe_mp4_codec(path) in BROWSER_PLAYABLE_CODECS
