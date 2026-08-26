#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_video_codec.py
#
# These tests exist because a recorded clip can be a perfectly valid .mp4 and
# still be undecodable by every browser on earth.
#
# The bug: clips were written with cv2.VideoWriter(*'mp4v'), which produces
# MPEG-4 Part 2 (DivX/Xvid era). The container is fine, so the browser parsed
# it, read the duration, drew the playback controls — and then rendered a black
# rectangle forever. No exception, no log line, no failed request. The only
# symptoms were visual, and only if you happened to press play.
#
# So "the file was written" and "nb_frames looks right" are both worthless as
# assertions here. The only thing worth asserting is the CODEC, which is why
# these tests read the sample-entry fourcc out of the container itself.
#
# Note the deliberate split: the test that guards the classification logic
# needs no ffmpeg, because a developer machine without ffmpeg is exactly where
# a regression would otherwise pass unnoticed.
#
# Run with pytest, or directly (no pytest on the board):
#     python tests/test_video_codec.py
# ============================================================

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage.media_store import MediaStore  # noqa: E402
from storage.video_encoder import (  # noqa: E402
    BROWSER_PLAYABLE_CODECS,
    H264_MIME_TYPE,
    encode_h264,
    encoder_status,
    is_browser_playable,
    probe_mp4_codec,
)


def _frames(n=10, w=64, h=48):
    """Gradient frames — cheap to encode and not degenerate like a flat colour."""
    out = []
    for i in range(n):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        f[:, :, 0] = (np.arange(w) + i * 4) % 256
        f[:, :, 1] = (np.arange(h)[:, None] + i * 2) % 256
        f[:, :, 2] = (i * 8) % 256
        out.append(f)
    return out


def _require_ffmpeg():
    ok, detail = encoder_status()
    if not ok:
        raise unittest.SkipTest(f"needs ffmpeg with libx264: {detail}")


# --------------------------------------------------------------------------
# These run everywhere — no ffmpeg required. They guard the classification
# logic, which is the part that decided a broken file was acceptable.
# --------------------------------------------------------------------------

def test_mp4v_is_not_browser_playable():
    """The exact codec that caused the bug must never be accepted again.

    Written with cv2 rather than hard-coded, so if someone reinstates the old
    writer this fails on a real file rather than on a constant.
    """
    import cv2

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / 'legacy.mp4'
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (64, 48))
        assert vw.isOpened(), "cv2 could not open an mp4v writer to build the fixture"
        for f in _frames():
            vw.write(f)
        vw.release()

        assert out.exists() and out.stat().st_size > 0, (
            "mp4v produced no file — the fixture itself is broken"
        )
        # The heart of the bug: a perfectly well-formed file that cannot play.
        assert probe_mp4_codec(out) == 'mp4v'
        assert not is_browser_playable(out), (
            "mp4v (MPEG-4 Part 2) is not decodable by any browser and must "
            "never be classified as playable"
        )


def test_mp4v_absent_from_the_playable_set():
    """Named explicitly, so removing it from the set is a deliberate act."""
    assert 'mp4v' not in BROWSER_PLAYABLE_CODECS
    assert 'avc1' in BROWSER_PLAYABLE_CODECS


def test_declared_mime_type_matches_the_encoder_profile():
    """The <video> type= string must describe what we actually encode.

    Declaring plain video/mp4 is what let the browser accept an undecodable
    file. The codecs= parameter is the part that makes rejection happen up
    front, so it has to track -profile:v baseline -level 3.0.
    """
    assert 'avc1.42E01E' in H264_MIME_TYPE   # baseline profile, level 3.0
    assert H264_MIME_TYPE.startswith('video/mp4;')


def test_probe_returns_none_for_a_non_mp4():
    with tempfile.TemporaryDirectory() as tmp:
        junk = Path(tmp) / 'not-a-video.mp4'
        junk.write_bytes(b'this is not an MP4 at all')
        assert probe_mp4_codec(junk) is None
        assert not is_browser_playable(junk)


def test_probe_returns_none_for_a_missing_file():
    assert probe_mp4_codec(Path('/nonexistent/never.mp4')) is None


# --------------------------------------------------------------------------
# These need a working H.264 encoder, so they skip on a machine without one.
# --------------------------------------------------------------------------

def test_saved_clip_uses_a_browser_playable_codec():
    """THE regression test: what MediaStore actually writes must be playable."""
    _require_ffmpeg()

    with tempfile.TemporaryDirectory() as tmp:
        store = MediaStore(tmp, fps=5.0)
        path = store.save_video_clip(_frames(10), 'heron')

        assert path, "save_video_clip returned empty — encode failed"
        clip = Path(path)
        assert clip.exists() and clip.stat().st_size > 0

        codec = probe_mp4_codec(clip)
        assert codec in BROWSER_PLAYABLE_CODECS, (
            f"clip was written as {codec!r}, which browsers cannot play. "
            f"Playback controls will render over a black frame with no error."
        )
        assert codec == 'avc1', f"expected H.264/avc1, got {codec!r}"


def test_encoded_clip_honours_the_requested_fps():
    """The legacy clips claim 10 fps for 5 fps footage, so they play at double
    speed and report half the real duration. The writer must pass through
    system.frame_rate rather than a hard-coded default."""
    _require_ffmpeg()
    if not __import__('shutil').which('ffprobe'):
        raise unittest.SkipTest("needs ffprobe to read container metadata")

    with tempfile.TemporaryDirectory() as tmp:
        store = MediaStore(tmp, fps=5.0)
        path = store.save_video_clip(_frames(25), 'unknown')

        rate = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0:nk=1', path],
            capture_output=True, text=True, timeout=30).stdout.strip()
        duration = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'csv=p=0:nk=1', path],
            capture_output=True, text=True, timeout=30).stdout.strip()

        assert rate == '5/1', f"expected 5/1, got {rate!r}"
        # 25 frames at 5 fps is 5 seconds of wall clock, not 2.5.
        assert abs(float(duration) - 5.0) < 0.25, f"duration {duration} != ~5.0s"


def test_encode_failure_leaves_no_partial_file():
    """A truncated clip in the gallery is another black rectangle. A failed
    encode must remove its own output rather than leave one behind."""
    _require_ffmpeg()

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / 'mismatched.mp4'
        bad = _frames(3) + [np.zeros((99, 99, 3), dtype=np.uint8)]

        try:
            encode_h264(bad, str(out), fps=5.0)
            raise AssertionError("expected a mismatched frame size to raise")
        except ValueError:
            pass

        assert not out.exists(), "a failed encode must not leave a partial file"


def test_empty_frame_list_is_refused():
    try:
        encode_h264([], '/tmp/never-written.mp4', fps=5.0)
        raise AssertionError("expected empty frame list to raise")
    except ValueError:
        pass


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = skipped = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except unittest.SkipTest as e:
            skipped += 1
            print(f"SKIP {t.__name__}: {e}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed - skipped}/{len(tests)} passed, {skipped} skipped")
    sys.exit(1 if failed else 0)
