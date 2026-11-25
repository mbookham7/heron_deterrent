#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_thumbnails.py
#
# The triage contact sheet asks for a thumbnail per clip, all at once. On
# the Pi that must not become a decode storm next to the detection loop:
# thumbnails are generated once, one at a time, and served from disk after.
# These tests pin the size, the cache, the one-at-a-time rule, the orphan
# prune, and the endpoint.
#
# Run with pytest, or directly:
#     python tests/test_thumbnails.py
# ============================================================

import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage.thumbnails import ThumbnailCache  # noqa: E402


def _clip(path: Path, w=640, h=480, n=5):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 5, (w, h))
    for i in range(n):
        writer.write(np.full((h, w, 3), 40 * i, np.uint8))
    writer.release()
    return path


def test_thumbnail_is_small_and_keeps_the_aspect():
    with tempfile.TemporaryDirectory() as t:
        clip = _clip(Path(t) / 'media' / 'motion' / 'motion_ir_1.mp4')
        cache = ThumbnailCache(Path(t) / 'thumbs', Path(t) / 'media')
        path, generated = cache.get(str(clip))
        assert generated
        img = cv2.imread(str(path))
        assert img.shape[:2] == (240, 320), img.shape     # 640x480 -> 320x240
        assert path.stat().st_size < 20_000


def test_second_request_is_a_cache_hit():
    with tempfile.TemporaryDirectory() as t:
        clip = _clip(Path(t) / 'media' / 'motion' / 'motion_ir_1.mp4')
        cache = ThumbnailCache(Path(t) / 'thumbs', Path(t) / 'media')
        first, gen1 = cache.get(str(clip))
        second, gen2 = cache.get(str(clip))
        assert gen1 and not gen2 and first == second


def test_concurrent_requests_decode_once():
    """Eight simultaneous requests for one thumbnail — what a browser does —
    must decode the clip once, not eight times."""
    with tempfile.TemporaryDirectory() as t:
        clip = _clip(Path(t) / 'media' / 'motion' / 'motion_ir_1.mp4')
        cache = ThumbnailCache(Path(t) / 'thumbs', Path(t) / 'media')
        calls = []
        real = ThumbnailCache._frame

        def slow_frame(video_path, frame):
            calls.append(frame)
            time.sleep(0.05)
            return real(video_path, frame)

        cache._frame = slow_frame
        threads = [threading.Thread(target=cache.get, args=(str(clip),)) for _ in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert len(calls) == 1, f"decoded {len(calls)} times"


def test_generation_is_one_at_a_time_across_clips():
    """Different clips still queue: at most one decode in flight, so a cold
    contact sheet uses one core, not four."""
    with tempfile.TemporaryDirectory() as t:
        media = Path(t) / 'media'
        clips = [_clip(media / 'motion' / f'motion_ir_{i}.mp4') for i in range(4)]
        cache = ThumbnailCache(Path(t) / 'thumbs', media)
        active, peak = [0], [0]
        guard = threading.Lock()
        real = ThumbnailCache._frame

        def tracked(video_path, frame):
            with guard:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            try:
                return real(video_path, frame)
            finally:
                with guard:
                    active[0] -= 1

        cache._frame = tracked
        threads = [threading.Thread(target=cache.get, args=(str(c),)) for c in clips]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert peak[0] == 1, f"{peak[0]} decodes ran at once"


def test_prune_removes_only_orphans():
    with tempfile.TemporaryDirectory() as t:
        media = Path(t) / 'media'
        keep = _clip(media / 'unknown' / 'unknown_day_1.mp4')
        gone = _clip(media / 'motion' / 'motion_ir_2.mp4')
        cache = ThumbnailCache(Path(t) / 'thumbs', media)
        kept, _ = cache.get(str(keep))
        orphan, _ = cache.get(str(gone))
        gone.unlink()                                   # retention took the clip
        assert cache.prune() == 1
        assert kept.exists() and not orphan.exists()


def test_unreadable_clip_returns_none():
    with tempfile.TemporaryDirectory() as t:
        cache = ThumbnailCache(Path(t) / 'thumbs', Path(t) / 'media')
        assert cache.get(str(Path(t) / 'nope.mp4')) == (None, False)


def test_endpoint_misses_then_hits_and_404s_a_missing_clip():
    with tempfile.TemporaryDirectory() as t:
        clip = _clip(Path(t) / 'media' / 'motion' / 'motion_ir_1.mp4')
        rows = {3: {'id': 3, 'video_path': str(clip)},
                4: {'id': 4, 'video_path': str(Path(t) / 'deleted.mp4')}}

        class _DB:
            def get_detection(self, i): return rows.get(i)

        class _System:
            database = _DB()
            thumbnails = ThumbnailCache(Path(t) / 'thumbs', Path(t) / 'media')

        from ui import app as ui_app
        ui_app.init_app(_System())
        client = ui_app.app.test_client()
        r1 = client.get('/api/detection/3/thumb')
        r2 = client.get('/api/detection/3/thumb')
        assert r1.status_code == 200 and r1.mimetype == 'image/jpeg'
        assert r1.headers['X-Thumb'] == 'miss' and r2.headers['X-Thumb'] == 'hit'
        assert client.get('/api/detection/4/thumb').status_code == 404
        assert client.get('/api/detection/99/thumb').status_code == 404


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
