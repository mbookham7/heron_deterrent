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


def test_endpoint_serves_with_relative_config_paths():
    """The deployed config uses relative paths (./data/, ./media/) and the
    service runs from the repo root. Flask's send_file resolves a relative
    path against the ui/ package, so a relative cache path 500s on every
    thumbnail — found with the dashboard against a copy of the Pi's data."""
    import os
    with tempfile.TemporaryDirectory() as t:
        old = os.getcwd()
        os.chdir(t)
        try:
            clip = _clip(Path('media') / 'motion' / 'motion_ir_1.mp4')
            rows = {3: {'id': 3, 'video_path': str(clip)}}   # relative, as in the DB

            class _DB:
                def get_detection(self, i): return rows.get(i)

            class _System:
                database = _DB()
                thumbnails = ThumbnailCache('data/thumbs', 'media')

            from ui import app as ui_app
            ui_app.init_app(_System())
            r = ui_app.app.test_client().get('/api/detection/3/thumb')
            assert r.status_code == 200, r.status_code
            assert r.data[:2] == b'\xff\xd8'                  # a JPEG
        finally:
            os.chdir(old)


def test_put_writes_the_same_thumbnail_without_decoding_the_clip():
    """Save time: the frame is in memory, so no decode — and get() afterwards
    is a hit, never a miss."""
    with tempfile.TemporaryDirectory() as t:
        cache = ThumbnailCache(Path(t) / 'thumbs', Path(t) / 'media')
        frame = np.full((480, 640, 3), 90, np.uint8)
        clip = Path(t) / 'media' / 'motion' / 'motion_day_1.mp4'   # never written
        path = cache.put(str(clip), frame)
        assert path == cache.path_for(str(clip), 0) and path.exists()
        assert cv2.imread(str(path)).shape[:2] == (240, 320)
        assert cache.get(str(clip)) == (path, False)
        assert not list(path.parent.glob('*.tmp'))


def test_put_stays_at_normal_priority_and_never_raises():
    """put() runs on the detection thread: it must not renice it, and a bad
    frame or an unwritable cache must not kill detection."""
    import os
    from unittest import mock
    with tempfile.TemporaryDirectory() as t:
        cache = ThumbnailCache(Path(t) / 'thumbs', Path(t) / 'media')
        with mock.patch.object(os, 'setpriority') as renice:
            cache.put('a.mp4', np.zeros((48, 64, 3), np.uint8))
        renice.assert_not_called()
        assert cache.put('', np.zeros((48, 64, 3), np.uint8)) is None
        assert cache.put('b.mp4', None) is None
        assert cache.put('c.mp4', np.zeros((0, 0, 3), np.uint8)) is None
        cache.cache_dir.chmod(0o500)
        try:
            assert cache.put('d.mp4', np.zeros((48, 64, 3), np.uint8)) is None
        finally:
            cache.cache_dir.chmod(0o700)


def test_every_saved_clip_gets_its_thumbnail_at_save_time():
    """All three save paths in main.py go through _save_clip."""
    from unittest import mock
    from main import HeronDeterrentSystem
    s = HeronDeterrentSystem.__new__(HeronDeterrentSystem)
    s.media_store = mock.Mock()
    s.thumbnails = mock.Mock()
    frames = [np.full((48, 64, 3), i, np.uint8) for i in range(3)]

    s.media_store.save_video_clip.return_value = '/m/motion_ir_1.mp4'
    assert s._save_clip(frames, 'motion', 'ir') == '/m/motion_ir_1.mp4'
    path, image = s.thumbnails.put.call_args[0]
    assert path == '/m/motion_ir_1.mp4' and image is frames[0]

    s.thumbnails.reset_mock()
    s.media_store.save_video_clip.return_value = ''        # encode failed
    assert s._save_clip(frames, 'motion', 'ir') == ''
    s.thumbnails.put.assert_not_called()

    source = (Path(__file__).resolve().parent.parent / 'main.py').read_text()
    assert source.count('media_store.save_video_clip(') == 1, \
        "a clip saved outside _save_clip gets no save-time thumbnail"


def test_endpoint_says_a_zero_byte_clip_is_unreadable_not_a_server_error():
    """Found on the Pi: a power cut mid-write left a 0-byte clip with a
    database row. It must not 500, and the answer must be cacheable."""
    with tempfile.TemporaryDirectory() as t:
        clip = Path(t) / 'media' / 'motion' / 'motion_ir_9.mp4'
        clip.parent.mkdir(parents=True)
        clip.write_bytes(b'')
        rows = {9: {'id': 9, 'video_path': str(clip)}}

        class _DB:
            def get_detection(self, i): return rows.get(i)

        class _System:
            database = _DB()
            thumbnails = ThumbnailCache(Path(t) / 'thumbs', Path(t) / 'media')

        from ui import app as ui_app
        ui_app.init_app(_System())
        r = ui_app.app.test_client().get('/api/detection/9/thumb')
        assert r.status_code == 422, r.status_code
        assert 'max-age' in r.headers.get('Cache-Control', '')


def test_gallery_clip_serves_with_relative_config_paths():
    """/api/media/<file> had the same relative-path bug as the thumbnail
    endpoint: found on the Pi when the first heron clip would not play."""
    import os
    from storage.media_store import MediaStore
    with tempfile.TemporaryDirectory() as t:
        old = os.getcwd()
        os.chdir(t)
        try:
            clip = _clip(Path('media') / 'heron' / 'heron_day_1.mp4')

            class _System:
                media_store = MediaStore('media')            # relative, as in config

            from ui import app as ui_app
            ui_app.init_app(_System())
            r = ui_app.app.test_client().get('/api/media/heron_day_1.mp4')
            assert r.status_code == 200, r.status_code
            assert r.data == clip.read_bytes()
            r = ui_app.app.test_client().get('/api/media/../../etc/passwd')
            assert r.status_code in (403, 404), r.status_code
        finally:
            os.chdir(old)


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
