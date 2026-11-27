#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_durable.py
#
# A power cut on the Pi left a clip at 0 bytes with its database row
# intact: the row was flushed to the card, the clip was not. Clips and hand
# labels are now flushed — file and directory — before anything refers to
# them. These tests pin that the flush happens, on the right files, before
# the path is handed back, and what a failed flush does.
#
# Run with pytest, or directly:
#     python tests/test_durable.py
# ============================================================

import os
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage import durable  # noqa: E402
from storage.durable import make_durable  # noqa: E402


def test_files_then_their_directory_are_flushed():
    with tempfile.TemporaryDirectory() as t:
        a, b = Path(t) / 'a.jpg', Path(t) / 'a.txt'
        a.write_bytes(b'x'); b.write_bytes(b'')
        with mock.patch.object(durable.os, 'fsync', wraps=os.fsync) as f:
            make_durable(a, b)
        assert f.call_count == 3                 # two files, one shared directory


def test_a_directory_that_refuses_is_tolerated_but_a_file_is_not():
    with tempfile.TemporaryDirectory() as t:
        a = Path(t) / 'clip.mp4'
        a.write_bytes(b'x')
        calls = {'n': 0}

        def dir_refuses(fd):
            calls['n'] += 1
            if calls['n'] == 2:
                raise OSError('EINVAL')
        with mock.patch.object(durable.os, 'fsync', side_effect=dir_refuses):
            make_durable(a)                      # no exception
        with mock.patch.object(durable.os, 'fsync', side_effect=OSError('EIO')):
            try:
                make_durable(a)
                assert False, 'a file that cannot be flushed must raise'
            except OSError:
                pass


def _store(t):
    from storage.media_store import MediaStore
    return MediaStore(str(Path(t) / 'media'))


def _fake_encode(frames, out_path, fps):
    """Stands in for ffmpeg (absent on some dev machines): these tests are
    about the flush, not the encoding — tests/test_video_codec.py covers that."""
    Path(out_path).write_bytes(b'\x00' * 1000)


def test_a_saved_clip_is_flushed_before_its_path_is_returned():
    from storage import media_store
    with tempfile.TemporaryDirectory() as t:
        store = _store(t)
        frames = [np.full((48, 64, 3), i * 20, np.uint8) for i in range(5)]
        with mock.patch.object(media_store, 'encode_h264', _fake_encode), \
                mock.patch.object(media_store, 'make_durable', wraps=make_durable) as md:
            path = store.save_video_clip(frames, 'motion', frame_type='ir')
        assert path and Path(path).stat().st_size > 0
        md.assert_called_once_with(Path(path))


def test_a_clip_whose_flush_fails_is_still_kept():
    """The clip is complete; failing to flush it is logged, not fatal —
    returning '' would lose it for certain."""
    from storage import media_store
    with tempfile.TemporaryDirectory() as t:
        store = _store(t)
        frames = [np.zeros((48, 64, 3), np.uint8)] * 5
        with mock.patch.object(media_store, 'encode_h264', _fake_encode), \
                mock.patch.object(media_store, 'make_durable', side_effect=OSError('EIO')):
            path = store.save_video_clip(frames, 'motion', frame_type='ir')
        assert path and Path(path).exists()


def test_label_files_are_flushed_and_a_failure_refuses_the_label():
    from storage import media_store
    with tempfile.TemporaryDirectory() as t:
        store = _store(t)
        img = np.zeros((480, 640, 3), np.uint8)
        with mock.patch.object(media_store, 'make_durable', wraps=make_durable) as md:
            image_path, label_path = store.save_labeled_image(
                img, 'heron', [(10, 20, 30, 40)], 7, frame_type='day')
        md.assert_called_once_with(Path(image_path), Path(label_path))
        with mock.patch.object(media_store, 'make_durable', side_effect=OSError('EIO')):
            try:
                store.save_labeled_image(img, 'background', [], 8, frame_type='day')
                assert False, 'an unflushed label must not be reported as saved'
            except OSError:
                pass


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
