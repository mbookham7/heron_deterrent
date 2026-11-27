#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_upload.py
#
# The test-video upload, found wanting on the Pi with two phone videos of a
# heron: both came back "unknown, 0.00" although the model scored their key
# frames 0.92-0.95 — just under the 0.95 alert threshold. The verdict was
# right by the rules; the report hid it. And each 1080x1920 30 fps video was
# held whole in memory (~2 GB), re-encoded at full size, and saved at 5 fps.
#
# These pin: the voting pass reports every key frame's score and stores the
# best one on an 'unknown' verdict, WITHOUT changing the verdict; and uploads
# are decoded at the camera's size and frame rate.
#
# Run with pytest, or directly:
#     python tests/test_upload.py
# ============================================================

import sys
import tempfile
import unittest.mock as mock
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.detector import Detection, ObjectDetector  # noqa: E402
from utils.modes import ModeConfig  # noqa: E402

DAY = ModeConfig(name='day', model='heron', deterrent=False, cooldown_seconds=10,
                 deter_threshold=0.90, alert_threshold=0.95, record_threshold=0.30)


class Scripted:
    """Returns the given heron scores in turn (0 = nothing found)."""

    def __init__(self, scores):
        self.scores = list(scores)

    def infer(self, frame):
        s = self.scores.pop(0)
        return Detection('heron', s, (0.1, 0.1, 0.2, 0.2)) if s else Detection('unknown', 0.0)

    aggregate_detections = ObjectDetector.aggregate_detections


def _system(scores):
    from main import HeronDeterrentSystem
    system = HeronDeterrentSystem.__new__(HeronDeterrentSystem)
    system.detectors = {'day': Scripted(scores)}
    system.alert_vote_k = 3
    system.alert_vote_m = 5
    system.media_store = mock.Mock()
    system.media_store.save_video_clip.return_value = '/m/clip.mp4'
    system.thumbnails = mock.Mock()
    system.handle_detection_result = mock.Mock()
    return system


def test_a_near_miss_stays_unknown_but_keeps_its_best_score():
    """Upload 2's real key-frame scores."""
    system = _system([0.946, 0.933, 0.0, 0.927, 0.851])
    meta = {'source': 'upload'}
    result = system.process_detection_event(list(range(25)), DAY, meta)

    verdict = system.handle_detection_result.call_args[0][0]
    assert verdict.label == 'unknown'                      # the rules did not move
    assert abs(verdict.confidence - 0.946) < 1e-9          # but 0.00 no longer hides it
    assert meta['vote'] == {'scores': [0.946, 0.933, 0.0, 0.927, 0.851], 'votes': 0,
                            'needed': 3, 'alert_threshold': 0.95,
                            'at_deter': 3, 'deter_threshold': 0.90}
    assert result['label'] == 'unknown' and result['video_path'] == '/m/clip.mp4'
    assert system.media_store.save_video_clip.call_args[0][1] == 'unknown'


def test_a_confirmed_heron_is_unchanged():
    system = _system([0.97, 0.96, 0.0, 0.99, 0.5])
    result = system.process_detection_event(list(range(25)), DAY, None)
    verdict = system.handle_detection_result.call_args[0][0]
    assert verdict.label == 'heron' and verdict.confidence == 0.99
    assert result['votes'] == 3 and result['at_deter'] == 3


def test_nothing_seen_still_reads_zero():
    system = _system([0.0] * 5)
    result = system.process_detection_event(list(range(25)), DAY, {})
    assert result['label'] == 'unknown' and result['confidence'] == 0.0


def _video(path, w, h, fps, seconds):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
    for i in range(int(fps * seconds)):
        f = np.full((h, w, 3), 40, np.uint8)
        cv2.putText(f, str(i), (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
        out.write(f)
    out.release()


def test_a_portrait_phone_video_is_loaded_like_the_camera_would_see_it():
    from ui.app import load_upload
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / 'phone.avi'
        _video(path, 1080, 1920, 30, 3)
        frames, info = load_upload(str(path), 5)
        assert info['frames_read'] == 90
        assert len(frames) == 15 and info['frames_processed'] == 15   # 3 s at 5 fps
        assert info['seconds'] == 3.0
        assert frames[0].shape == (640, 360, 3)                       # long side 640
        assert info['source_size'] == [1080, 1920] and info['size'] == [360, 640]
        assert not info['truncated']


def test_a_long_video_is_cut_and_says_so():
    from ui.app import load_upload
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / 'long.avi'
        _video(path, 320, 240, 10, 4)
        frames, info = load_upload(str(path), 5, max_seconds=2)
        assert info['truncated'] and len(frames) == 10
        assert frames[0].shape == (240, 320, 3)                       # small: untouched


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
