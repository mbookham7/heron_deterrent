#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_vote_frame_types.py
#
# The camera can cut between day and IR while a clip is being captured —
# on the bench, switching the room lights off did exactly that: motion in
# day mode, an inconclusive trigger frame, then a 5 s voting clip that was
# 7 day frames and 18 IR. The voting pass used to score every key frame
# with the daylight model, IR included.
#
# These tests pin the fix: only key frames whose recorded type matches the
# event's mode are scored, and vote_k does not shrink, so a mid-clip
# switch can demote an event but never produce a heron alert.
#
# Run with pytest, or directly:
#     python tests/test_vote_frame_types.py
# ============================================================

import sys
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.detector import Detection, ObjectDetector  # noqa: E402
from utils.modes import ModeConfig  # noqa: E402

DAY = ModeConfig(name='day', model='heron', deterrent=False, cooldown_seconds=10,
                 deter_threshold=0.90, alert_threshold=0.95, record_threshold=0.30)


class EveryFrameIsAHeron:
    """The worst case: a model that calls every frame it sees a confident
    heron. Records which frames it was shown."""

    def __init__(self):
        self.seen = []

    def infer(self, frame):
        self.seen.append(frame)
        return Detection('heron', 0.99)

    aggregate_detections = ObjectDetector.aggregate_detections


def _system(detector):
    from main import HeronDeterrentSystem
    system = HeronDeterrentSystem.__new__(HeronDeterrentSystem)
    system.detectors = {'day': detector}
    system.alert_vote_k = 3
    system.alert_vote_m = 5
    system.media_store = mock.Mock()
    system.thumbnails = mock.Mock()
    system.handle_detection_result = mock.Mock()
    return system


def _verdict(system):
    return system.handle_detection_result.call_args[0][0]


def test_clean_day_clip_scores_every_key_frame():
    det = EveryFrameIsAHeron()
    system = _system(det)
    clip = list(range(25))
    system.process_detection_event(clip, DAY, {'clip_frame_types': ['day'] * 25})
    assert det.seen == [0, 5, 10, 15, 20]
    assert _verdict(system).label == 'heron'


def test_ir_frames_are_never_shown_to_the_day_model():
    """The bench case: 7 day frames, then 18 IR."""
    det = EveryFrameIsAHeron()
    system = _system(det)
    clip = list(range(25))
    meta = {'clip_frame_types': ['day'] * 7 + ['ir'] * 18}
    system.process_detection_event(clip, DAY, meta)
    assert det.seen == [0, 5]                  # key frames 10, 15, 20 are IR
    assert meta['vote_frames_skipped'] == 3


def test_a_mixed_clip_cannot_alert_on_too_few_day_frames():
    """Two day votes are not three, however confident."""
    det = EveryFrameIsAHeron()
    system = _system(det)
    system.process_detection_event(list(range(25)), DAY,
                                   {'clip_frame_types': ['day'] * 7 + ['ir'] * 18})
    assert _verdict(system).label == 'unknown'


def test_ambiguous_switching_frames_do_not_vote_either():
    det = EveryFrameIsAHeron()
    system = _system(det)
    types = ['day'] * 25
    types[10] = 'ambiguous'
    system.process_detection_event(list(range(25)), DAY, {'clip_frame_types': types})
    assert 10 not in det.seen and len(det.seen) == 4


def test_no_record_scores_every_key_frame():
    """Uploads have no per-frame record; their own gate (every key frame must
    be day) runs before they get here."""
    det = EveryFrameIsAHeron()
    system = _system(det)
    system.process_detection_event(list(range(25)), DAY, {})
    assert det.seen == [0, 5, 10, 15, 20]


def test_key_frames_unchanged_for_other_callers():
    from main import HeronDeterrentSystem
    system = HeronDeterrentSystem.__new__(HeronDeterrentSystem)
    assert system.extract_key_frames(list(range(25)), 5) == [0, 5, 10, 15, 20]
    assert system.extract_key_frames([1, 2, 3], 5) == [1, 2, 3]


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
