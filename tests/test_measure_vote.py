#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_measure_vote.py
#
# tools/measure_thresholds.py must vote the way the device votes. It used to
# ask "do ANY k frames of the file clear the threshold", which made still
# images unvotable (alert recall 0.00 at every threshold) and made a
# 340-frame phone video pass at 0.95 when the device, voting on 5 key
# frames of a 5 s clip, had missed it. These tests pin the device-faithful
# windows, the margin on the suggested threshold, and the labelled-negatives
# query.
#
# Run with pytest, or directly:
#     python tests/test_measure_vote.py
# ============================================================

import sqlite3
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import HeronDeterrentSystem  # noqa: E402
from tools.measure_thresholds import (  # noqa: E402
    _pick_alert_threshold, iter_frames, labelled_negatives, live_windows,
    voting_recall,
)


def test_a_live_clip_is_one_window_of_the_devices_own_key_frames():
    clip = [i / 100 for i in range(25)]
    (window,) = live_windows([clip], 25, 5)
    assert window == [clip[i] for i in HeronDeterrentSystem.key_frame_indices(25, 5)]


def test_a_still_image_cannot_be_voted_on():
    assert live_windows([[0.99]], 25, 5) == []
    assert np.isnan(voting_recall([], 0.9, 3))


def test_a_long_video_is_cut_into_live_clips():
    """The case that fooled the old script: lots of frames, a few spikes.
    'Any 3 of 340 frames >= 0.95' passed; no 5 s clip's key frames do."""
    video = [0.93] * 340
    for i in (3, 101, 207, 290, 333):        # five isolated 0.95+ frames
        video[i] = 0.955
    windows = live_windows([video], 25, 5)
    assert len(windows) == 14                 # 13 full clips + a 15-frame tail
    assert voting_recall(windows, 0.95, 3) == 0.0
    assert voting_recall(windows, 0.90, 3) == 1.0


def test_a_short_tail_is_not_voted_on():
    assert len(live_windows([[0.5] * 30], 25, 5)) == 1     # 5-frame tail dropped


def test_the_suggestion_keeps_a_margin_over_the_worst_negative():
    """Without the floor the sweep picks the lowest zero-false-alert value —
    0.85 against a worst negative of 0.845 on the first field run."""
    neg = [[0.845, 0.82, 0.80, 0.83, 0.81]] * 10
    pos = [[0.93, 0.92, 0.91, 0.90, 0.95], [0.87, 0.88, 0.86, 0.89, 0.87]]
    t, _, _ = _pick_alert_threshold(pos, neg, 3, floor=0.0)
    assert t == 0.83                          # 2 negative key frames >= it: no alert
    t, rec, prec = _pick_alert_threshold(pos, neg, 3, floor=0.845 + 0.05)
    assert t == 0.9 and rec == 0.5 and prec == 1.0


def test_no_positive_windows_means_no_suggestion():
    t, rec, prec = _pick_alert_threshold([], [[0.5] * 5], 3)
    assert np.isnan(t)


def test_labelled_negatives_are_day_clips_a_person_called_not_heron():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        def clip(name, data=b'x'):
            p = t / name
            p.write_bytes(data)
            return str(p)
        db = t / 'heron.db'
        with sqlite3.connect(db) as c:
            c.execute("CREATE TABLE detections (id INTEGER PRIMARY KEY, label TEXT, "
                      "video_path TEXT, manually_labeled INTEGER, user_label TEXT, "
                      "frame_type TEXT)")
            rows = [
                (clip('bg_day.mp4'),    1, 'background', 'day'),   # yes
                (clip('person.mp4'),    1, 'person',     'day'),   # yes
                (clip('heron.mp4'),     1, 'heron',      'day'),   # a heron
                (clip('bg_ir.mp4'),     1, 'background', 'ir'),    # not day
                (clip('upload.mp4'),    0, None,         'day'),   # unlabelled
                (clip('empty.mp4', b''), 1, 'background', 'day'),  # 0 bytes
                (str(t / 'gone.mp4'),   1, 'background', 'day'),   # missing
            ]
            c.executemany("INSERT INTO detections (video_path, manually_labeled, "
                          "user_label, frame_type) VALUES (?,?,?,?)", rows)
        got = sorted(p.name for p in labelled_negatives(db))
        assert got == ['bg_day.mp4', 'person.mp4'], got


def test_a_phone_video_is_thinned_to_the_camera_rate():
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / 'phone.avi'
        out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, (64, 48))
        for _ in range(60):
            out.write(np.zeros((48, 64, 3), np.uint8))
        out.release()
        assert sum(1 for _ in iter_frames(path, 1)) == 60
        assert sum(1 for _ in iter_frames(path, 1, target_fps=5)) == 10


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
