#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_alert_details.py
#
# The dashboard groups identical alerts and shows each one's frame type and
# thumbnail rather than repeating "unknown detected (confidence: 0.00)".
# That needs each alert to carry what its detection was, and whether it has
# been labelled since. This pins the join.
#
# Run with pytest, or directly:
#     python tests/test_alert_details.py
# ============================================================

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage.database import Database  # noqa: E402


def test_alerts_carry_their_detection():
    with tempfile.TemporaryDirectory() as t:
        db = Database(f'{t}/heron.db')
        did = db.save_detection(label='unknown', confidence=0.0, video_path='x.mp4',
                                metadata={}, bbox=None, frame_type='ir')
        db.save_alert('WARNING', did, 'WARNING: unknown detected (confidence: 0.00)')
        db.save_alert('INFO', None, 'no detection behind this one')

        alerts = db.get_recent_alerts(10)          # same-second timestamps: order by id, not position
        with_det = next(a for a in alerts if a['detection_id'] == did)
        no_det = next(a for a in alerts if a['detection_id'] is None)
        assert with_det['det_label'] == 'unknown'
        assert with_det['det_frame_type'] == 'ir'
        assert with_det['det_confidence'] == 0.0
        assert with_det['det_user_label'] is None
        # an alert with no detection still comes back, with empty details
        assert no_det['det_label'] is None and no_det['message']

        db.update_manual_label(did, 'background')
        labelled = [a for a in db.get_recent_alerts(10) if a['detection_id'] == did][0]
        assert labelled['det_user_label'] == 'background'


def test_alert_fields_keep_their_own_names():
    """The join must not let the detection's id or timestamp overwrite the
    alert's: the dashboard keys groups by alert id."""
    with tempfile.TemporaryDirectory() as t:
        db = Database(f'{t}/heron.db')
        for _ in range(3):        # push detection ids ahead of alert ids
            db.save_detection(label='motion', confidence=0.0, video_path='y.mp4',
                              metadata={}, bbox=None, frame_type='ir')
        did = db.save_detection(label='heron', confidence=0.93, video_path='z.mp4',
                                metadata={}, bbox=None, frame_type='day')
        aid = db.save_alert('CRITICAL', did, 'CRITICAL: heron detected (confidence: 0.93)')
        a = db.get_recent_alerts(1)[0]
        assert a['id'] == aid != did
        assert a['type'] == 'CRITICAL' and a['det_confidence'] == 0.93


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
