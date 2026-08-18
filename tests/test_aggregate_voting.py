#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_aggregate_voting.py
#
# aggregate_detections has two modes now:
#
#   mode='max'  — deter path. Best heron across frames, no threshold.
#   mode='vote' — alert path. Heron only if k-of-m frames >= alert_threshold.
#
# The old (single-mode) implementation gave a spurious high-scoring frame
# five independent chances to clear the alert bar. This test pins the fix:
# one lucky frame at 0.99 in a clip of otherwise 0.50s must NOT count as
# heron under vote mode.
#
# Run with pytest, or directly:
#     python tests/test_aggregate_voting.py
# ============================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.detector import Detection, ObjectDetector  # noqa: E402


class _StubDetector:
    """Just enough of ObjectDetector to call aggregate_detections. The method
    doesn't touch self, but instantiating the real detector needs a tflite
    interpreter — so use the unbound method directly."""
    pass


def _agg(detections, **kw):
    return ObjectDetector.aggregate_detections(_StubDetector(), detections, **kw)


def _heron(conf):
    return Detection('heron', conf)


def _unknown(conf=0.0):
    return Detection('unknown', conf)


def test_max_mode_is_unchanged_deter_path():
    """The deter path's fast/optimistic behaviour must not have shifted."""
    dets = [_heron(0.60), _heron(0.85), _unknown(0.10)]
    result = _agg(dets, mode='max')
    assert result.label == 'heron'
    assert abs(result.confidence - 0.85) < 1e-9


def test_vote_mode_rejects_single_lucky_frame():
    """The whole reason vote mode exists: five frames should not give a false
    positive five chances to win. One frame at 0.99 with the rest at 0.60
    must not clear a 3-of-5 vote at threshold 0.90."""
    dets = [_heron(0.99), _heron(0.60), _heron(0.60), _heron(0.60), _heron(0.60)]
    result = _agg(dets, mode='vote', alert_threshold=0.90, vote_k=3)
    assert result.label == 'unknown', (
        f"a single 0.99 spike must not pass a 3-of-5 vote at 0.90 — "
        f"got {result.label} conf={result.confidence}"
    )


def test_vote_mode_accepts_sustained_signal():
    """The other side: 3 of 5 above threshold should pass."""
    dets = [_heron(0.95), _heron(0.91), _heron(0.90), _heron(0.60), _heron(0.60)]
    result = _agg(dets, mode='vote', alert_threshold=0.90, vote_k=3)
    assert result.label == 'heron'
    assert result.confidence >= 0.90


def test_vote_mode_boundary_exact_k():
    """Exactly k passing frames must pass (>= not >). Off-by-one guard."""
    dets = [_heron(0.90), _heron(0.90), _heron(0.90), _heron(0.60), _heron(0.60)]
    result = _agg(dets, mode='vote', alert_threshold=0.90, vote_k=3)
    assert result.label == 'heron', "exactly k frames at exactly threshold must pass"


def test_vote_mode_one_below_k_fails():
    """k-1 passing frames must NOT pass."""
    dets = [_heron(0.95), _heron(0.90), _heron(0.60), _heron(0.60), _heron(0.60)]
    result = _agg(dets, mode='vote', alert_threshold=0.90, vote_k=3)
    assert result.label == 'unknown'


def test_vote_mode_preserves_other_known_labels():
    """If voting rejects heron but another non-unknown label is present, keep
    it — the operator still wants to see the cat detection in the dashboard."""
    dets = [_heron(0.99), Detection('cat', 0.80), _unknown()]
    result = _agg(dets, mode='vote', alert_threshold=0.90, vote_k=3)
    assert result.label == 'cat'
    assert abs(result.confidence - 0.80) < 1e-9


def test_vote_mode_empty_returns_unknown():
    """No detections at all: unknown."""
    result = _agg([], mode='vote', alert_threshold=0.90, vote_k=3)
    assert result.label == 'unknown'


def test_vote_mode_all_unknown_returns_unknown():
    result = _agg([_unknown(), _unknown(), _unknown()],
                  mode='vote', alert_threshold=0.90, vote_k=3)
    assert result.label == 'unknown'


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
