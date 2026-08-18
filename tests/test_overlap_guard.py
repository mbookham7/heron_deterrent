#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_overlap_guard.py
#
# The measurement script's overlap guard exists because the previous version
# defaulted --positives to media/heron/ and --negatives to media/, then
# silently subtracted the positives from the negatives to avoid double-count.
#
# That silent subtraction hid two errors at once:
#   1. 175 frames of the model's WORST false positives were counted as
#      positives (making apparent recall look good).
#   2. Those same frames were removed from the negatives pool (making
#      apparent precision look good).
#
# Both errors flattered the model. The guard makes overlap a hard error so
# the numbers cannot ever be reported quietly with one set inside the other
# again.
#
# Run with pytest, or directly:
#     python tests/test_overlap_guard.py
# ============================================================

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.measure_thresholds import (  # noqa: E402
    OverlappingSetsError,
    check_disjoint,
)


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'x')
    return path


def test_disjoint_directories_pass():
    """The happy path: no overlap, no exception."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pos = root / 'pos'
        neg = root / 'neg'
        pos_files = [_touch(pos / 'a.jpg')]
        neg_files = [_touch(neg / 'b.mp4')]
        # Should not raise.
        check_disjoint(pos_files, neg_files, pos, neg)


def test_same_directory_is_rejected():
    """Point --positives and --negatives at the same path -> refuse."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'shared'
        files = [_touch(root / 'a.jpg')]
        try:
            check_disjoint(files, files, root, root)
        except OverlappingSetsError as e:
            assert 'same directory' in str(e)
            return
        assert False, "expected OverlappingSetsError"


def test_negatives_nested_under_positives_is_rejected():
    """The exact shape of the previous silent-subtraction bug in reverse:
    --negatives is inside --positives. Even if no files happen to overlap
    right now, the shape is wrong."""
    with tempfile.TemporaryDirectory() as tmp:
        pos_root = Path(tmp) / 'pos'
        neg_root = pos_root / 'inner'
        pos_files = [_touch(pos_root / 'a.jpg')]
        neg_files = [_touch(neg_root / 'b.mp4')]
        try:
            check_disjoint(pos_files, neg_files, pos_root, neg_root)
        except OverlappingSetsError as e:
            assert 'nested' in str(e).lower()
            return
        assert False, "expected OverlappingSetsError"


def test_positives_nested_under_negatives_is_rejected():
    """The exact shape of the previous silent-subtraction bug:
    --positives=media/heron  --negatives=media."""
    with tempfile.TemporaryDirectory() as tmp:
        neg_root = Path(tmp) / 'media'
        pos_root = neg_root / 'heron'
        pos_files = [_touch(pos_root / 'clip.mp4')]
        # Neg files under media/ but NOT under media/heron/ — mimic media/unknown
        neg_files = [_touch(neg_root / 'unknown' / 'other.mp4')]
        try:
            check_disjoint(pos_files, neg_files, pos_root, neg_root)
        except OverlappingSetsError as e:
            msg = str(e).lower()
            assert 'nested' in msg
            assert 'silent-subtraction' in msg, (
                "guard message should recall the bug it prevents: "
                f"got {e}"
            )
            return
        assert False, "expected OverlappingSetsError"


def test_shared_file_is_rejected_even_from_unrelated_roots():
    """Two roots that are not nested but happen to include the same file via
    symlink or duplicate placement. The set intersection must catch it."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pos_root = root / 'pos'
        neg_root = root / 'neg'
        pos_root.mkdir(); neg_root.mkdir()
        shared = _touch(root / 'shared.jpg')
        try:
            check_disjoint([shared], [shared], pos_root, neg_root)
        except OverlappingSetsError as e:
            assert 'share' in str(e).lower()
            return
        assert False, "expected OverlappingSetsError"


def test_default_wiring_is_disjoint():
    """The shipping defaults must themselves pass the guard, or the very
    thing they encode (disjoint sets) is broken from day one."""
    from tools.measure_thresholds import DEFAULT_NEGATIVES, DEFAULT_POSITIVES
    pos_r = DEFAULT_POSITIVES.resolve()
    neg_r = DEFAULT_NEGATIVES.resolve()
    assert pos_r != neg_r
    assert pos_r not in neg_r.parents
    assert neg_r not in pos_r.parents


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
