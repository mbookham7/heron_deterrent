#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_media_store.py
#
# These tests exist because the previous version of media_store.py silently
# treated the hand-labelled training data under labeled/ as disposable clip
# storage. Two failure modes:
#
#   1. The 365-day retention sweep would delete a labelled image (and its
#      50-byte annotation file) once it aged out.
#   2. The oldest-first size cap would delete the 100 KB .jpg before its 50 B
#      .txt, leaving orphan labels — and, in the other direction, images whose
#      label was deleted, which YOLO treats as background examples. A heron
#      image silently becomes "no heron here". Weeks of labelling ruined.
#
# Both failures were invisible until the training run. These tests fail loudly
# if labeled/ is ever reachable from either deletion path again.
#
# Run with pytest, or directly (no pytest on the board):
#     python tests/test_media_store.py
# ============================================================

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage.media_store import (  # noqa: E402
    CURATED_MEDIA_SUBDIRS,
    EPHEMERAL_MEDIA_SUBDIRS,
    MediaStore,
)


def _touch(path: Path, size_bytes: int = 1, age_days: float = 0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'x' * size_bytes)
    if age_days:
        ts = time.time() - age_days * 86400
        os.utime(path, (ts, ts))


def _labeled_pair(store: MediaStore, name: str, image_bytes: int, age_days: float = 0):
    img = store.base_path / 'labeled' / 'images' / f'{name}.jpg'
    lab = store.base_path / 'labeled' / 'labels' / f'{name}.txt'
    _touch(img, size_bytes=image_bytes, age_days=age_days)
    _touch(lab, size_bytes=50, age_days=age_days)
    return img, lab


def test_curated_and_ephemeral_are_disjoint():
    """A subdir in both tuples would defeat the whole exemption."""
    assert not (set(EPHEMERAL_MEDIA_SUBDIRS) & set(CURATED_MEDIA_SUBDIRS))


def test_labeled_dirs_are_curated_not_ephemeral():
    """The specific claim the bug rested on: labeled/ must not be in the
    deletable set. Named explicitly so a rename of the tuple contents
    without re-thinking still trips the test."""
    for sub in ('labeled/images', 'labeled/labels'):
        assert sub in CURATED_MEDIA_SUBDIRS, f"{sub} must be curated"
        assert sub not in EPHEMERAL_MEDIA_SUBDIRS, f"{sub} must NOT be ephemeral"


def test_cleanup_never_touches_labeled():
    """The retention sweep, given ancient labelled data, must leave it alone."""
    with tempfile.TemporaryDirectory() as tmp:
        store = MediaStore(tmp)
        img, lab = _labeled_pair(store, 'ancient', image_bytes=100_000, age_days=9999)
        # Something ephemeral old enough to be deleted, so the sweep actually runs.
        clip = store.base_path / 'heron' / 'ancient.mp4'
        _touch(clip, size_bytes=1000, age_days=9999)

        removed = store.cleanup_old_media(days=365)

        assert not clip.exists(), "ephemeral clip should have been deleted"
        assert img.exists(), "labelled image must survive retention"
        assert lab.exists(), "labelled annotation must survive retention"
        assert removed == 1  # only the ephemeral one


def test_size_cap_never_touches_labeled():
    """The size cap, faced with a labeled/ tree over the ceiling, must not
    touch it. Fill labeled/ well above the cap and set an aggressive limit."""
    with tempfile.TemporaryDirectory() as tmp:
        store = MediaStore(tmp)
        # 20 labelled pairs at 100 KB each = ~2 MB of images, well above the
        # 1 KB cap set below.
        pairs = [_labeled_pair(store, f'pair_{i}', image_bytes=100_000, age_days=i)
                 for i in range(20)]

        removed = store.enforce_size_cap(max_gb=1e-6)  # ~1 KB

        assert removed == 0, "size cap must not delete any curated files"
        for img, lab in pairs:
            assert img.exists(), f"{img} must survive"
            assert lab.exists(), f"{lab} must survive"


def test_size_cap_deletes_ephemeral_oldest_first():
    """Positive control: ephemeral clips still get pruned. Guards against a
    fix that accidentally exempts everything."""
    with tempfile.TemporaryDirectory() as tmp:
        store = MediaStore(tmp)
        old = store.base_path / 'heron' / 'old.mp4'
        new = store.base_path / 'heron' / 'new.mp4'
        _touch(old, size_bytes=500_000, age_days=30)
        _touch(new, size_bytes=500_000, age_days=1)

        removed = store.enforce_size_cap(max_gb=500_001 / (1024 ** 3))  # room for one

        assert removed == 1
        assert not old.exists(), "oldest ephemeral clip should have gone first"
        assert new.exists(), "newer ephemeral clip should remain"


def test_labeled_pair_stays_paired():
    """The compounding failure was orphaned images/labels. With curated data
    exempt, a labelled image and its label must stay together no matter what
    the ephemeral pressure looks like."""
    with tempfile.TemporaryDirectory() as tmp:
        store = MediaStore(tmp)
        img, lab = _labeled_pair(store, 'kept', image_bytes=200_000, age_days=1000)
        # Ephemeral pressure: 5 MB of clips, aggressive cap.
        for i in range(50):
            _touch(store.base_path / 'other' / f'clip_{i}.mp4',
                   size_bytes=100_000, age_days=i)

        store.cleanup_old_media(days=1)
        store.enforce_size_cap(max_gb=1e-6)

        assert img.exists() and lab.exists(), (
            "labelled pair must survive both cleanup paths — losing either half "
            "would poison the training set (empty label = background example)"
        )


def test_warn_labeled_size_never_deletes():
    """The soft warning on labeled/ must be observable but never destructive."""
    with tempfile.TemporaryDirectory() as tmp:
        store = MediaStore(tmp)
        img, lab = _labeled_pair(store, 'big', image_bytes=1_000_000, age_days=0)

        tripped = store.warn_labeled_size(max_labeled_gb=1e-9)

        assert tripped is True, "warning should trip when over the limit"
        assert img.exists() and lab.exists(), "warning must not delete"


def test_total_bytes_still_counts_labeled():
    """The low-disk warning is about the whole partition. Excluding labeled/
    from deletion must not exclude it from the disk-usage total."""
    with tempfile.TemporaryDirectory() as tmp:
        store = MediaStore(tmp)
        _labeled_pair(store, 'a', image_bytes=100_000)
        _touch(store.base_path / 'heron' / 'clip.mp4', size_bytes=50_000)

        assert store.total_bytes() == 100_000 + 50 + 50_000
        assert store.labeled_bytes() == 100_050


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
