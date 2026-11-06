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


# --------------------------------------------------------------------------
# Day/night split. labeled/ grew per-domain subtrees and the clip store grew a
# motion/ pool. A previous change re-included labeled/ in the sweep while
# claiming to fix it, so every new directory is checked against BOTH deletion
# paths, by name, here.
# --------------------------------------------------------------------------

DOMAIN_SUBDIRS = [f'labeled/{ft}/{kind}' for ft in ('day', 'ir', 'dark')
                  for kind in ('images', 'labels')]


def test_every_domain_subdir_is_curated_not_ephemeral():
    for sub in DOMAIN_SUBDIRS:
        assert sub in CURATED_MEDIA_SUBDIRS, f"{sub} must be curated"
        assert sub not in EPHEMERAL_MEDIA_SUBDIRS, f"{sub} must NOT be ephemeral"
    assert 'motion' in EPHEMERAL_MEDIA_SUBDIRS, "night clips must be bounded"


def test_no_deletion_path_reaches_any_domain_subdir():
    """Ancient, oversized files in every per-domain tree — plus one in a
    subdirectory nobody has named yet — survive retention and both caps."""
    with tempfile.TemporaryDirectory() as tmp:
        store = MediaStore(tmp)
        kept = []
        for sub in DOMAIN_SUBDIRS + ['labeled/ir/some_future_subdir']:
            f = store.base_path / sub / 'old.bin'
            _touch(f, size_bytes=200_000, age_days=9999)
            kept.append(f)
        for pool_dir in ('heron', 'motion'):
            _touch(store.base_path / pool_dir / 'old.mp4', size_bytes=1000, age_days=9999)

        store.cleanup_old_media(days=1)
        store.enforce_size_cap(max_gb=1e-9, pool='classified')
        store.enforce_size_cap(max_gb=1e-9, pool='motion')

        for f in kept:
            assert f.exists(), f"{f} was deleted"
        assert not (store.base_path / 'heron' / 'old.mp4').exists()
        assert not (store.base_path / 'motion' / 'old.mp4').exists()


def test_labeled_bytes_counts_every_domain_and_unnamed_subdirs():
    """The size warning must see the whole tree. A non-recursive listing would
    silently skip labeled/ir/images and report 0 bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        store = MediaStore(tmp)
        for sub in DOMAIN_SUBDIRS + ['labeled/ir/some_future_subdir']:
            _touch(store.base_path / sub / 'f.bin', size_bytes=1000)
        assert store.labeled_bytes() == 1000 * (len(DOMAIN_SUBDIRS) + 1)


def test_motion_pool_cannot_evict_classified_clips():
    """A noisy night fills motion/ far past its cap. Daylight clips — even
    older ones — must be untouched."""
    with tempfile.TemporaryDirectory() as tmp:
        store = MediaStore(tmp)
        day = store.base_path / 'unknown' / 'old_day.mp4'
        _touch(day, size_bytes=200_000, age_days=30)
        moths = [store.base_path / 'motion' / f'moth_{i}.mp4' for i in range(50)]
        for i, m in enumerate(moths):
            _touch(m, size_bytes=200_000, age_days=1 - i / 100)

        removed = store.enforce_size_cap(max_gb=1_000_001 / (1024 ** 3), pool='motion')

        assert day.exists(), "motion pool cap evicted a daylight clip"
        assert removed == 45
        assert all(m.exists() for m in moths[-5:]), "newest motion clips should remain"


def test_classified_pool_cap_ignores_motion_clips():
    with tempfile.TemporaryDirectory() as tmp:
        store = MediaStore(tmp)
        night = store.base_path / 'motion' / 'night.mp4'
        _touch(night, size_bytes=5_000_000, age_days=30)
        _touch(store.base_path / 'heron' / 'day.mp4', size_bytes=1000)

        removed = store.enforce_size_cap(max_gb=1e-3, pool='classified')

        assert removed == 0, "motion bytes must not count against the classified cap"
        assert night.exists()


def test_labelled_image_lands_in_its_domain_tree():
    frame = np.zeros((48, 64, 3), np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        store = MediaStore(tmp)
        for ft in ('day', 'ir', 'dark'):
            img, lab = store.save_labeled_image(frame, 'cat', [], detection_id=7, frame_type=ft)
            assert Path(img).parent == store.base_path / 'labeled' / ft / 'images'
            assert Path(lab).parent == store.base_path / 'labeled' / ft / 'labels'


def test_labelled_image_refuses_unsettled_or_missing_frame_type():
    """An establishing/switching clip has no domain. Filing it anywhere would
    be a guess, and a wrong guess silently contaminates a training set."""
    frame = np.zeros((48, 64, 3), np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        store = MediaStore(tmp)
        for bad in ('establishing', 'switching', None, 'night', ''):
            try:
                store.save_labeled_image(frame, 'heron', [(1, 1, 5, 5)], 1, frame_type=bad)
            except ValueError:
                continue
            raise AssertionError(f"accepted frame_type={bad!r}")
        assert not list((store.base_path / 'labeled').rglob('*.jpg'))


def test_nothing_writes_to_the_legacy_flat_layout():
    frame = np.zeros((48, 64, 3), np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        store = MediaStore(tmp)
        store.save_labeled_image(frame, 'cat', [], detection_id=1, frame_type='day')
        assert store.legacy_labeled_files() == []
        _labeled_pair(store, 'old', image_bytes=10)
        assert len(store.legacy_labeled_files()) == 2


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
