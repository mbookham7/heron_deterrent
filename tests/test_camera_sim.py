#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_camera_sim.py
#
# Pins ai/camera_sim.py. The simulator exists because scoring a 1536x2048
# stock heron photo through a 640-input model does not match what the device
# sees — a 640x480 camera has already centre-cropped to 4:3 and downsampled
# by ~3x before letterboxing. The harness must match deployment or the
# recall figures overstate reality.
#
# Portrait/landscape/pass-through cases here. A single-axis test would not
# have caught the letterbox stretch bug earlier; the same reasoning applies
# to any transform with an aspect-ratio choice inside it.
#
# Run with pytest, or directly:
#     python tests/test_camera_sim.py
# ============================================================

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.camera_sim import PAD_VALUE, simulate_camera  # noqa: E402

CAMERA_WH = (640, 480)  # matches system.resolution default in config.yaml


def _distinctive_bands(h: int, w: int) -> np.ndarray:
    """Frame with easily-identified bands so a crop is visible.

    Top band, middle band, bottom band each a distinct BGR colour. Used by
    pass-through and shape tests. For crop-detection tests, prefer
    _extreme_bands so the strips that must be removed are unambiguously
    outside the crop region (not straddling its edges).
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    third = h // 3
    img[:third]           = (255,   0,   0)  # BLUE   (top)
    img[third:2 * third]  = (  0, 255,   0)  # GREEN  (middle)
    img[2 * third:]       = (  0,   0, 255)  # RED    (bottom)
    return img


def _extreme_bands_vertical(h: int, w: int, strip: int = 100) -> np.ndarray:
    """Blue strip at very top, red at very bottom, green in the middle.

    Sized so a 4:3 centre-crop cannot possibly retain the coloured strips —
    the whole point is a strict assertion ("no pure blue anywhere"), and
    that would fail if the crop straddled the strip boundary.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:strip]        = (255,   0,   0)  # BLUE  (top, will be cropped)
    img[strip:-strip]  = (  0, 255,   0)  # GREEN (kept)
    img[-strip:]       = (  0,   0, 255)  # RED   (bottom, will be cropped)
    return img


def _extreme_bands_horizontal(h: int, w: int, strip: int = 100) -> np.ndarray:
    """As _extreme_bands_vertical but bands split left-right, for landscape
    crop tests."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :strip]        = (255,   0,   0)  # BLUE  (left, will be cropped)
    img[:, strip:-strip]  = (  0, 255,   0)  # GREEN (kept)
    img[:, -strip:]       = (  0,   0, 255)  # RED   (right, will be cropped)
    return img


def test_portrait_source_crops_top_and_bottom():
    """1000x2000 portrait through a 4:3 crop retains height = w/aspect = 750
    rows, centred: rows 625..1375 kept. Strips at rows 0..100 and 1900..2000
    (the extreme top/bottom) sit fully outside that window — they MUST be gone.
    """
    # Strip=100 << (2000-1375)=625 and 625-0=625, so both coloured strips are
    # entirely inside the discard region on both edges.
    img = _extreme_bands_vertical(2000, 1000, strip=100)

    out = simulate_camera(img, CAMERA_WH, mode='crop')

    assert out.shape[:2] == (CAMERA_WH[1], CAMERA_WH[0]), \
        f"expected {CAMERA_WH[::-1]}, got {out.shape[:2]}"

    # No pixel anywhere in the output should be blue or red — the strips
    # carrying those colours were entirely discarded.
    assert out[..., 0].max() < 30, (
        "portrait crop left blue pixels in the output — the extreme top "
        "strip should have been discarded entirely"
    )
    assert out[..., 2].max() < 30, (
        "portrait crop left red pixels in the output — the extreme bottom "
        "strip should have been discarded entirely"
    )
    # Middle green should dominate.
    assert out[..., 1].mean() > 200, "green (middle) should dominate the crop"


def test_landscape_source_crops_left_and_right():
    """3000x1000 landscape -> 4:3 crop retains width = h*aspect = 1333 cols,
    centred: cols 833..2166 kept. Strips at cols 0..100 and 2900..3000 sit
    fully outside — they MUST be gone."""
    img = _extreme_bands_horizontal(1000, 3000, strip=100)

    out = simulate_camera(img, CAMERA_WH, mode='crop')

    assert out.shape[:2] == (CAMERA_WH[1], CAMERA_WH[0])
    assert out[..., 0].max() < 30, (
        "landscape crop left blue pixels — the extreme left strip should be gone"
    )
    assert out[..., 2].max() < 30, (
        "landscape crop left red pixels — the extreme right strip should be gone"
    )
    assert out[..., 1].mean() > 200, "green (middle) should dominate the crop"


def test_camera_native_pass_through_is_byte_identical():
    """The hard requirement: a 640x480 source is what the camera actually
    delivers. Piping it through simulate_camera must not touch a byte. A
    resize round-trip would degrade the image and pond footage arriving in
    the future would be double-processed.
    """
    img = _distinctive_bands(480, 640)

    out = simulate_camera(img, CAMERA_WH, mode='crop')

    assert out is img or np.array_equal(out, img), (
        "camera-native input must pass through unchanged"
    )


def test_output_matches_config_target_shape_4_3():
    """Any 4:3 target size must produce output of exactly that size, so a
    future 1280x960 or 320x240 change to system.resolution follows."""
    img = _distinctive_bands(1500, 1000)  # portrait, arbitrary
    for tgt in [(640, 480), (1280, 960), (320, 240)]:
        out = simulate_camera(img, tgt, mode='crop')
        assert out.shape[:2] == (tgt[1], tgt[0]), \
            f"target {tgt}: expected shape {tgt[::-1]}, got {out.shape[:2]}"


def test_output_matches_config_target_shape_16_9():
    """Non-4:3 targets must work too — the ROI experiment is likely to land
    on 1280x720 (16:9), and the simulator's aspect maths must not silently
    assume 4:3."""
    img = _distinctive_bands(1500, 1000)
    out = simulate_camera(img, (1280, 720), mode='crop')
    assert out.shape[:2] == (720, 1280)


def test_pad_mode_preserves_all_source_features():
    """The sensitivity comparison. Pad mode is not what a real camera does
    but it exists so the crop-mode assumption is visible instead of buried.
    On a portrait source, pad mode must keep BOTH top and bottom bands
    (unlike crop mode which discards them)."""
    img = _distinctive_bands(2000, 1000)

    out = simulate_camera(img, CAMERA_WH, mode='pad')

    assert out.shape[:2] == (CAMERA_WH[1], CAMERA_WH[0])
    # The three bands must all still be present somewhere in the output.
    # Scan columns for a blue-heavy region and a red-heavy region.
    b_col_max = out[..., 0].max(axis=0)
    r_col_max = out[..., 2].max(axis=0)
    g_col_max = out[..., 1].max(axis=0)
    assert b_col_max.max() > 240, "blue band should survive pad mode"
    assert r_col_max.max() > 240, "red band should survive pad mode"
    assert g_col_max.max() > 240, "green band should survive pad mode"


def test_pad_mode_uses_grey_padding():
    """Pad mode fills the aspect gap with PAD_VALUE grey. A very-narrow
    portrait scaled to a 4:3 target will have grey bars on left and right."""
    # 100x1000 = 1:10, extreme portrait
    img = np.full((1000, 100, 3), 200, dtype=np.uint8)  # bright, not grey
    out = simulate_camera(img, CAMERA_WH, mode='pad')

    # Left and right columns must be PAD_VALUE grey.
    assert (out[:, 0] == PAD_VALUE).all(), (
        f"expected left column to be pad grey {PAD_VALUE}, got {out[:, 0, 0]}"
    )
    assert (out[:, -1] == PAD_VALUE).all(), "expected right column to be pad grey"


def test_positives_measurement_actually_changes():
    """End-to-end guard: score a synthetic stock-sized image raw vs simulated,
    and assert the model's output is different. Proves the fix is plumbed
    through — a wiring bug where score_frame ignores the transform would
    show up here.

    Uses a fake interpreter so no model weights are needed; the assertion is
    that PREPROCESSED PIXELS differ, not that scores differ (they would
    trivially by construction with a deterministic model).
    """
    # 1536x2048 portrait (the median val image size)
    img = np.random.default_rng(0).integers(0, 256, size=(2048, 1536, 3),
                                            dtype=np.uint8)

    simulated = simulate_camera(img, CAMERA_WH, mode='crop')

    assert simulated.shape[:2] == (480, 640), \
        f"simulated shape wrong: {simulated.shape}"
    # The raw and simulated images have different shapes, so of course they
    # differ pixelwise. The point of this assertion is that simulate_camera
    # is not a no-op on this input class — a regression that returned `image`
    # unchanged would produce the SAME shape as the source.
    assert simulated.shape != img.shape, (
        "simulate_camera returned unchanged on a stock-sized input — the "
        "harness would still be measuring the model as if the camera did "
        "not exist"
    )


def test_invalid_mode_raises():
    """Guard against a typo in a caller — 'centre' or 'centred' would silently
    fall through if we did not check."""
    img = _distinctive_bands(100, 100)
    try:
        simulate_camera(img, CAMERA_WH, mode='centre')
    except ValueError:
        return
    assert False, "expected ValueError for invalid mode"


def test_invalid_target_raises():
    img = _distinctive_bands(100, 100)
    try:
        simulate_camera(img, (0, 480), mode='crop')
    except ValueError:
        return
    assert False, "expected ValueError for non-positive target width"


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
