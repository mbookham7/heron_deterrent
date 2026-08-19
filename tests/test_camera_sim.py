#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_camera_sim.py
#
# Pins tools/camera_sim.py — the canonical camera simulator, shared with
# heron-detector by vendoring (see SHARING.md).
#
# The simulator exists because scoring a 1536x2048 stock heron photo
# through a 640-input model does not match what the device sees — a
# 640x480 camera has already centre-cropped to 4:3 and downsampled by
# ~3x before letterboxing. The harness must match deployment or the
# recall figures overstate reality.
#
# Portrait/landscape/pass-through cases here, plus the transform_xyxy
# property that keeps IoU honest when a bird is clipped out of the crop.
# A single-axis test would not have caught the letterbox stretch bug
# earlier; the same reasoning applies to any transform with an
# aspect-ratio choice inside it.
# ============================================================

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))

from camera_sim import (  # noqa: E402
    PAD_GREY,
    CameraGeometry,
    CAMERA_NATIVE_MARKER,
    load_geometry_from_deterrent_config,
    positives_dir_is_camera_native,
    simulate_camera_frame,
    simulate_with_transform,
    transform_xyxy,
)

G = CameraGeometry(width=640, height=480, jpeg_quality=0)
G_JPEG = CameraGeometry(width=640, height=480, jpeg_quality=90)


def _rng(seed, h, w):
    r = np.random.default_rng(seed)
    return r.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _extreme_bands_vertical(h, w, strip=100):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:strip]       = (255,   0,   0)  # BLUE top
    img[strip:-strip] = (  0, 255,   0)  # GREEN middle
    img[-strip:]      = (  0,   0, 255)  # RED bottom
    return img


def _extreme_bands_horizontal(h, w, strip=100):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :strip]       = (255,   0,   0)
    img[:, strip:-strip] = (  0, 255,   0)
    img[:, -strip:]      = (  0,   0, 255)
    return img


def test_portrait_source_crops_top_and_bottom():
    img = _extreme_bands_vertical(2000, 1000, strip=100)
    out = simulate_camera_frame(img, G, method='center_crop')
    assert out.shape[:2] == (G.height, G.width)
    assert out[..., 0].max() < 30, "portrait crop left blue pixels — extreme top strip should be gone"
    assert out[..., 2].max() < 30, "portrait crop left red pixels — extreme bottom strip should be gone"
    assert out[..., 1].mean() > 200, "green (middle) should dominate the crop"


def test_landscape_source_crops_left_and_right():
    img = _extreme_bands_horizontal(1000, 3000, strip=100)
    out = simulate_camera_frame(img, G, method='center_crop')
    assert out.shape[:2] == (G.height, G.width)
    assert out[..., 0].max() < 30
    assert out[..., 2].max() < 30
    assert out[..., 1].mean() > 200


def test_camera_native_pass_through_is_byte_identical():
    """A 640x480 source is what the camera actually delivers. Piping it
    through simulate_camera_frame must not touch a byte when jpeg_quality
    is 0. Double-processing pond footage would degrade it and offset our
    numbers from deployment."""
    img = _rng(42, 480, 640)
    out = simulate_camera_frame(img, G, method='center_crop')
    assert np.array_equal(out, img), "camera-native frame was modified"


def test_output_matches_config_target_shape_4_3():
    img = _rng(0, 1500, 1000)
    for w, h in [(640, 480), (1280, 960), (320, 240)]:
        out = simulate_camera_frame(img, CameraGeometry(width=w, height=h), method='center_crop')
        assert out.shape[:2] == (h, w), f"target {(w, h)}: expected {(h, w)}, got {out.shape[:2]}"


def test_output_matches_config_target_shape_16_9():
    """The ROI experiment is likely to land on 1280x720 (16:9). The
    aspect maths must not silently assume 4:3."""
    img = _rng(0, 1500, 1000)
    out = simulate_camera_frame(img, CameraGeometry(width=1280, height=720), method='center_crop')
    assert out.shape[:2] == (720, 1280)


def test_pad_mode_preserves_all_source_features():
    """pad_to_aspect is the sensitivity mode — nothing is discarded, so
    top/bottom bands of a portrait source must all still be visible."""
    img = np.zeros((2000, 1000, 3), dtype=np.uint8)
    img[:667]        = (255,   0,   0)
    img[667:1333]    = (  0, 255,   0)
    img[1333:]       = (  0,   0, 255)
    out = simulate_camera_frame(img, G, method='pad_to_aspect')
    assert out.shape[:2] == (G.height, G.width)
    b_col_max = out[..., 0].max(axis=0)
    r_col_max = out[..., 2].max(axis=0)
    g_col_max = out[..., 1].max(axis=0)
    assert b_col_max.max() > 240
    assert r_col_max.max() > 240
    assert g_col_max.max() > 240


def test_pad_mode_uses_grey_padding():
    img = np.full((1000, 100, 3), 200, dtype=np.uint8)   # extreme portrait
    out = simulate_camera_frame(img, G, method='pad_to_aspect')
    assert (out[:, 0] == PAD_GREY).all()
    assert (out[:, -1] == PAD_GREY).all()


def test_positives_measurement_actually_changes():
    """A regression that returned `image` unchanged on stock-sized input
    would silently undo the whole camera-sim fix. Assert the shape
    changes on 1536x2048 portrait input."""
    img = _rng(0, 2048, 1536)
    simulated = simulate_camera_frame(img, G, method='center_crop')
    assert simulated.shape[:2] == (480, 640)
    assert simulated.shape != img.shape


def test_invalid_method_raises():
    try:
        simulate_camera_frame(_rng(0, 100, 100), G, method='centre')
    except ValueError:
        return
    assert False, "expected ValueError for invalid method"


# ---------------------------------------------------------------------
# transform_xyxy — the load-bearing "GT stays honest" property
# ---------------------------------------------------------------------


def test_centred_box_stays_centred_after_center_crop():
    img = _rng(0, 2048, 1536)
    _, tr = simulate_with_transform(img, G, method='center_crop')
    src = np.array([[668, 924, 868, 1124]], dtype=np.float64)
    dst = transform_xyxy(src, tr)
    cx_dst = (dst[0, 0] + dst[0, 2]) / 2
    cy_dst = (dst[0, 1] + dst[0, 3]) / 2
    assert abs(cx_dst - G.width / 2) < 2
    assert abs(cy_dst - G.height / 2) < 2


def test_top_bird_becomes_zero_area_after_center_crop():
    """The load-bearing honesty: a bird cropped out of the sensor's view
    MUST still count against recall."""
    img = _rng(0, 2048, 1536)
    _, tr = simulate_with_transform(img, G, method='center_crop')
    src = np.array([[600, 50, 800, 250]], dtype=np.float64)
    dst = transform_xyxy(src, tr)
    area = max(0.0, (dst[0, 2] - dst[0, 0])) * max(0.0, (dst[0, 3] - dst[0, 1]))
    assert area == 0.0


def test_transform_native_is_identity():
    img = _rng(0, 480, 640)
    _, tr = simulate_with_transform(img, G, method='center_crop')
    src = np.array([[100, 100, 400, 300]], dtype=np.float64)
    dst = transform_xyxy(src, tr)
    np.testing.assert_allclose(dst, src)


def test_pad_to_aspect_keeps_top_bird_visible():
    img = _rng(0, 2048, 1536)
    _, tr = simulate_with_transform(img, G, method='pad_to_aspect')
    src = np.array([[600, 50, 800, 250]], dtype=np.float64)
    dst = transform_xyxy(src, tr)
    area = (dst[0, 2] - dst[0, 0]) * (dst[0, 3] - dst[0, 1])
    assert area > 0.0


# ---------------------------------------------------------------------
# Interpolation policy — the regression that hid an 18/44-vs-19/44 gap
# ---------------------------------------------------------------------


def test_upsample_uses_bilinear_not_area():
    """A small portrait source (600x800) crops to 600x450 and must be
    UPSAMPLED to 640x480. INTER_AREA on an upsample degenerates to
    nearest-neighbour and aliases high-frequency content — that shifted
    image94.jpg's per-frame score above 0.90 in a prior implementation
    that used INTER_AREA unconditionally. Correct policy: INTER_LINEAR
    on upsample, INTER_AREA on downsample.

    Fingerprint of the right choice: on a source with a sharp step edge,
    bilinear produces intermediate values (edge softening) that nearest-
    neighbour never emits. If this test starts failing, someone changed
    the resize interpolation.
    """
    img = np.zeros((800, 600, 3), dtype=np.uint8)   # portrait, upsample path
    img[:, 300:] = 255                              # sharp vertical edge
    out = simulate_camera_frame(img, G, method='center_crop')
    mid_row = out[G.height // 2, :, 0]
    # Bilinear must produce at least one pixel strictly between 0 and 255
    # near the edge. Nearest-neighbour would never.
    intermediate = ((mid_row > 5) & (mid_row < 250)).any()
    assert intermediate, (
        "no intermediate greyscale value in the upsampled step edge — "
        "interpolation regressed to INTER_AREA on the upsample path."
    )


# ---------------------------------------------------------------------
# Marker file for camera-native directories
# ---------------------------------------------------------------------


def test_camera_native_marker_present(tmp_path):
    (tmp_path / CAMERA_NATIVE_MARKER).write_text('')
    assert positives_dir_is_camera_native(tmp_path) is True


def test_camera_native_marker_absent(tmp_path):
    assert positives_dir_is_camera_native(tmp_path) is False


# ---------------------------------------------------------------------
# Config loader — canonical file's same-repo path is authoritative here
# ---------------------------------------------------------------------


def test_config_loader_reads_this_repo_yaml():
    """When run from heron_deterrent, load_geometry_from_deterrent_config
    must find ../config.yaml. This is the load-bearing correctness check:
    if the app config moves to 1280x720 later, measure_thresholds.py
    tracks it only if this function keeps working."""
    g = load_geometry_from_deterrent_config()
    assert g is not None
    assert g.width == 640 and g.height == 480, \
        f"expected 640x480 in config.yaml, got {g.width}x{g.height}"


def test_config_loader_missing_returns_none(tmp_path):
    fake = tmp_path / 'nope.yaml'
    assert load_geometry_from_deterrent_config(fake) is None


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))
