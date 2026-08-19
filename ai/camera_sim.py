# ============================================================
# FILE: ai/camera_sim.py
#
# Simulate the camera pipeline, so measurement matches deployment.
#
# The model was trained on Ultralytics-preprocessed images at 640x640. The
# preprocessing has TWO stages:
#
#   1. camera:    sensor -> 640x480 frame (crop and downsample by lens+ISP)
#   2. letterbox: 640x480 -> 640x640 (grey padding to a square) — see letterbox.py
#
# letterbox.py is the second stage. This module is the first. Together they
# reproduce, on a stock image, the pixels the model would have seen from the
# camera on that same scene.
#
# WHY THIS EXISTS
#
# The validation set (heron-data/heron_project/dataset/val/) is 44 stock photos
# — median 1536x2048, 32 of them portrait, none anywhere near 640x480. If the
# harness letterboxes those straight to the model input, two things happen and
# both flatter the model:
#
#   * nothing is cropped, so a portrait heron with head and legs at the top
#     and bottom of the frame stays entirely visible — a 4:3 camera would
#     have clipped both
#   * the source has ~10x more pixels than 640x480, and downsampling to 640x640
#     produces a sharper output than the camera can, so texture the camera
#     would blur is presented intact
#
# An earlier measurement that did simulate the camera reported positives
# median 0.887 and recall at deter=0.90 of 0.42. Without simulation the
# harness reports 0.902 and 0.55. Recall is overstated by roughly a third.
# Recall is the number that says whether the device actually catches herons.
#
# SIMULATION MODES
#
# crop (canonical): centre-crop the source to the camera's aspect ratio, then
#     downsample to the camera's resolution. Matches what the camera actually
#     does — the lens sees a wider scene than the sensor keeps, and the parts
#     outside 4:3 do not exist in the resulting frame. This is the conservative
#     claim: a portrait heron LOSES features to the crop.
# pad (sensitivity):  pad the source to the target aspect ratio with grey 114,
#     then downsample. Gentler on portraits (nothing is discarded) but no
#     camera does this — it exists only so the sensitivity of downstream
#     metrics to the camera assumption is visible instead of buried.
#
# When new pond footage arrives, those frames are already 640x480: pass them
# through unchanged (simulate_camera skipped at the caller). Simulating a
# camera on a frame that already came from one would double-process it.
#
# NOT INCLUDED: JPEG round-trip. Val images are already JPEG-decoded off disk,
# so encoding+decoding again would stack a second round of compression
# artefacts on top of whatever the source photo already carries. A single
# camera capture is one JPEG encode; matching that would need control over
# the source's original encoding, which we do not have. Leaving JPEG out
# matches the earlier 0.887/0.42 reference measurement — if it ever needs
# adding, do it in one place here, not sprinkled through callers.
#
# PARITY CONTRACT: this file will be copied verbatim to
# heron-detector/src/camera_sim.py when that repo's harness is retooled. The
# two must stay pixel-identical, enforced by tests/test_camera_sim_parity.py
# on the heron-detector side (same pattern as letterbox.py). Do not "tidy"
# the arithmetic here without editing BOTH copies and running the parity test.
# ============================================================

from typing import Tuple

import cv2
import numpy as np

# Match letterbox.py's grey. Only used by pad mode, and only so the sensitivity
# report's second column shares a padding colour with the letterbox step below
# it — makes the two padded regions visually distinguishable from real content.
PAD_VALUE = 114


def simulate_camera(image: np.ndarray, target_wh: Tuple[int, int],
                    mode: str = 'crop') -> np.ndarray:
    """Return what the camera would have delivered for this scene.

    Args:
        image: HxWx3 BGR (or RGB — this function does not care). Any size.
        target_wh: (width, height) of the camera's output frame, from
            config.yaml -> system.resolution. NEVER hardcoded: a future move
            to 1280x720 for region-of-interest cropping must propagate here.
        mode: 'crop' (canonical, matches a real 4:3 camera) or 'pad'
            (sensitivity — for the report only, no camera does this).

    Returns:
        HxWx3 image of exactly target_wh, same dtype as input.

    A source already at target_wh is returned untouched — callers can safely
    pipe camera-native positives through in 'crop' mode too, but the intended
    idiom is to skip this function entirely for camera-source positives so the
    intent stays explicit at the call site.
    """
    if mode not in ('crop', 'pad'):
        raise ValueError(f"mode must be 'crop' or 'pad', got {mode!r}")

    src_h, src_w = image.shape[:2]
    tgt_w, tgt_h = int(target_wh[0]), int(target_wh[1])
    if tgt_w <= 0 or tgt_h <= 0:
        raise ValueError(f"target_wh must be positive, got {target_wh}")

    # A source that already matches the target is returned as-is. Two reasons:
    # avoids a pointless resize round-trip, and avoids introducing any
    # interpolation artefacts on frames that came from the camera we are
    # simulating.
    if (src_w, src_h) == (tgt_w, tgt_h):
        return image

    tgt_aspect = tgt_w / tgt_h
    src_aspect = src_w / src_h

    if mode == 'crop':
        # Centre-crop to the target aspect. If the source is TALLER than the
        # target (portrait, or wider aspect the other way), take the widest
        # slice we can and clip the top/bottom. If it is WIDER, clip the sides.
        if src_aspect > tgt_aspect:
            # source too wide: clip sides
            new_w = int(round(src_h * tgt_aspect))
            new_h = src_h
            x0 = (src_w - new_w) // 2
            y0 = 0
        else:
            # source too tall: clip top/bottom — this is the case for portrait
            # heron photos, where head or legs may be lost
            new_w = src_w
            new_h = int(round(src_w / tgt_aspect))
            x0 = 0
            y0 = (src_h - new_h) // 2
        cropped = image[y0:y0 + new_h, x0:x0 + new_w]
        # INTER_AREA is the right pick for downsampling — averages source
        # pixels into each target pixel, matching what a sensor's own averaging
        # does. INTER_LINEAR (letterbox's choice) is for the near-1x resize
        # letterboxing does; it aliases when downsampling by 3x.
        return cv2.resize(cropped, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)

    # mode == 'pad': fit-and-pad. Scale so the whole source fits inside the
    # target box, then pad the remaining edges with PAD_VALUE grey.
    scale = min(tgt_w / src_w, tgt_h / src_h)
    scaled_w = int(round(src_w * scale))
    scaled_h = int(round(src_h * scale))
    scaled = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    pad_x = tgt_w - scaled_w
    pad_y = tgt_h - scaled_h
    top = pad_y // 2
    bottom = pad_y - top
    left = pad_x // 2
    right = pad_x - left
    return cv2.copyMakeBorder(scaled, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
