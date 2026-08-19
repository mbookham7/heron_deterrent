# ============================================================
# FILE: tools/camera_sim.py
#
# The FIRST step of the input path — matched between measurement and
# deployment. `letterbox()` matches the last step; this matches what
# happens before the letterbox.
#
# ------------------------------------------------------------------
# WHY THIS EXISTS
# ------------------------------------------------------------------
# The eval harness used to load a 1536x2048 portrait stock photo, letterbox
# it straight to 640x640, and score it. Deployment does something quite
# different: the webcam delivers a 640x480 frame, which is then
# letterboxed. Two problems:
#
#   1. NOTHING IS CROPPED in the harness pipeline. A photo composed by a
#      human — subject usually centred, whole bird visible — always
#      survives letterboxing intact. Deployment gets whatever the camera's
#      4:3 sensor sees, which for a tall thin bird routinely clips head
#      or legs. Not simulating that biases positives optimistic.
#
#   2. DOWNSAMPLING FROM 1536x2048 to 640x640 produces a sharper 640x640
#      than a real 640x480 camera frame ever could. Fine detail survives.
#      Small distant birds in the stock set look larger and clearer than
#      their pond-camera equivalents will.
#
# This module is the single definition of "what the camera would have
# delivered." Two policies are supported for the aspect-ratio step, both
# necessary to report as sensitivity because they encode different
# tradeoffs:
#
#   * center_crop   — matches the camera physically (a sensor is a
#                     rectangle; anything outside the sensor is not seen).
#                     May clip a tall bird. This is CANONICAL because it
#                     matches deployment; it is also the more conservative
#                     claim about model quality.
#
#   * pad_to_aspect — pads the source to camera aspect with grey before
#                     scaling down. No real camera does this. Reported as
#                     an upper bound on recall — "what the model could do
#                     if the sensor were bigger."
#
# ------------------------------------------------------------------
# CANONICAL LOCATION & VENDORED COPY IN heron-detector
# ------------------------------------------------------------------
# THIS FILE is canonical. It lives at heron_deterrent/tools/camera_sim.py
# because camera geometry originates in the app (heron_deterrent owns the
# camera and config.yaml). It sits under tools/ rather than ai/ because
# the deployed device never simulates a camera — it HAS one. Nothing
# under tools/ ships to the Coral or the Pi.
#
# heron-detector vendors a byte-identical copy at src/camera_sim.py.
# heron-detector runs on EC2 where heron_deterrent is not checked out;
# vendoring is what lets the harness import without a sibling on disk.
# tests/test_camera_sim_parity.py (heron-detector) fails if the two copies
# drift. When editing this file: edit here, then run scripts/sync_from_
# heron_deterrent.sh in heron-detector, then run both repos' tests.
#
# The same sharing contract governs letterbox.py. See SHARING.md in both
# repos for the reasoning — do NOT invent a third mechanism.
#
# INTERPOLATION POLICY (load-bearing correctness detail)
# ------------------------------------------------------------------
# The resize step picks INTER_AREA when downsampling (cw >= target width)
# and INTER_LINEAR when upsampling. INTER_AREA on an upsample degenerates
# to nearest-neighbour behaviour, which introduces aliasing that can
# shift a per-frame score by ~0.003 near the confidence tail — enough to
# move one image from 0.898 to 0.900 on the val set. A prior implementation
# that used INTER_AREA unconditionally reported recall@0.90 as 19/44 on
# the deployed model; the correct number is 18/44.
# ============================================================

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import cv2
import numpy as np


# Grey used by cv2.copyMakeBorder in the pad_to_aspect method. Chosen to
# match letterbox()'s PAD_VALUE — if a padded region ends up inside the
# model's letterbox output, at least it's not a colour shift AND a value
# jump. Not load-bearing; the model was never trained on this grey either
# way.
PAD_GREY = 114


@dataclass(frozen=True)
class CameraGeometry:
    """What the camera would have delivered.

    width, height: the sensor's output resolution in pixels. 640x480 today;
        parametrised because the app may move to 1280x720 for
        region-of-interest cropping.
    jpeg_quality: 0 disables the JPEG round-trip. 1..100 encodes then
        decodes the frame at that quality after resize. This is a cheap
        proxy for the compression artefacts a real camera's H.264 mp4
        introduces between capture and inference; it is NOT bit-exact.
        Default 90 is close to Chromium's usual mid-quality setting and
        matches what most webcams do internally.
    """
    width: int = 640
    height: int = 480
    # 0 = OFF (default). Turning JPEG on flattens positive scores by
    # roughly 0.04 at the recall@0.90 tail, which matters more than the
    # nominal "closer to codec" benefit — the reference measurement this
    # module was calibrated against did not include it, and matching that
    # is the higher-value property. Left as an available knob because
    # once real footage exists to calibrate against, the codec-artefact
    # story may be worth revisiting.
    jpeg_quality: int = 0

    @property
    def aspect(self) -> float:
        return self.width / self.height


@dataclass(frozen=True)
class ResolvedGeometry:
    """A CameraGeometry plus where its width/height came from.

    Provenance is load-bearing: on EC2 the sibling repo may not be checked
    out, and a silent fallback to (640, 480) would mask a config typo or a
    future move to 1280x720. Callers should refuse to run when
    source == "fallback-default" unless the user explicitly opted in.
    """
    geometry: CameraGeometry
    source: str  # "config:/abs/path", "cli-override", "explicit-file:/abs/path", "fallback-default"


def _parse_resolution_yaml(config_path: Path) -> Optional[dict]:
    """Read `system.resolution` out of a YAML file by hand.

    Deliberately avoids PyYAML — this module has to run in the eval venv
    AND on the deployed device, and adding a dep for one two-key lookup
    is not worth it. If the YAML grows more complex, replace this with
    `yaml.safe_load`.
    """
    resolution: dict = {}
    in_resolution = False
    base_indent = 0
    for raw in config_path.read_text().splitlines():
        line = raw.rstrip()
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped.startswith("resolution:"):
            in_resolution = True
            base_indent = indent
            continue
        if in_resolution:
            if indent <= base_indent and stripped:
                break
            if ":" in stripped:
                k, v = stripped.split(":", 1)
                v = v.strip()
                try:
                    resolution[k.strip()] = int(v)
                except ValueError:
                    pass
    if "width" not in resolution or "height" not in resolution:
        return None
    return resolution


def load_geometry_from_deterrent_config(
    config_path: Optional[Path] = None,
) -> Optional[CameraGeometry]:
    """Read `system.resolution` out of heron_deterrent's config.yaml.

    Returns None if the file is not found. When config_path is None, we
    try two well-known locations so the same file works whether it is
    running as heron_deterrent/tools/camera_sim.py or as the vendored
    copy at heron-detector/src/camera_sim.py:

      * heron_deterrent/tools/camera_sim.py -> ../config.yaml
      * heron-detector/src/camera_sim.py    -> ../../heron_deterrent/config.yaml

    Caller decides whether None is a hard error. Historically this
    function paired with a silent 640x480 default in the harness; that
    fallback is now opt-in — see resolve_geometry() in this file.
    """
    if config_path is not None:
        if not config_path.exists():
            return None
        parsed = _parse_resolution_yaml(config_path)
        return None if parsed is None else CameraGeometry(width=parsed["width"], height=parsed["height"])

    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "config.yaml",                              # same-repo (heron_deterrent)
        here.parent.parent.parent / "heron_deterrent" / "config.yaml",   # sibling  (heron-detector)
    ]
    for candidate in candidates:
        if candidate.exists():
            parsed = _parse_resolution_yaml(candidate)
            if parsed is None:
                return None
            return CameraGeometry(width=parsed["width"], height=parsed["height"])
    return None


def load_geometry_with_source(
    config_path: Optional[Path] = None,
) -> Optional[Tuple[CameraGeometry, Path]]:
    """Same as load_geometry_from_deterrent_config but also returns the
    path we read from — for provenance in the fingerprint JSON.

    Returns None if no config was found (or the found one was
    malformed). Callers that must never silently default should refuse
    to run in that case unless the user opts in explicitly.
    """
    if config_path is not None:
        if not config_path.exists():
            return None
        parsed = _parse_resolution_yaml(config_path)
        if parsed is None:
            return None
        return CameraGeometry(width=parsed["width"], height=parsed["height"]), config_path.resolve()

    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "config.yaml",
        here.parent.parent.parent / "heron_deterrent" / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            parsed = _parse_resolution_yaml(candidate)
            if parsed is None:
                return None
            return CameraGeometry(width=parsed["width"], height=parsed["height"]), candidate.resolve()
    return None


class CameraSimTransform(NamedTuple):
    """Everything needed to map a bounding box from the SOURCE frame's pixel
    space into the SIMULATED frame's pixel space, and vice versa.

    method:      the aspect-ratio policy actually applied.
    src_hw:      (h, w) of the source image.
    dst_hw:      (h, w) of the simulated frame (== geometry.height/width).
    crop_off_xy: (x, y) in SOURCE pixels of the top-left corner of the
                 crop rectangle. Negative for pad_to_aspect (the "crop"
                 rectangle extends outside the source).
    scale:       one scale factor applied to both axes after crop/pad.
    """
    method: str
    src_hw: Tuple[int, int]
    dst_hw: Tuple[int, int]
    crop_off_xy: Tuple[int, int]
    scale: float


def simulate_camera_frame(
    image: np.ndarray,
    geometry: CameraGeometry,
    method: str = "center_crop",
) -> np.ndarray:
    """Produce what the camera would have delivered. See simulate_with_transform()
    for the version that also returns transform params for coordinate mapping."""
    frame, _ = simulate_with_transform(image, geometry, method)
    return frame


def simulate_with_transform(
    image: np.ndarray,
    geometry: CameraGeometry,
    method: str = "center_crop",
) -> Tuple[np.ndarray, CameraSimTransform]:
    """Simulate AND return the parameters needed to map source boxes into
    the simulated frame.

    Args:
        image: HxWx3 source frame, BGR uint8.
        geometry: target resolution + JPEG round-trip setting.
        method: "center_crop" (canonical, matches sensor physics) or
            "pad_to_aspect" (softer, over-estimates recall).

    Returns:
        (simulated frame, transform params).

    A source that already matches the target geometry passes through
    with only the JPEG round-trip applied (if jpeg_quality > 0), and
    the transform is the identity. Do NOT double-process camera-native
    frames.
    """
    if method not in ("center_crop", "pad_to_aspect"):
        raise ValueError(
            f"method must be 'center_crop' or 'pad_to_aspect', got {method!r}"
        )

    h, w = image.shape[:2]
    dst_hw = (geometry.height, geometry.width)

    # Camera-native passthrough. Identity transform.
    if (w, h) == (geometry.width, geometry.height):
        return (
            _maybe_jpeg_roundtrip(image, geometry.jpeg_quality),
            CameraSimTransform(method="passthrough", src_hw=(h, w),
                               dst_hw=dst_hw, crop_off_xy=(0, 0), scale=1.0),
        )

    if method == "center_crop":
        cropped, crop_off = _center_crop_to_aspect(image, geometry.aspect)
    else:
        cropped, crop_off = _pad_to_aspect(image, geometry.aspect)

    ch, cw = cropped.shape[:2]
    interp = cv2.INTER_AREA if cw >= geometry.width else cv2.INTER_LINEAR
    resized = cv2.resize(cropped, (geometry.width, geometry.height),
                         interpolation=interp)

    scale = geometry.width / cw   # cropped is aspect-corrected, both axes scale equally
    frame = _maybe_jpeg_roundtrip(resized, geometry.jpeg_quality)
    return frame, CameraSimTransform(method=method, src_hw=(h, w), dst_hw=dst_hw,
                                     crop_off_xy=crop_off, scale=scale)


def transform_xyxy(boxes_src: np.ndarray, transform: CameraSimTransform) -> np.ndarray:
    """Map xyxy pixel boxes from the SOURCE frame to the SIMULATED frame.

    A box that is entirely outside the simulated frame becomes a zero-
    area box — kept in the array (so total_gt is unchanged) but with
    coordinates that no detection can match. That is the load-bearing
    honesty: a bird cropped out of the sensor's view MUST still count
    against recall, because the deployed camera couldn't see it either.

    Args:
        boxes_src: (N, 4) xyxy in SOURCE pixel coordinates.
        transform: from simulate_with_transform().
    Returns:
        (N, 4) xyxy in SIMULATED-frame pixel coordinates.
    """
    if boxes_src.size == 0:
        return boxes_src.copy()
    ox, oy = transform.crop_off_xy
    out = boxes_src.astype(np.float64).copy()
    # Subtract crop offset, then scale.
    out[:, [0, 2]] = (out[:, [0, 2]] - ox) * transform.scale
    out[:, [1, 3]] = (out[:, [1, 3]] - oy) * transform.scale
    # Clip to the simulated frame. Boxes entirely outside end up as
    # zero-area rectangles (x1==x2 or y1==y2), which IoU handles as 0.
    dst_h, dst_w = transform.dst_hw
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, dst_w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, dst_h)
    # A box whose x2 < x1 after clip would be degenerate — force normal order.
    out[:, 2] = np.maximum(out[:, 2], out[:, 0])
    out[:, 3] = np.maximum(out[:, 3], out[:, 1])
    return out


def _center_crop_to_aspect(image: np.ndarray, target_aspect: float
                           ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Crop centred to match target_aspect. Returns (crop, (x_off, y_off))
    where (x_off, y_off) is the top-left corner of the crop rect in the
    SOURCE image's pixel coordinates. Never enlarges."""
    h, w = image.shape[:2]
    source_aspect = w / h
    if abs(source_aspect - target_aspect) < 1e-6:
        return image, (0, 0)
    if source_aspect > target_aspect:
        new_w = int(round(h * target_aspect))
        x0 = (w - new_w) // 2
        return image[:, x0:x0 + new_w], (x0, 0)
    new_h = int(round(w / target_aspect))
    y0 = (h - new_h) // 2
    return image[y0:y0 + new_h, :], (0, y0)


def _pad_to_aspect(image: np.ndarray, target_aspect: float
                   ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad centred with PAD_GREY to match target_aspect. Returns (padded,
    (x_off, y_off)) where the offsets are NEGATIVE and describe where the
    original SOURCE image sits inside the padded frame — equivalent to the
    top-left corner of the crop rectangle in source coords (which extends
    outside the source). Never crops."""
    h, w = image.shape[:2]
    source_aspect = w / h
    if abs(source_aspect - target_aspect) < 1e-6:
        return image, (0, 0)
    if source_aspect > target_aspect:
        new_h = int(round(w / target_aspect))
        extra = new_h - h
        top = extra // 2
        bottom = extra - top
        padded = cv2.copyMakeBorder(image, top, bottom, 0, 0,
                                    cv2.BORDER_CONSTANT, value=(PAD_GREY,) * 3)
        return padded, (0, -top)
    new_w = int(round(h * target_aspect))
    extra = new_w - w
    left = extra // 2
    right = extra - left
    padded = cv2.copyMakeBorder(image, 0, 0, left, right,
                                cv2.BORDER_CONSTANT, value=(PAD_GREY,) * 3)
    return padded, (-left, 0)


def _maybe_jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    """Encode then decode through JPEG at the given quality. quality=0 skips.

    Not a bit-exact simulation of any specific codec. It is a cheap
    approximation of the "the source went through lossy compression at
    some point" reality — the mp4 pipeline uses H.264 which is a
    different codec, but the perceptual character (8x8 block artefacts,
    softening of high-frequency detail) is the same.
    """
    if quality <= 0:
        return image
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# -----------------------------------------------------------------------------
# Per-source switch: is this frame already camera-native?
# -----------------------------------------------------------------------------


CAMERA_NATIVE_MARKER = ".camera_native"


def positives_dir_is_camera_native(pos_dir: Path) -> bool:
    """Return True if a positives directory is marked camera-native.

    A directory of frames that came directly from a camera (pond footage,
    once it arrives) should NOT be simulated — that would double-process
    them. Mark such a directory by dropping an empty file named
    `.camera_native` in its root. The name is a hidden dotfile so it
    doesn't clutter listings, and calling it out here (not deducing it
    from filenames or resolutions) means the user's intent is unambiguous.

    Auto-detecting camera-native by "all images are 640x480" was
    considered and rejected: the moment the app moves to 1280x720 and
    the marker file is what says "this dir was captured at the new
    resolution," heuristic detection would misfire.
    """
    return (pos_dir / CAMERA_NATIVE_MARKER).exists()
