#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_letterbox.py
#
# This file is the enforcement mechanism for the preprocessing half of
# the model contract. The class count is enforced at startup by
# ClassCountMismatch; the preprocessing convention was enforced by
# nothing at all, which is how the app came to stretch 640x480 frames
# into a 640x640 model input for months.
#
# Three things are pinned here:
#
#   1. The input tensor. Our letterbox() must produce byte-identical
#      output to Ultralytics' own LetterBox — checked directly against
#      the installed ultralytics package where one is available, and
#      against explicit hand-computed geometry everywhere else (the
#      board has no ultralytics and never will).
#
#   2. The bbox round trip. A known box must survive
#      original frame -> padded square -> back to original frame
#      to sub-pixel accuracy. This is the property that keeps the
#      dashboard's manual labelling from writing shifted annotation
#      files into the next training set.
#
#   3. The whole app path. A synthetic model output carrying a known box
#      is driven through ObjectDetector.infer() with a fake interpreter,
#      and the resulting Detection.bbox is checked against the box that
#      went in. Tests 1 and 2 can both pass while infer() forgets to call
#      rescale_to_frame(); this one cannot.
#
# Run with pytest, or directly (no pytest on the board):
#     python tests/test_letterbox.py
# ============================================================

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.detector import Detection, ObjectDetector          # noqa: E402
from ai.letterbox import (PAD_VALUE, forward_xyxy,          # noqa: E402
                          letterbox, undo_xyxy)

# The deployment case: 4:3 camera into a square model input.
CAMERA_SHAPE = (480, 640)   # (h, w) — system.resolution in config.yaml
MODEL_SHAPE = (640, 640)    # (h, w) — verified from both .tflite files

# (frame_h, frame_w) -> (ratio, pad_x, pad_y) into a 640x640 square.
# Hand-computed from the algorithm, NOT copied from our own output, so a change
# in behaviour shows up as a failure rather than being silently blessed.
#
#   ratio = min(640/h, 640/w);  unpad = (round(w*ratio), round(h*ratio))
#   pad_x = round((640-unpad_w)/2 - 0.1);  pad_y = round((640-unpad_h)/2 - 0.1)
EXPECTED_GEOMETRY = {
    (480, 640): (1.0, 0, 80),          # the camera: 160px of padding, 80 top and bottom
    (640, 480): (1.0, 80, 0),          # portrait: padding moves to left/right
    (640, 640): (1.0, 0, 0),           # already square: no padding at all
    (1080, 1920): (640 / 1920, 0, 140),  # 16:9 -> unpad h=360, 280px padding
    (481, 640): (1.0, 0, 79),          # ODD padding: 159px -> 79 top, 80 bottom
    (640, 481): (1.0, 79, 0),          # odd the other way: 79 left, 80 right
    (300, 400): (1.6, 0, 80),          # smaller than the input: scaled UP
}


def _synthetic_frame(h, w, seed=0):
    """A deterministic, high-frequency test image.

    Noise rather than a smooth gradient on purpose: it makes any difference in
    interpolation or in which edge gets padded visible immediately.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ------------------------------------------------------------------
# 1. The input tensor
# ------------------------------------------------------------------

def test_geometry_matches_expected():
    """Ratio and padding offsets, including which edge gets the odd pixel."""
    for (h, w), (exp_ratio, exp_pad_x, exp_pad_y) in EXPECTED_GEOMETRY.items():
        _, p = letterbox(_synthetic_frame(h, w), MODEL_SHAPE)
        assert abs(p.ratio - exp_ratio) < 1e-12, f"{h}x{w}: ratio {p.ratio} != {exp_ratio}"
        assert (p.pad_x, p.pad_y) == (exp_pad_x, exp_pad_y), \
            f"{h}x{w}: padding ({p.pad_x},{p.pad_y}) != ({exp_pad_x},{exp_pad_y})"
        assert (p.orig_w, p.orig_h) == (w, h)


def test_output_shape_and_pad_colour():
    """Exactly the model's input size, padded with Ultralytics' grey 114."""
    assert PAD_VALUE == 114, "114 is the grey the model was trained against"

    img, p = letterbox(_synthetic_frame(*CAMERA_SHAPE), MODEL_SHAPE)
    assert img.shape == (MODEL_SHAPE[0], MODEL_SHAPE[1], 3)

    # Top and bottom bars are pure 114; the middle band is image.
    assert (img[:p.pad_y] == PAD_VALUE).all(), "top bar is not grey 114"
    assert (img[p.pad_y + 480:] == PAD_VALUE).all(), "bottom bar is not grey 114"
    assert not (img[p.pad_y:p.pad_y + 480] == PAD_VALUE).all()


def test_aspect_ratio_is_preserved():
    """The point of the exercise: no anisotropic scaling.

    A rectangle drawn in the frame must come out with its width:height ratio
    intact. Under the old cv2.resize() straight to 640x640 it came out 1.33x
    too tall.
    """
    frame = np.zeros((*CAMERA_SHAPE, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)  # 100x100 square

    img, _ = letterbox(frame, MODEL_SHAPE)
    ys, xs = np.where(img[:, :, 0] == 255)
    height, width = ys.max() - ys.min(), xs.max() - xs.min()
    assert abs(height - width) <= 1, f"square came out {width}x{height} — aspect distorted"


def test_unpadded_region_is_a_plain_aspect_preserving_resize():
    """Inside the bars, the image is exactly cv2.resize at the single ratio."""
    frame = _synthetic_frame(*CAMERA_SHAPE, seed=7)
    img, p = letterbox(frame, MODEL_SHAPE)

    unpad_w, unpad_h = round(p.orig_w * p.ratio), round(p.orig_h * p.ratio)
    expected = cv2.resize(frame, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    actual = img[p.pad_y:p.pad_y + unpad_h, p.pad_x:p.pad_x + unpad_w]
    assert np.array_equal(actual, expected)


def test_matches_ultralytics_exactly():
    """Byte-for-byte equality with Ultralytics' own LetterBox.

    Skipped where ultralytics is not installed (the board, and any bare venv).
    This is the assertion that makes the rest of the file more than an
    elaborate description of our own behaviour, so run it on a dev machine:
        pip install ultralytics
    """
    try:
        from ultralytics.data.augment import LetterBox
    except ImportError:
        print("SKIP test_matches_ultralytics_exactly (ultralytics not installed)")
        return

    # auto=False and scale_fill=False are what Ultralytics' predictor uses for a
    # model with a fixed input size, which every exported .tflite has.
    reference = LetterBox(new_shape=MODEL_SHAPE, auto=False, scale_fill=False,
                          scaleup=True, center=True)

    for i, (h, w) in enumerate(EXPECTED_GEOMETRY):
        frame = _synthetic_frame(h, w, seed=i)
        ours, _ = letterbox(frame, MODEL_SHAPE)
        theirs = reference(image=frame.copy())
        assert np.array_equal(ours, theirs), \
            f"{h}x{w}: our letterbox differs from Ultralytics LetterBox"


# ------------------------------------------------------------------
# 2. The bbox round trip
# ------------------------------------------------------------------

# Boxes in original-frame pixels (x1, y1, x2, y2) on a 640x480 frame.
ROUND_TRIP_BOXES = [
    (100.0, 50.0, 200.0, 300.0),    # tall and thin, like a heron
    (0.0, 0.0, 640.0, 480.0),       # the whole frame, touching every edge
    (10.0, 10.0, 20.0, 20.0),       # tiny, in the corner
    (319.0, 239.0, 321.0, 241.0),   # sub-pixel-ish, dead centre
    (500.0, 400.0, 639.0, 479.0),   # bottom right
]


def test_bbox_round_trip_is_lossless():
    """frame -> padded square -> frame must return the original coordinates."""
    for shape in [(480, 640), (640, 480), (481, 640), (1080, 1920), (300, 400)]:
        h, w = shape
        _, p = letterbox(_synthetic_frame(h, w), MODEL_SHAPE)
        boxes = np.array([(x1 * w / 640, y1 * h / 480, x2 * w / 640, y2 * h / 480)
                          for x1, y1, x2, y2 in ROUND_TRIP_BOXES])

        back = undo_xyxy(forward_xyxy(boxes, p), p)
        assert np.allclose(back, boxes, atol=1e-6), \
            f"{h}x{w}: round trip drifted by up to {np.abs(back - boxes).max():.4f}px"


def test_undo_matches_ultralytics_scale_boxes():
    """undo_xyxy() must agree with Ultralytics' ops.scale_boxes().

    scale_boxes() is the function Ultralytics itself uses to bring predictions
    out of letterboxed space, so this checks the undo direction against the
    reference rather than only against our own forward transform (which would
    pass even if both directions were wrong in the same way).

    Skipped where ultralytics is not installed.
    """
    try:
        from ultralytics.utils.ops import scale_boxes
    except ImportError:
        print("SKIP test_undo_matches_ultralytics_scale_boxes (ultralytics not installed)")
        return

    rng = np.random.default_rng(42)
    for h, w in EXPECTED_GEOMETRY:
        _, p = letterbox(_synthetic_frame(h, w), MODEL_SHAPE)
        x1 = rng.uniform(-20, 640, 40)
        y1 = rng.uniform(-20, 640, 40)
        boxes = np.stack([x1, y1, x1 + rng.uniform(1, 200, 40),
                          y1 + rng.uniform(1, 200, 40)], axis=1)

        ours = undo_xyxy(boxes, p)
        theirs = scale_boxes(MODEL_SHAPE, boxes.copy(), (h, w))
        assert np.allclose(ours, theirs, atol=1e-9), \
            f"{h}x{w}: undo differs from scale_boxes by up to {np.abs(ours - theirs).max()}"


def test_undo_actually_moves_the_box():
    """Guard against a no-op 'undo'.

    If someone reverts rescale_to_frame() to a pass-through, the round-trip test
    above still passes (identity composed with identity). This asserts the two
    coordinate systems really are different for the camera's frame size: a box
    at the top of the frame sits 80px lower in the padded square.
    """
    _, p = letterbox(_synthetic_frame(*CAMERA_SHAPE), MODEL_SHAPE)
    assert p.pad_y == 80
    box = np.array([[100.0, 0.0, 200.0, 100.0]])
    assert forward_xyxy(box, p)[0][1] == 80.0, "forward transform did not apply padding"


def test_yolo_annotation_would_be_correct():
    """The dataset-poisoning case, stated in the units that actually get written.

    media_store.save_labeled_image() converts a box to YOLO normalized centre
    form. Feed it a box that has been through the full letterbox round trip and
    check the numbers a training file would contain.
    """
    h, w = CAMERA_SHAPE
    _, p = letterbox(_synthetic_frame(h, w), MODEL_SHAPE)

    # A heron standing in the upper half of the frame — deliberately off-centre.
    # A centred box is the one case where the mistake is invisible in the y
    # centre, because centre padding maps the frame's centre to the square's.
    truth_xyxy = np.array([[240.0, 48.0, 400.0, 192.0]])
    recovered = undo_xyxy(forward_xyxy(truth_xyxy, p), p)[0]

    def yolo(x1, y1, x2, y2, height, width):
        return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height,
                (x2 - x1) / width, (y2 - y1) / height)

    expected = yolo(*truth_xyxy[0], h, w)
    actual = yolo(*recovered, h, w)
    assert np.allclose(actual, expected, atol=1e-6), \
        f"YOLO annotation would be written as {actual}, not {expected}"

    # And the size of the error had the padding never been undone: this is the
    # annotation that would have gone into the next training set.
    sq = forward_xyxy(truth_xyxy, p)[0]
    naive = yolo(*sq, p.new_h, p.new_w)
    assert abs(naive[1] - expected[1]) > 0.05, \
        f"expected a materially wrong y centre, got {naive[1]} vs {expected[1]}"
    assert abs(naive[3] - expected[3]) > 0.05, \
        f"expected a materially wrong height, got {naive[3]} vs {expected[3]}"


# ------------------------------------------------------------------
# 3. The whole app path, through ObjectDetector.infer()
# ------------------------------------------------------------------

class _FakeInterpreter:
    """Emits one anchor holding a known box, in the model's own output format.

    Output layout matches the real models: [1, 5, 8400] float32, boxes as
    normalized centre-form xywh against the padded square.
    """

    def __init__(self, box_xywh_norm, score, num_anchors=8400):
        self.box, self.score, self.num_anchors = box_xywh_norm, score, num_anchors
        self._input = None

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [{'index': 0, 'shape': np.array([1, 640, 640, 3]),
                 'dtype': np.float32, 'quantization': (0.0, 0)}]

    def get_output_details(self):
        return [{'index': 1, 'shape': np.array([1, 5, self.num_anchors]),
                 'dtype': np.float32, 'quantization': (0.0, 0)}]

    def set_tensor(self, index, data):
        self._input = data

    def invoke(self):
        pass

    def get_tensor(self, index):
        out = np.zeros((1, 5, self.num_anchors), dtype=np.float32)
        cx, cy, bw, bh = self.box
        out[0, 0, 0], out[0, 1, 0], out[0, 2, 0], out[0, 3, 0] = cx, cy, bw, bh
        out[0, 4, 0] = self.score
        return out


class _FakeLoader:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.model_path = Path('fake_model.tflite')

    def get_interpreter(self):
        return self.interpreter

    def get_input_details(self):
        return self.interpreter.get_input_details()

    def get_output_details(self):
        return self.interpreter.get_output_details()


def test_infer_returns_frame_relative_bbox():
    """End to end: a box the model reports must come back in frame coordinates.

    The model 'sees' a heron filling the middle of the padded square. On a
    640x480 frame, 80px of that square's top and bottom are grey padding, so the
    same box is proportionally TALLER relative to the frame than to the square.
    """
    h, w = CAMERA_SHAPE
    frame = _synthetic_frame(h, w)

    # A box in original-frame pixels, pushed forward into square-normalized
    # centre form — exactly what a correctly-behaving model would emit.
    truth_xyxy = np.array([[160.0, 96.0, 480.0, 384.0]])
    _, p = letterbox(frame, MODEL_SHAPE)
    sq = forward_xyxy(truth_xyxy, p)[0]
    box_norm = (((sq[0] + sq[2]) / 2) / p.new_w, ((sq[1] + sq[3]) / 2) / p.new_h,
                (sq[2] - sq[0]) / p.new_w, (sq[3] - sq[1]) / p.new_h)

    detector = ObjectDetector(_FakeLoader(_FakeInterpreter(box_norm, 0.9)),
                             confidence_threshold=0.5, labels={0: 'heron'})
    det = detector.infer(frame)

    assert det.label == 'heron'
    assert abs(det.confidence - 0.9) < 1e-5

    x, y, bw, bh = det.bbox
    expected = (truth_xyxy[0][0] / w, truth_xyxy[0][1] / h,
                (truth_xyxy[0][2] - truth_xyxy[0][0]) / w,
                (truth_xyxy[0][3] - truth_xyxy[0][1]) / h)
    assert np.allclose((x, y, bw, bh), expected, atol=1e-4), \
        f"bbox {(x, y, bw, bh)} != frame-relative {expected}"


def test_infer_bbox_is_not_left_in_square_space():
    """The specific regression: bbox still normalized against the padded square.

    Under the old code the height of this box would come back as 0.45 (its share
    of the 640px square) instead of 0.6 (its share of the 480px frame). A 33%
    vertical error, invisible unless you check the number.
    """
    h, w = CAMERA_SHAPE
    frame = _synthetic_frame(h, w)
    truth_xyxy = np.array([[160.0, 96.0, 480.0, 384.0]])
    _, p = letterbox(frame, MODEL_SHAPE)
    sq = forward_xyxy(truth_xyxy, p)[0]
    box_norm = (((sq[0] + sq[2]) / 2) / p.new_w, ((sq[1] + sq[3]) / 2) / p.new_h,
                (sq[2] - sq[0]) / p.new_w, (sq[3] - sq[1]) / p.new_h)

    detector = ObjectDetector(_FakeLoader(_FakeInterpreter(box_norm, 0.9)),
                             confidence_threshold=0.5, labels={0: 'heron'})
    det = detector.infer(frame)

    square_relative_h = box_norm[3]
    assert abs(det.bbox[3] - square_relative_h) > 0.1, \
        "bbox height still measured against the padded square"
    assert abs(det.bbox[3] - 0.6) < 1e-3


def test_infer_clips_detections_that_extend_into_padding():
    """A box overlapping the grey bars must be clipped, not left negative.

    Negative coordinates in an annotation file are silently accepted by some
    tooling and mangled by the rest.
    """
    frame = _synthetic_frame(*CAMERA_SHAPE)
    # Centred at the very top of the square, i.e. entirely inside the top bar
    # and spilling above it.
    detector = ObjectDetector(_FakeLoader(_FakeInterpreter((0.5, 0.02, 0.2, 0.2), 0.9)),
                              confidence_threshold=0.5, labels={0: 'heron'})
    det = detector.infer(frame)
    x, y, bw, bh = det.bbox
    assert x >= 0 and y >= 0, f"bbox has negative coordinates: {det.bbox}"
    assert x + bw <= 1.0 + 1e-9 and y + bh <= 1.0 + 1e-9, f"bbox exceeds frame: {det.bbox}"


def test_preprocess_frame_returns_params():
    """The signature change itself: a caller cannot get a tensor without the
    parameters needed to undo the transform."""
    frame = _synthetic_frame(*CAMERA_SHAPE)
    detector = ObjectDetector(_FakeLoader(_FakeInterpreter((0.5, 0.5, 0.1, 0.1), 0.9)),
                              confidence_threshold=0.5, labels={0: 'heron'})
    tensor, params = detector.preprocess_frame(frame)
    assert tensor.shape == (1, 640, 640, 3)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0
    assert params.pad_y == 80 and params.pad_x == 0


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
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
