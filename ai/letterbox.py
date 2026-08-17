# ============================================================
# FILE: ai/letterbox.py
#
# The preprocessing half of the model contract.
#
# The model was trained by Ultralytics at imgsz=640, which means every
# image it ever saw was LETTERBOXED: scaled by a single ratio so the
# aspect ratio is preserved, then padded to a square with grey 114.
# The camera delivers 640x480 (4:3). Resizing that straight to 640x640
# stretches the frame vertically by 1.33x and hands the model a shape it
# was never trained on — worst case for a tall thin bird.
#
# This module is the single definition of that convention for the app.
# It is a deliberate, exact transcription of Ultralytics' LetterBox
# (ultralytics/data/augment.py, v8.4.121, with auto=False, scale_fill=False,
# center=True, scaleup=True) — the same defaults its predictor uses for an
# exported fixed-input-size model such as our .tflite. Every arithmetic
# detail below matters, including the -0.1/+0.1 rounding trick that decides
# which edge gets the extra pixel when the padding is odd.
#
# tests/test_letterbox.py pins this against the real Ultralytics
# implementation. Do not "tidy" the arithmetic here without running it.
# ============================================================

from typing import NamedTuple, Tuple

import cv2
import numpy as np

# Ultralytics' LetterBox padding_value default. Not arbitrary: it is the grey
# the model saw in the padding of every training and validation image.
PAD_VALUE = 114


class LetterboxParams(NamedTuple):
    """Everything needed to map coordinates back to the original frame.

    ratio:  single scale factor applied to BOTH axes (that is the point of
            letterboxing — one ratio, no aspect distortion)
    pad_x:  pixels of padding added on the LEFT edge
    pad_y:  pixels of padding added on the TOP edge
    orig_w/orig_h: the original frame size, i.e. what undo() maps back to
    new_w/new_h:   the padded square handed to the model
    """
    ratio: float
    pad_x: int
    pad_y: int
    orig_w: int
    orig_h: int
    new_w: int
    new_h: int


def letterbox(image: np.ndarray, new_shape: Tuple[int, int] = (640, 640),
              pad_value: int = PAD_VALUE,
              scaleup: bool = True) -> Tuple[np.ndarray, LetterboxParams]:
    """Resize preserving aspect ratio and pad to `new_shape` with grey.

    Args:
        image: HxWx3 frame (BGR or RGB — this function does not care).
        new_shape: (height, width) of the model input.
        pad_value: grey level for the bars. Leave at 114 to match training.
        scaleup: allow enlarging a frame smaller than new_shape. True matches
            Ultralytics' predictor default.

    Returns:
        (padded image of exactly new_shape, params for undoing the transform)
    """
    h, w = image.shape[:2]
    new_h, new_w = int(new_shape[0]), int(new_shape[1])

    # One ratio for both axes: the smaller of the two possible scales, so the
    # scaled image fits inside the square on its long side.
    ratio = min(new_h / h, new_w / w)
    if not scaleup:
        ratio = min(ratio, 1.0)

    # round(), not int(): Ultralytics uses Python's round here, and int()
    # truncation would put us a pixel out on many frame sizes.
    new_unpad = (round(w * ratio), round(h * ratio))  # (width, height)

    dw = (new_w - new_unpad[0]) / 2  # padding split between left and right
    dh = (new_h - new_unpad[1]) / 2  # ...and between top and bottom

    # The -0.1/+0.1 is Ultralytics'. It exists so that an odd number of
    # padding pixels lands the extra one on the BOTTOM/RIGHT edge, rather than
    # being decided by Python's banker's rounding of exactly .5.
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)

    if (w, h) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    padded = cv2.copyMakeBorder(image, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(pad_value,) * 3)

    return padded, LetterboxParams(ratio=ratio, pad_x=left, pad_y=top,
                                   orig_w=w, orig_h=h, new_w=new_w, new_h=new_h)


def undo_xyxy(boxes: np.ndarray, params: LetterboxParams,
              clip: bool = True) -> np.ndarray:
    """Map xyxy boxes from padded-square PIXELS back to original-frame PIXELS.

    The inverse of letterbox(): subtract the padding, then divide by the single
    scale ratio. Equivalent to Ultralytics' ops.scale_boxes().

    Args:
        boxes: (N, 4) array of x1, y1, x2, y2 in padded-square pixel space.
        params: as returned by letterbox().
        clip: clamp to the frame. A detection may legitimately extend into the
            padding — that region does not exist in the original frame, so
            without clipping a box can carry negative or out-of-frame
            coordinates into anything that consumes it (annotation files
            especially).

    Returns:
        (N, 4) float array of x1, y1, x2, y2 in original-frame pixels.
    """
    out = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - params.pad_x) / params.ratio
    out[:, [1, 3]] = (out[:, [1, 3]] - params.pad_y) / params.ratio
    if clip:
        out[:, [0, 2]] = out[:, [0, 2]].clip(0, params.orig_w)
        out[:, [1, 3]] = out[:, [1, 3]].clip(0, params.orig_h)
    return out


def forward_xyxy(boxes: np.ndarray, params: LetterboxParams) -> np.ndarray:
    """Map xyxy boxes from original-frame PIXELS into padded-square PIXELS.

    Only the tests need this direction — it is how a known box is pushed
    through the transform so that undo_xyxy() can be checked for round-trip
    accuracy. Kept here, next to its inverse, so the two cannot drift apart.
    """
    out = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    out[:, [0, 2]] = out[:, [0, 2]] * params.ratio + params.pad_x
    out[:, [1, 3]] = out[:, [1, 3]] * params.ratio + params.pad_y
    return out
