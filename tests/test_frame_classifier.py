#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_frame_classifier.py
#
# Guards the day/IR/dark frame classifier and its hysteresis.
#
# The failure modes that matter:
#   - an IR frame read as daylight, so the daylight model runs on it;
#   - a grey overcast DAY frame read as IR, so a whole day runs blind;
#   - a dark colour frame (camera with no IR illuminator) read as day;
#   - the dawn/dusk flicker flapping the mode back and forth.
#
# Frames are synthetic but shaped like the real thing: the IR frame goes
# through a JPEG round-trip the way a USB camera's MJPEG stream does, and the
# dark frame carries the independent per-channel noise of a high-gain sensor.
#
# Run with pytest, or directly:
#     python tests/test_frame_classifier.py
# ============================================================

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture.frame_classifier import (  # noqa: E402
    DARK, DAY, ESTABLISHING, IR, SWITCHING,
    FrameTypeConfig, FrameTypeTracker, classify_frame, measure,
)

H, W = 480, 640
CFG = FrameTypeConfig()
RNG = np.random.default_rng(0)


def _colour_frame():
    """Pond-edge daylight: green foliage band, blue-grey water, textured."""
    f = np.zeros((H, W, 3), np.float32)
    f[: H // 2] = (60, 140, 70)       # BGR foliage
    f[H // 2:] = (130, 110, 90)       # BGR water
    f += RNG.normal(0, 12, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def _overcast_frame():
    """Grey, low-saturation daylight — the case most likely to be mistaken
    for IR. Channels differ by a few levels, not by zero."""
    base = RNG.normal(140, 25, (H, W, 1))
    tint = np.array([-3.0, 1.0, 4.0])  # faint warm cast
    f = base + tint + RNG.normal(0, 2.5, (H, W, 3))
    return np.clip(f, 0, 255).astype(np.uint8)


def _ir_frame():
    """True monochrome: one luminance plane copied to all three channels,
    then MJPEG-encoded and decoded as the camera path would."""
    lum = np.clip(RNG.normal(100, 40, (H, W)), 0, 255).astype(np.uint8)
    lum = cv2.GaussianBlur(lum, (5, 5), 0)
    bgr = cv2.cvtColor(lum, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _dark_colour_frame():
    """Night from a colour camera with no IR: dim, and noisy per channel."""
    f = RNG.normal(20, 6, (H, W, 3))
    return np.clip(f, 0, 255).astype(np.uint8)


def _ambiguous_frame():
    """Mid-cut frame: the camera's chroma is fading but not gone."""
    f = _ir_frame().astype(np.float32)
    f[..., 2] += 3.0   # a sliver of red: chroma ~2.0, inside the dead band
    return np.clip(f, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Single-frame classification
# --------------------------------------------------------------------------

def test_colour_frame_is_day():
    assert classify_frame(_colour_frame(), CFG) == DAY


def test_monochrome_frame_is_ir():
    chroma, _ = measure(_ir_frame())
    assert chroma < CFG.mono_max_chroma, f"JPEG round-trip left chroma {chroma:.2f}"
    assert classify_frame(_ir_frame(), CFG) == IR


def test_single_channel_frame_is_ir():
    """Some drivers hand back a 2-D greyscale array in IR mode."""
    lum = np.full((H, W), 90, np.uint8)
    assert classify_frame(lum, CFG) == IR


def test_dark_colour_frame_is_dark():
    assert classify_frame(_dark_colour_frame(), CFG) == DARK


def test_black_frame_is_dark_not_ir():
    """R=G=B=0 has zero chroma, but that is absence of light, not IR."""
    assert classify_frame(np.zeros((H, W, 3), np.uint8), CFG) == DARK


def test_overcast_day_is_not_ir():
    """The dangerous misclassification: a grey day read as IR would run the
    whole day motion-only, with no model."""
    chroma, luma = measure(_overcast_frame())
    assert chroma > CFG.ambiguous_chroma, f"overcast chroma {chroma:.2f} too low for the test"
    assert classify_frame(_overcast_frame(), CFG) == DAY


def test_mid_transition_frame_is_ambiguous():
    assert classify_frame(_ambiguous_frame(), CFG) is None


# --------------------------------------------------------------------------
# Hysteresis
# --------------------------------------------------------------------------

def _feed(tracker, frames):
    return [tracker.update(f) for f in frames]


def test_startup_is_establishing_until_n_frames_agree():
    t = FrameTypeTracker(CFG)
    n = CFG.switch_after_frames
    out = _feed(t, [_colour_frame() for _ in range(n - 1)])
    assert all(s.state == ESTABLISHING for s, _ in out)
    assert not any(sw for _, sw in out)

    snap, switched = t.update(_colour_frame())
    assert switched and snap.state == DAY and t.mode == DAY


def test_sustained_change_switches_after_exactly_n_frames():
    t = FrameTypeTracker(CFG)
    n = CFG.switch_after_frames
    _feed(t, [_colour_frame() for _ in range(n)])
    assert t.mode == DAY

    out = _feed(t, [_ir_frame() for _ in range(n)])
    states = [s.state for s, _ in out]
    assert states[:-1] == [SWITCHING] * (n - 1), states
    assert states[-1] == IR
    assert [sw for _, sw in out] == [False] * (n - 1) + [True]


def test_dusk_flicker_does_not_flap():
    """The camera cutting in and out of IR at dusk, with transition frames
    in between. N-1 IR frames at a time is never enough."""
    t = FrameTypeTracker(CFG)
    n = CFG.switch_after_frames
    _feed(t, [_colour_frame() for _ in range(n)])

    flicker = []
    for burst in range(20):
        flicker += [_ir_frame() for _ in range(n - 1)]
        flicker += [_ambiguous_frame()] if burst % 2 else [_colour_frame()]
    out = _feed(t, flicker)

    assert t.mode == DAY
    assert not any(sw for _, sw in out), "mode switched during a flicker"


def test_ambiguous_frames_do_not_count_toward_a_switch():
    t = FrameTypeTracker(CFG)
    n = CFG.switch_after_frames
    _feed(t, [_colour_frame() for _ in range(n)])
    # Interleave so that IR frames total far more than n, but never n in a row.
    seq = []
    for _ in range(n * 3):
        seq += [_ir_frame(), _ir_frame(), _ambiguous_frame()]
    _feed(t, seq)
    assert t.mode == DAY


def test_switching_state_clears_when_frames_return_to_the_mode():
    t = FrameTypeTracker(CFG)
    _feed(t, [_colour_frame() for _ in range(CFG.switch_after_frames)])
    snap, _ = t.update(_ir_frame())
    assert snap.state == SWITCHING and snap.candidate == IR
    snap, _ = t.update(_colour_frame())
    assert snap.state == DAY and snap.candidate is None


def test_snapshot_records_raw_type_for_unsettled_frames():
    """Clips recorded while unsettled store what the classifier believed,
    including 'ambiguous' — never forced into day/ir/dark."""
    t = FrameTypeTracker(CFG)
    snap, _ = t.update(_ambiguous_frame())
    md = snap.as_metadata()
    assert md['frame_state'] == ESTABLISHING
    assert md['raw_frame_type'] == 'ambiguous'
    assert md['settled_mode'] is None


def test_config_rejects_unknown_and_incoherent_keys():
    for bad in ({'mono_max_chroma': 3.0, 'ambiguous_chroma': 2.0},
                {'switch_after_frames': 0},
                {'typo_key': 1}):
        try:
            FrameTypeConfig.from_dict(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_decoded_clip_frames_are_not_trusted_for_classification():
    """Why labelling uses the capture-time record and never re-classifies.

    Writing a pure-grey IR frame into a video and decoding it back gives it a
    uniform colour cast from the encoder's colour conversion — enough to read
    as daylight. If this test ever starts failing because the decoded frame
    DOES read as IR, the codec changed; the capture-time record is still the
    right source, so do not "simplify" labelling back to re-classifying.
    """
    import tempfile
    from capture.frame_classifier import clip_frame_types

    mono = np.full((H, W, 3), 100, np.uint8)
    mono[100:200, 100:300] = 180
    assert clip_frame_types([mono], CFG) == [IR]

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/clip.mp4"
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (W, H))
        if not writer.isOpened():
            import pytest
            pytest.skip("no mp4v writer on this platform")
        for _ in range(5):
            writer.write(mono)
        writer.release()
        cap = cv2.VideoCapture(path)
        cap.read()
        ok, decoded = cap.read()
        cap.release()
    assert ok
    assert classify_frame(decoded, CFG) != IR, (
        "decoded IR frame now classifies as IR — see docstring before acting on this"
    )


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
