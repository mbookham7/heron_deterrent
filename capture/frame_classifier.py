# ============================================================
# FILE: capture/frame_classifier.py
#
# Decides what KIND of image the camera is producing: daylight
# colour, infrared monochrome, or dark colour. The camera decides
# when to cut to IR; this module reads that decision off the
# pixels. Time of day is deliberately not an input — a clock
# drifts against sunrise with the seasons, the frame does not.
#
# Two layers:
#   classify_frame()   — one frame, no memory. Returns day/ir/dark,
#                        or None when the frame sits in the dead band.
#   FrameTypeTracker   — hysteresis over a stream of frames, so the
#                        dawn/dusk flicker between IR and colour does
#                        not flap the system's mode.
# ============================================================

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DAY = 'day'
IR = 'ir'
DARK = 'dark'
# The only values a settled mode can take, and the only names a labelled
# training directory may have.
FRAME_TYPES = (DAY, IR, DARK)

# Unsettled states. A frame captured in one of these is recorded, but never
# classified and never routed into training data automatically.
ESTABLISHING = 'establishing'   # startup: no mode agreed yet
SWITCHING = 'switching'         # a mode is set, but frames disagree with it
UNSETTLED = (ESTABLISHING, SWITCHING)

# Name used in metadata for a frame inside the dead band.
AMBIGUOUS = 'ambiguous'


@dataclass(frozen=True)
class FrameTypeConfig:
    """Thresholds, all on the 0-255 pixel scale.

    chroma is the mean absolute difference between colour channels. A camera
    that has cut to IR emits R=G=B (it sends YUV with neutral chroma), so its
    chroma is ~0 — usually under 1 after decoding. A colour scene, even a grey
    overcast one, has some colour somewhere and sits well above that.
    """
    mono_max_chroma: float = 1.5     # below this (and not black) = IR
    ambiguous_chroma: float = 2.5    # between the two = dead band, counts for nothing
    dark_max_luma: float = 35.0      # colour frame dimmer than this = dark
    black_max_luma: float = 8.0      # nothing visible; colour cannot be judged
    switch_after_frames: int = 10    # consecutive agreeing frames to change mode
    sample_step: int = 8             # every Nth pixel on each axis

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> 'FrameTypeConfig':
        d = d or {}
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"frame_type: unknown key(s) {sorted(unknown)}; "
                             f"valid keys are {sorted(known)}")
        cfg = cls(**d)
        if not (0 <= cfg.mono_max_chroma <= cfg.ambiguous_chroma):
            raise ValueError("frame_type: need 0 <= mono_max_chroma <= ambiguous_chroma")
        if not (0 <= cfg.black_max_luma <= cfg.dark_max_luma):
            raise ValueError("frame_type: need 0 <= black_max_luma <= dark_max_luma")
        if cfg.switch_after_frames < 1 or cfg.sample_step < 1:
            raise ValueError("frame_type: switch_after_frames and sample_step must be >= 1")
        return cfg


def measure(frame: np.ndarray, sample_step: int = 8) -> Tuple[float, float]:
    """(chroma, luma) of a BGR or single-channel frame, from a pixel sample.

    A 640x480 frame at step 8 is 4,800 pixels — cheap enough for every frame.
    """
    s = frame[::sample_step, ::sample_step]
    if s.ndim == 2 or s.shape[-1] == 1:
        return 0.0, float(s.mean())
    s = s.astype(np.int16)
    b, g, r = s[..., 0], s[..., 1], s[..., 2]
    chroma = (np.abs(b - g) + np.abs(g - r) + np.abs(r - b)).mean() / 3.0
    luma = (0.114 * b + 0.587 * g + 0.299 * r).mean()
    return float(chroma), float(luma)


def classify_measured(chroma: float, luma: float,
                      cfg: FrameTypeConfig) -> Optional[str]:
    """One frame's type from its measurements, or None in the dead band.

    Order matters:
      1. A near-black frame is 'dark' whatever its chroma — with nothing lit,
         R=G=B=0 is not evidence of IR. This is what a camera WITHOUT an IR
         illuminator produces at night.
      2. Near-zero chroma on a lit frame is IR.
      3. A dim colour frame is 'dark'. Sensor noise at high gain gives it some
         chroma, which is what keeps it out of step 2.
      4. Low-but-not-zero chroma is the dead band: plausibly a transition
         frame, plausibly a very grey scene. It never votes for a switch.
    """
    if luma <= cfg.black_max_luma:
        return DARK
    if chroma < cfg.mono_max_chroma:
        return IR
    if luma < cfg.dark_max_luma:
        return DARK
    if chroma < cfg.ambiguous_chroma:
        return None
    return DAY


def classify_frame(frame: np.ndarray, cfg: FrameTypeConfig) -> Optional[str]:
    chroma, luma = measure(frame, cfg.sample_step)
    return classify_measured(chroma, luma, cfg)


@dataclass(frozen=True)
class FrameTypeSnapshot:
    """What the tracker believed at one instant. Stored with every clip."""
    state: str               # day/ir/dark, or establishing/switching
    mode: Optional[str]      # last settled mode (None before the first)
    raw: str                 # this frame's own type, or 'ambiguous'
    candidate: Optional[str] # the mode being switched to, if any
    chroma: float
    luma: float

    def as_metadata(self) -> dict:
        return {
            'frame_state': self.state,
            'settled_mode': self.mode,
            'raw_frame_type': self.raw,
            'candidate_mode': self.candidate,
            'chroma': round(self.chroma, 2),
            'luma': round(self.luma, 1),
        }


class FrameTypeTracker:
    """Hysteresis over per-frame classifications.

    Works like a readiness probe's successThreshold: the mode changes only
    after `switch_after_frames` consecutive frames agree on the same NEW type.
    A frame that agrees with the current mode, or sits in the dead band,
    resets the count — so a flickering dusk sequence never switches.

    While a switch is pending (or before the first mode is agreed) the state
    is SWITCHING / ESTABLISHING. Callers must not classify those frames with
    any model — nobody knows which model applies — but must still record them.
    """

    def __init__(self, cfg: FrameTypeConfig):
        self.cfg = cfg
        self.mode: Optional[str] = None
        self._candidate: Optional[str] = None
        self._streak = 0
        self.last: Optional[FrameTypeSnapshot] = None

    @property
    def state(self) -> str:
        if self.mode is None:
            return ESTABLISHING
        if self._candidate is not None:
            return SWITCHING
        return self.mode

    def update(self, frame: np.ndarray) -> Tuple[FrameTypeSnapshot, bool]:
        """Feed one frame. Returns (snapshot, switched) — `switched` is True on
        the frame that settled a new mode, including the very first."""
        chroma, luma = measure(frame, self.cfg.sample_step)
        raw = classify_measured(chroma, luma, self.cfg)
        switched = self._step(raw)
        self.last = FrameTypeSnapshot(
            state=self.state, mode=self.mode, raw=raw or AMBIGUOUS,
            candidate=self._candidate, chroma=chroma, luma=luma,
        )
        return self.last, switched

    def _step(self, raw: Optional[str]) -> bool:
        if raw is None or raw == self.mode:
            self._candidate, self._streak = None, 0
            return False
        if raw == self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = raw, 1

        if self._streak >= self.cfg.switch_after_frames:
            previous = self.mode
            self.mode = raw
            self._candidate, self._streak = None, 0
            logger.warning(f"Frame type: {previous or 'startup'} -> {raw}")
            return True
        return False
