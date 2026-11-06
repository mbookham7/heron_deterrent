# ============================================================
# FILE: utils/modes.py
#
# Per-frame-type operating modes: what the system does when the
# camera is producing day, IR, or dark frames.
#
# Each mode says, explicitly:
#   model           — a registered model name, or null. Null is a
#                     first-class state: motion-only, no inference.
#   *_threshold     — required with a model, forbidden without one.
#   deterrent       — whether the sound MAY fire. A mode can only
#                     restrict the deterrent, never enable it beyond
#                     what the model decides. This is the ONE switch
#                     for the sound; nothing else gates it.
#   cooldown_seconds— minimum gap between motion triggers.
#   shadow_model    — motion-only modes only: run this model, LOG its
#                     confidence, never act on it.
#
# Incoherent config is a startup error, never a silent default.
# ============================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ai.model_loader import MODELS, TARGETS, model_path
from capture.frame_classifier import FRAME_TYPES, FrameTypeConfig


class ModeConfigError(RuntimeError):
    """The mode configuration is incoherent. The message is shown to the operator."""


# Keys from before per-mode config. Each would be a second source of truth for
# something modes.* now owns, so a leftover is an error rather than ignored.
LEGACY_KEYS = {
    'system.active_hours': "removed — the system runs round the clock and "
                           "branches on frame type. Delete the block.",
    'ai.deter_threshold': "moved to modes.day.deter_threshold",
    'ai.alert_threshold': "moved to modes.day.alert_threshold",
    'ai.confidence_threshold': "removed — use modes.day.deter_threshold / alert_threshold",
    'collect_mode.record_threshold': "moved to modes.day.record_threshold",
    'collect_mode.deterrent_enabled': "replaced by modes.<mode>.deterrent, which "
                                      "is now the only switch for the sound",
}

_THRESHOLDS = ('deter_threshold', 'alert_threshold', 'record_threshold')
_ALLOWED = {'model', 'deterrent', 'cooldown_seconds', 'shadow_model', *_THRESHOLDS}


@dataclass(frozen=True)
class ModeConfig:
    name: str
    model: Optional[str]
    deterrent: bool
    cooldown_seconds: float
    deter_threshold: Optional[float] = None
    alert_threshold: Optional[float] = None
    record_threshold: Optional[float] = None
    shadow_model: Optional[str] = None

    @property
    def motion_only(self) -> bool:
        return self.model is None

    def describe(self) -> str:
        sound = 'deterrent ON' if self.deterrent else 'silent'
        if self.motion_only:
            shadow = f", shadow-logging {self.shadow_model}" if self.shadow_model else ""
            return (f"{self.name}: motion-only (no model){shadow}, {sound}, "
                    f"cooldown {self.cooldown_seconds:g}s")
        return (f"{self.name}: model={self.model} deter={self.deter_threshold} "
                f"alert={self.alert_threshold} record={self.record_threshold}, "
                f"{sound}, cooldown {self.cooldown_seconds:g}s")


def _is_set(config, dotted: str) -> bool:
    node = config.config
    for k in dotted.split('.'):
        if not isinstance(node, dict) or k not in node:
            return False
        node = node[k]
    return True


def _unit_interval(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v <= 1


def _parse_mode(name: str, raw, collect_enabled: bool, errors: List[str]) -> Optional[ModeConfig]:
    where = f"modes.{name}"
    if not isinstance(raw, dict):
        errors.append(f"{where} must be a mapping")
        return None

    unknown = set(raw) - _ALLOWED
    if unknown:
        errors.append(f"{where}: unknown key(s) {sorted(unknown)}")

    # `model` must be written out, even as null: a forgotten key and a
    # deliberate "no model" must not look the same.
    if 'model' not in raw:
        errors.append(f"{where}.model is missing — set a model name, or null for motion-only")
    model = raw.get('model')
    if model is not None and model not in MODELS:
        errors.append(f"{where}.model is {model!r}; registered models: {', '.join(sorted(MODELS))}")

    deterrent = raw.get('deterrent')
    if not isinstance(deterrent, bool):
        errors.append(f"{where}.deterrent must be true or false (got {deterrent!r})")

    cooldown = raw.get('cooldown_seconds')
    if not isinstance(cooldown, (int, float)) or isinstance(cooldown, bool) or cooldown <= 0:
        errors.append(f"{where}.cooldown_seconds must be a positive number (got {cooldown!r})")

    shadow = raw.get('shadow_model')

    if model is None:
        present = [t for t in _THRESHOLDS if t in raw]
        if present:
            errors.append(f"{where} has no model, so {', '.join(present)} would be "
                          f"ignored — remove them")
        if deterrent is True:
            errors.append(f"{where}.deterrent is true but the mode has no model: the "
                          f"sound would fire on every moth. Motion-only modes must be silent.")
        if shadow is not None and shadow not in MODELS:
            errors.append(f"{where}.shadow_model is {shadow!r}; registered models: "
                          f"{', '.join(sorted(MODELS))}")
    else:
        for t in ('deter_threshold', 'alert_threshold'):
            if not _unit_interval(raw.get(t)):
                errors.append(f"{where}.{t} must be in (0, 1] (got {raw.get(t)!r})")
        if 'record_threshold' in raw and not _unit_interval(raw['record_threshold']):
            errors.append(f"{where}.record_threshold must be in (0, 1]")
        if collect_enabled and 'record_threshold' not in raw:
            errors.append(f"collect_mode.enabled is true but {where}.record_threshold is unset")
        if shadow is not None:
            errors.append(f"{where}.shadow_model is only for motion-only modes — "
                          f"this mode already runs {model!r}")

    if errors:
        return None
    return ModeConfig(
        name=name, model=model, deterrent=deterrent, cooldown_seconds=float(cooldown),
        deter_threshold=raw.get('deter_threshold'),
        alert_threshold=raw.get('alert_threshold'),
        record_threshold=raw.get('record_threshold'),
        shadow_model=shadow,
    )


def load_modes(config) -> Dict[str, ModeConfig]:
    """Parse and validate `modes:`. Raises ModeConfigError listing every
    problem at once, so a broken config is fixed in one pass."""
    errors = []
    for key, why in LEGACY_KEYS.items():
        if _is_set(config, key):
            errors.append(f"{key} is set but {why}")

    raw_modes = config.get('modes')
    if not isinstance(raw_modes, dict):
        errors.append("modes: block is missing — see config.yaml in the repo for the schema")
        raise ModeConfigError(_format(errors))

    missing = [m for m in FRAME_TYPES if m not in raw_modes]
    extra = [m for m in raw_modes if m not in FRAME_TYPES]
    if missing:
        errors.append(f"modes: missing {', '.join(missing)} — every frame type needs a mode")
    if extra:
        errors.append(f"modes: unknown mode(s) {', '.join(extra)}; "
                      f"valid: {', '.join(FRAME_TYPES)}")

    collect_enabled = bool(config.get('collect_mode.enabled', False))
    modes = {}
    for name in FRAME_TYPES:
        if name in raw_modes:
            mode_errors: List[str] = []
            mode = _parse_mode(name, raw_modes[name], collect_enabled, mode_errors)
            errors += mode_errors
            if mode:
                modes[name] = mode

    if errors:
        raise ModeConfigError(_format(errors))
    return modes


def load_frame_type_config(config) -> FrameTypeConfig:
    try:
        return FrameTypeConfig.from_dict(config.get('frame_type'))
    except (TypeError, ValueError) as e:
        raise ModeConfigError(_format([str(e)])) from e


def missing_model_files(modes: Dict[str, ModeConfig], target: str,
                        model_dir: str = './models') -> List[str]:
    """One message per mode whose model (or shadow model) has no file for the
    current ai.target. Used by preflight; startup finds the same thing by
    trying to load."""
    out = []
    if target not in TARGETS:
        return out  # reported separately as a bad ai.target
    for mode in modes.values():
        for role, name in (('model', mode.model), ('shadow_model', mode.shadow_model)):
            if name is None:
                continue
            path = model_path(name, target, model_dir)
            if not Path(path).exists():
                out.append(f"modes.{mode.name}.{role} is {name!r}, which on target "
                           f"{target!r} needs {path} — that file does not exist")
    return out


def _format(errors: List[str]) -> str:
    return "Mode configuration is invalid:\n" + "\n".join(f"  - {e}" for e in errors) + \
        "\n  Fix: edit config.yaml (modes:, frame_type:), then run ./preflight.py"
