#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_modes.py
#
# Guards per-frame-type operation:
#
#   1. Config: the shipped config.yaml is valid and silent everywhere;
#      every incoherent combination is a startup error, not a default.
#   2. Branching: night and unsettled triggers run NO inference, never
#      touch the deterrent, and are still recorded and alerted — as
#      'motion', with the frame state they were captured in.
#   3. Labelling: a frame is filed under the domain it belongs to, or
#      refused. Never guessed.
#
# Run with pytest, or directly:
#     python tests/test_modes.py
# ============================================================

import copy
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.detector import Detection  # noqa: E402
from capture.frame_classifier import (  # noqa: E402
    FrameTypeConfig, FrameTypeSnapshot, FrameTypeTracker, resolve_label_domain,
)
from utils.config_loader import Config  # noqa: E402
from utils.modes import (  # noqa: E402
    ModeConfigError, load_modes, missing_model_files,
)

SHIPPED = yaml.safe_load((ROOT / 'config.yaml').read_text())


def _config(mutate=None) -> Config:
    data = copy.deepcopy(SHIPPED)
    if mutate:
        mutate(data)
    tmp = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False)
    yaml.safe_dump(data, tmp)
    tmp.close()
    return Config(tmp.name)


def _rejects(mutate, fragment):
    try:
        load_modes(_config(mutate))
    except ModeConfigError as e:
        assert fragment in str(e), f"error did not mention {fragment!r}:\n{e}"
        return
    raise AssertionError(f"config accepted; expected error about {fragment!r}")


# --------------------------------------------------------------------------
# 1. Config
# --------------------------------------------------------------------------

def test_shipped_config_is_valid_and_silent_everywhere():
    modes = load_modes(_config())
    assert set(modes) == {'day', 'ir', 'dark'}
    assert all(not m.deterrent for m in modes.values()), "the deterrent must be silent in every mode"
    assert modes['day'].model == 'heron'
    assert modes['ir'].motion_only and modes['dark'].motion_only
    assert modes['ir'].shadow_model is None, "shadow inference must default to off"


def test_shipped_day_thresholds_unchanged():
    day = load_modes(_config())['day']
    assert (day.deter_threshold, day.alert_threshold) == (0.90, 0.95)


def test_night_cooldown_is_longer_than_day():
    modes = load_modes(_config())
    assert modes['ir'].cooldown_seconds >= 30 and modes['dark'].cooldown_seconds >= 30
    assert modes['day'].cooldown_seconds < modes['ir'].cooldown_seconds


def test_motion_only_mode_cannot_arm_the_deterrent():
    _rejects(lambda d: d['modes']['ir'].update(deterrent=True), 'every moth')


def test_leftover_legacy_keys_are_errors():
    cases = [
        (lambda d: d['system'].update(active_hours={'start': '06:00', 'end': '20:00'}),
         'system.active_hours'),
        (lambda d: d['ai'].update(deter_threshold=0.9), 'ai.deter_threshold'),
        (lambda d: d['ai'].update(confidence_threshold=0.9), 'ai.confidence_threshold'),
        (lambda d: d['collect_mode'].update(deterrent_enabled=False),
         'collect_mode.deterrent_enabled'),
    ]
    for mutate, key in cases:
        _rejects(mutate, key)


def test_model_key_must_be_explicit():
    """A forgotten model key and a deliberate null must not look the same."""
    _rejects(lambda d: d['modes']['dark'].pop('model'), 'modes.dark.model is missing')


def test_unregistered_model_is_an_error():
    _rejects(lambda d: d['modes']['ir'].update(model='heron_ir'), "'heron_ir'")


def test_thresholds_on_a_motion_only_mode_are_an_error():
    _rejects(lambda d: d['modes']['ir'].update(deter_threshold=0.5), 'would be ignored')


def test_model_mode_needs_thresholds():
    _rejects(lambda d: d['modes']['day'].pop('alert_threshold'), 'modes.day.alert_threshold')


def test_shadow_model_only_on_motion_only_modes():
    _rejects(lambda d: d['modes']['day'].update(shadow_model='heron'), 'only for motion-only')


def test_every_frame_type_needs_a_mode():
    _rejects(lambda d: d['modes'].pop('dark'), 'missing dark')


def test_missing_model_file_is_reported():
    modes = load_modes(_config())
    with tempfile.TemporaryDirectory() as empty:
        problems = missing_model_files(modes, 'cpu', model_dir=empty)
    assert len(problems) == 1 and 'modes.day.model' in problems[0]


# --------------------------------------------------------------------------
# 2. Branching in the detection path
# --------------------------------------------------------------------------

def _snapshot(state, mode=None, raw=None):
    return FrameTypeSnapshot(state=state, mode=mode, raw=raw or state,
                             candidate=None, chroma=0.5, luma=90.0)


def _system(mutate=None, shadow_conf=None):
    from main import HeronDeterrentSystem
    s = HeronDeterrentSystem.__new__(HeronDeterrentSystem)
    s.modes = load_modes(_config(mutate))
    s.collect_mode_enabled = False
    s.alert_vote_k, s.alert_vote_m = 3, 5
    s.unsettled_cooldown_seconds = 10
    s.detectors = {'day': mock.Mock()}
    s.shadow_detectors = {}
    if shadow_conf is not None:
        sd = mock.Mock()
        sd.infer.return_value = Detection('heron', shadow_conf)
        s.shadow_detectors = {'ir': sd}
    s.media_store = mock.Mock()
    s.media_store.save_video_clip.return_value = '/media/clip.mp4'
    s.alert_manager = mock.Mock()
    s.database = mock.Mock()
    s.audio_player = mock.Mock()
    s.mqtt_client = mock.Mock()
    s._detection_semaphore = mock.Mock()
    s.frame_tracker = FrameTypeTracker(FrameTypeConfig())
    s.capture_video_clip = mock.Mock(return_value=[np.full((48, 64, 3), 100, np.uint8)] * 5)
    return s


def _assert_motion_only(s, frame_type):
    s.detectors['day'].infer.assert_not_called()
    s.audio_player.play_random.assert_not_called()
    s.capture_video_clip.assert_called_once()
    args, kwargs = s.media_store.save_video_clip.call_args
    assert args[1] == 'motion' and kwargs['frame_type'] == frame_type
    s.alert_manager.create_motion_alert.assert_called_once()
    ft, _path, md = s.alert_manager.create_motion_alert.call_args[0]
    assert ft == frame_type and md['trigger'] == 'motion_only'
    s.alert_manager.create_alert.assert_not_called()
    s._detection_semaphore.release.assert_called_once()
    return md


def test_ir_trigger_runs_no_inference_and_is_recorded():
    s = _system()
    s._handle_motion_event(np.zeros((4, 4, 3), np.uint8), _snapshot('ir', 'ir'))
    _assert_motion_only(s, 'ir')


def test_dark_trigger_is_treated_like_night():
    s = _system()
    s._handle_motion_event(np.zeros((4, 4, 3), np.uint8), _snapshot('dark', 'dark'))
    _assert_motion_only(s, 'dark')


def test_switching_trigger_is_recorded_unclassified_not_dropped():
    """Dawn: mode still 'day' but frames say IR. Nobody knows which model
    applies, so none runs — but the clip is kept, labelled with the state."""
    s = _system()
    snap = FrameTypeSnapshot(state='switching', mode='day', raw='ir',
                             candidate='ir', chroma=0.4, luma=80.0)
    s._handle_motion_event(np.zeros((4, 4, 3), np.uint8), snap)
    md = _assert_motion_only(s, 'switching')
    assert md['settled_mode'] == 'day' and md['raw_frame_type'] == 'ir'


def test_establishing_trigger_is_recorded_with_ambiguous_raw_type():
    s = _system()
    snap = FrameTypeSnapshot(state='establishing', mode=None, raw='ambiguous',
                             candidate=None, chroma=2.0, luma=80.0)
    s._handle_motion_event(np.zeros((4, 4, 3), np.uint8), snap)
    md = _assert_motion_only(s, 'establishing')
    assert md['raw_frame_type'] == 'ambiguous'


def test_shadow_inference_is_logged_and_never_acted_on():
    s = _system(lambda d: d['modes']['ir'].update(shadow_model='heron'), shadow_conf=0.99)
    s._handle_motion_event(np.zeros((4, 4, 3), np.uint8), _snapshot('ir', 'ir'))
    md = _assert_motion_only(s, 'ir')
    assert md['shadow_confidence'] == 0.99 and md['shadow_model'] == 'heron'
    s.shadow_detectors['ir'].infer.assert_called_once()


def test_silent_day_heron_still_records_and_alerts():
    """A confident daylight heron with the deterrent off: no sound, but the
    clip and the CRITICAL alert happen exactly as before."""
    s = _system()
    s.detectors['day'].infer.return_value = Detection('heron', 0.97, (0.1, 0.1, 0.2, 0.2))
    s._handle_motion_event(np.zeros((4, 4, 3), np.uint8), _snapshot('day', 'day'))

    s.audio_player.play_random.assert_not_called()
    args, kwargs = s.media_store.save_video_clip.call_args
    assert args[1] == 'heron' and kwargs['frame_type'] == 'day'
    alert_type = s.alert_manager.create_alert.call_args[0][0]
    assert alert_type == 'CRITICAL'
    assert s.alert_manager.create_alert.call_args[1]['frame_type'] == 'day'


def test_cooldown_is_per_mode():
    s = _system()
    assert s.cooldown_for('day') == 10
    assert s.cooldown_for('ir') == 45
    assert s.cooldown_for('switching') == 10
    assert s.cooldown_for('establishing') == 10


# --------------------------------------------------------------------------
# 3. Which labelled-data tree a frame goes in
# --------------------------------------------------------------------------

def test_settled_clip_and_agreeing_frame_files_under_that_domain():
    assert resolve_label_domain('ir', 'ir')[0] == 'ir'
    assert resolve_label_domain('day', 'day')[0] == 'day'


def test_settled_clip_with_disagreeing_frame_is_refused():
    """An IR frame inside a clip recorded in day mode (it straddled the cut)
    must not land in the daylight set."""
    for recorded in ('ir', 'dark', 'ambiguous'):
        domain, why = resolve_label_domain('day', recorded)
        assert domain is None and 'transition' in why


def test_operator_cannot_override_a_settled_clip():
    assert resolve_label_domain('ir', 'ir', requested='day')[0] is None
    assert resolve_label_domain('day', None, requested='ir')[0] is None


def test_unsettled_clip_is_never_filed_without_operator_choice():
    for row in ('switching', 'establishing', None):
        domain, why = resolve_label_domain(row, 'ir')
        assert domain is None and 'confirm' in why
        assert resolve_label_domain(row, 'ir', requested='ir')[0] == 'ir'


def test_operator_choice_must_match_the_recorded_frame():
    assert resolve_label_domain('switching', 'ir', requested='day')[0] is None


def test_ambiguous_frame_from_unsettled_clip_cannot_be_filed():
    for req in (None, 'day', 'ir', 'dark'):
        assert resolve_label_domain('switching', 'ambiguous', requested=req)[0] is None


def test_no_record_needs_operator_confirmation():
    """Legacy rows and uploads have no capture-time record. The decoded frame
    cannot be trusted (see test_frame_classifier codec-cast test), so only a
    human can place it."""
    assert resolve_label_domain(None, None)[0] is None
    assert resolve_label_domain(None, None, requested='dark')[0] == 'dark'
    assert resolve_label_domain('day', None)[0] is None
    assert resolve_label_domain('day', None, requested='day')[0] == 'day'


def test_every_capture_path_records_per_frame_types():
    """Labelling depends on the capture-time record; every path that saves a
    clip must write it, one entry per frame."""
    mono = np.full((48, 64, 3), 100, np.uint8)

    s = _system()
    s._handle_motion_event(mono, _snapshot('ir', 'ir'))
    md = s.alert_manager.create_motion_alert.call_args[0][2]
    assert md['clip_frame_types'] == ['ir'] * 5

    s = _system()
    s.detectors['day'].infer.return_value = Detection('heron', 0.97)
    s._handle_motion_event(mono, _snapshot('day', 'day'))
    md = s.alert_manager.create_alert.call_args[1]['metadata']
    assert len(md['clip_frame_types']) == 5

    s = _system()
    s.detectors['day'].infer.return_value = Detection('unknown', 0.0)
    s.detectors['day'].aggregate_detections.return_value = Detection('unknown', 0.0)
    s._handle_motion_event(mono, _snapshot('day', 'day'))
    md = s.alert_manager.create_alert.call_args[1]['metadata']
    assert len(md['clip_frame_types']) == 5


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
