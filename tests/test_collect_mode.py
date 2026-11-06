#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_collect_mode.py
#
# Guards the deterrent gate and collect mode's (now narrower) job.
#
#   modes.<mode>.deterrent  — the ONE switch for the sound. A silent mode
#                             still records and alerts; it just never plays.
#   collect_mode.enabled    — lowers the detector's anchor floor to
#                             modes.day.record_threshold so borderline
#                             events surface. It has NO say over the sound.
#
# The previous design silenced the sound only when collect_mode.enabled AND
# collect_mode.deterrent_enabled=false, so a deploy with collect mode off
# fired the deterrent regardless. Two flags, one behaviour. These tests pin
# the decoupling.
#
# Run with pytest, or directly:
#     python tests/test_collect_mode.py
# ============================================================

import sys
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.modes import ModeConfig  # noqa: E402


def _mode(deterrent: bool, record=0.30):
    return ModeConfig(name='day', model='heron', deterrent=deterrent,
                      cooldown_seconds=10, deter_threshold=0.90,
                      alert_threshold=0.95, record_threshold=record)


def _build_system(collect_enabled: bool):
    """Instantiate HeronDeterrentSystem with every heavy dependency stubbed,
    so trigger_deterrent's gate can be exercised in isolation.

    Reaches directly for the __new__ trick because __init__ pulls in a
    camera, a tflite interpreter, alsa, and mqtt — none of which belong
    in a unit test that is only about a config gate.
    """
    from main import HeronDeterrentSystem

    system = HeronDeterrentSystem.__new__(HeronDeterrentSystem)
    system.collect_mode_enabled = collect_enabled
    system.audio_player = mock.Mock()
    system.mqtt_client = mock.Mock()
    return system


def test_silent_mode_is_silent_with_collect_mode_off():
    """The coupling bug: collect mode off used to mean the sound fired no
    matter what. The mode flag alone must silence it."""
    system = _build_system(collect_enabled=False)

    system.trigger_deterrent(_mode(deterrent=False))

    system.audio_player.play_random.assert_not_called()


def test_silent_mode_is_silent_with_collect_mode_on():
    system = _build_system(collect_enabled=True)

    system.trigger_deterrent(_mode(deterrent=False))

    system.audio_player.play_random.assert_not_called()


def test_armed_mode_plays_whatever_collect_mode_says():
    """Collect mode no longer has a say over the sound in either direction."""
    for collect in (False, True):
        system = _build_system(collect_enabled=collect)
        system.trigger_deterrent(_mode(deterrent=True))
        system.audio_player.play_random.assert_called_once()


def test_silenced_still_publishes_mqtt():
    """Silenced runs still emit an MQTT event so the counterfactual is visible.
    Otherwise a user could not tell 'deterrent disabled' from 'model never
    fired'."""
    system = _build_system(collect_enabled=False)

    system.trigger_deterrent(_mode(deterrent=False))

    system.mqtt_client.publish.assert_called_once()
    topic, payload = system.mqtt_client.publish.call_args[0]
    assert topic == 'detection'
    assert payload['event'] == 'deterrent_silenced', (
        f"silenced event must be distinguishable from a real fire: {payload}"
    )
    assert payload['mode'] == 'day'


def test_anchor_floor_drops_when_collect_mode_lowers_the_bar():
    """Config side: enabling collect mode with a low record_threshold must
    reach the detector. Otherwise the detector still filters at alert_threshold
    and record_threshold does nothing."""
    from main import anchor_floor

    assert anchor_floor(_mode(False), collect_enabled=False) == 0.90
    assert anchor_floor(_mode(False), collect_enabled=True) == 0.30, (
        "collect mode must lower the detector's anchor floor to "
        "record_threshold — otherwise borderline events never surface"
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
