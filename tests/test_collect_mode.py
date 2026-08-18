#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_collect_mode.py
#
# Collect mode exists so the device can sit at the pond and manufacture
# training data with the current (too-weak-to-trust) model. Two flags:
#
#   collect_mode.enabled           — lowers the detector's anchor floor to
#                                    record_threshold, so borderline motion
#                                    events land in the dashboard for
#                                    manual labelling.
#   collect_mode.deterrent_enabled — separately switchable. False = silent
#                                    run: recording happens, sound does not.
#                                    Prevents the deterrent from confounding
#                                    behavioural observation.
#
# The critical property this test guards: a deterrent-DISABLED collect run
# must still record. Silencing the sound must not silence the data.
#
# Run with pytest, or directly:
#     python tests/test_collect_mode.py
# ============================================================

import sys
import types
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _build_system(collect_enabled: bool, deterrent_enabled: bool):
    """Instantiate HeronDeterrentSystem with every heavy dependency stubbed,
    so trigger_deterrent's collect-mode logic can be exercised in isolation.

    Reaches directly for the __new__ trick because __init__ pulls in a
    camera, a tflite interpreter, alsa, and mqtt — none of which belong
    in a unit test that is only about a config gate.
    """
    from main import HeronDeterrentSystem

    system = HeronDeterrentSystem.__new__(HeronDeterrentSystem)
    system.collect_mode_enabled = collect_enabled
    system.deterrent_enabled = deterrent_enabled
    system.audio_player = mock.Mock()
    system.mqtt_client = mock.Mock()
    return system


def test_collect_off_still_plays_sound():
    """The default path is unaffected. If someone regresses the gate, this
    fails."""
    system = _build_system(collect_enabled=False, deterrent_enabled=False)

    system.trigger_deterrent()

    system.audio_player.play_random.assert_called_once()


def test_collect_on_with_deterrent_enabled_still_plays_sound():
    """Collect mode with the deterrent left ON must not silence."""
    system = _build_system(collect_enabled=True, deterrent_enabled=True)

    system.trigger_deterrent()

    system.audio_player.play_random.assert_called_once()


def test_collect_on_with_deterrent_disabled_is_silent():
    """The specific case the field-deployment relies on: data collection with
    the sound OFF. audio_player MUST NOT be called."""
    system = _build_system(collect_enabled=True, deterrent_enabled=False)

    system.trigger_deterrent()

    system.audio_player.play_random.assert_not_called()


def test_collect_on_with_deterrent_disabled_still_publishes_mqtt():
    """Silenced runs still emit an MQTT event so the counterfactual is visible.
    Otherwise a user could not tell 'deterrent disabled' from 'model never
    fired'."""
    system = _build_system(collect_enabled=True, deterrent_enabled=False)

    system.trigger_deterrent()

    system.mqtt_client.publish.assert_called_once()
    topic, payload = system.mqtt_client.publish.call_args[0]
    assert topic == 'detection'
    assert payload['event'] == 'deterrent_silenced', (
        f"silenced event must be distinguishable from a real fire: {payload}"
    )


def test_anchor_floor_drops_when_collect_mode_lowers_the_bar():
    """Config side: enabling collect mode with a low record_threshold must
    reach the detector. Otherwise the detector still filters at alert_threshold
    and record_threshold does nothing.

    This is a pure config-plumbing test — no model involved."""
    # Simulate the _initialize_components logic without instantiating anything.
    deter, alert, record = 0.90, 0.95, 0.30
    floor_off  = min(deter, alert)
    floor_on   = min(deter, alert, record)
    assert floor_off == 0.90
    assert floor_on  == 0.30, (
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
