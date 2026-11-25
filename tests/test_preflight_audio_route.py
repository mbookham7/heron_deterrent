#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_preflight_audio_route.py
#
# On a Raspberry Pi, ALSA's default output is HDMI. The deterrent then
# plays faithfully to a monitor that is not there: aplay succeeds, and
# preflight used to PASS while naming the first card in the list rather
# than the one that played. This pins the replacement: parse which card
# is which, and fail when the sound went to HDMI.
#
# Run with pytest, or directly:
#     python tests/test_preflight_audio_route.py
# ============================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preflight import audio_route_verdict, parse_asound_cards  # noqa: E402

# /proc/asound/cards, verbatim, from the build Pi (Pi 4, Raspberry Pi OS 13).
PI4_CARDS = """\
 0 [vc4hdmi0       ]: vc4-hdmi - vc4-hdmi-0
                      vc4-hdmi-0
 1 [vc4hdmi1       ]: vc4-hdmi - vc4-hdmi-1
                      vc4-hdmi-1
 2 [Headphones     ]: bcm2835_headpho - bcm2835 Headphones
                      bcm2835 Headphones
"""


def test_parses_every_card_and_skips_continuation_lines():
    assert parse_asound_cards(PI4_CARDS) == {
        0: ('vc4hdmi0', 'vc4-hdmi-0'),
        1: ('vc4hdmi1', 'vc4-hdmi-1'),
        2: ('Headphones', 'bcm2835 Headphones'),
    }


def test_empty_or_garbage_parses_to_nothing():
    assert parse_asound_cards('') == {}
    assert parse_asound_cards('--- no soundcards ---\n') == {}


def test_hdmi_fails():
    assert audio_route_verdict('vc4hdmi0') == 'fail'
    assert audio_route_verdict('vc4hdmi1') == 'fail'


def test_the_jack_passes():
    assert audio_route_verdict('Headphones') == 'pass'


def test_a_usb_speaker_passes():
    assert audio_route_verdict('Device') == 'pass'


def test_unseen_route_is_unknown_not_pass():
    """If nothing was seen RUNNING, say so — do not claim a device."""
    assert audio_route_verdict(None) == 'unknown'


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
