#!/usr/bin/env python3
# ============================================================
# FILE: tests/test_audio_card.py
#
# audio.card names the sound card the deterrent plays to, so it no longer
# depends on ALSA's default — which on a Pi is HDMI, and is only fixed by
# /etc/asound.conf. These tests pin that the card reaches both aplay and
# amixer, that null keeps the old behaviour, and that a card which is not
# there refuses to start rather than playing to nothing.
#
# Run with pytest, or directly:
#     python tests/test_audio_card.py
# ============================================================

import sys
import tempfile
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deterrent.audio_player import AudioPlayer, DeterrentCardMissing  # noqa: E402

SOUNDS = str(Path(__file__).resolve().parent.parent / 'sounds')

# /proc/asound/cards, verbatim, from the build Pi.
PI4_CARDS = """\
 0 [vc4hdmi0       ]: vc4-hdmi - vc4-hdmi-0
                      vc4-hdmi-0
 1 [vc4hdmi1       ]: vc4-hdmi - vc4-hdmi-1
                      vc4-hdmi-1
 2 [Headphones     ]: bcm2835_headpho - bcm2835 Headphones
                      bcm2835 Headphones
"""


def _player(card, cards_text=PI4_CARDS):
    with tempfile.NamedTemporaryFile('w', suffix='.cards', delete=False) as f:
        f.write(cards_text)
    with mock.patch.object(AudioPlayer, 'CARDS_FILE', Path(f.name)):
        return AudioPlayer(SOUNDS, 80, card)


def _commands(player):
    with mock.patch('deterrent.audio_player.subprocess.run') as run:
        run.return_value = mock.Mock(returncode=0)
        player.play_random()
    return [c.args[0] for c in run.call_args_list]


def test_named_card_reaches_aplay_and_amixer():
    cmds = _commands(_player('Headphones'))
    amixer, aplay = cmds[0], cmds[-1]
    assert amixer[:3] == ['amixer', '-c', 'Headphones']
    assert aplay[:3] == ['aplay', '-D', 'plughw:CARD=Headphones']


def test_null_card_uses_the_alsa_default():
    cmds = _commands(_player(None))
    assert cmds[0][:2] == ['amixer', 'set']
    assert cmds[-1][0] == 'aplay' and '-D' not in cmds[-1]


def test_missing_card_refuses_to_start():
    try:
        _player('USBSpeaker')
    except DeterrentCardMissing as e:
        assert 'Headphones' in str(e)          # names what IS there
    else:
        raise AssertionError("a missing card must refuse to start, not play silently")


def test_card_ids_are_exact():
    """ALSA card ids are case-sensitive; 'headphones' is not a card here."""
    try:
        _player('headphones')
    except DeterrentCardMissing:
        pass
    else:
        raise AssertionError("expected DeterrentCardMissing")


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
