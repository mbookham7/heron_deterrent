# ============================================================
# FILE: deterrent/audio_player.py
# ============================================================

import random
from pathlib import Path
from typing import List, Optional
import logging
import subprocess

logger = logging.getLogger(__name__)


class NoDeterrentSoundError(RuntimeError):
    """No WAV files to play. A silent deterrent is not a deterrent."""


class DeterrentCardMissing(RuntimeError):
    """audio.card names a sound card that is not there."""


def parse_asound_cards(text):
    """{card index: (id, description)} from /proc/asound/cards, whose card
    lines look like ' 2 [Headphones     ]: bcm2835_headpho - bcm2835 Headphones'."""
    cards = {}
    for line in text.splitlines():
        head, _, rest = line.partition(']:')
        if '[' not in head or not head.split('[')[0].strip().isdigit():
            continue
        index, card_id = head.split('[', 1)
        cards[int(index)] = (card_id.strip(), rest.split(' - ', 1)[-1].strip())
    return cards


class AudioPlayer:
    CARDS_FILE = Path('/proc/asound/cards')

    def __init__(self, sound_path: str, volume: int = 80, card: Optional[str] = None):
        self.sound_path = Path(sound_path)
        self.volume = volume
        # Named card (e.g. 'Headphones', the Pi's 3.5 mm jack), or None for
        # ALSA's default. Named is safer: on a Pi the default is HDMI, and it
        # depends on /etc/asound.conf, which a desktop or a re-image can lose.
        self.card = card
        self.sound_files: List[Path] = []
        self._load_sound_files()
        self._check_card()

    def _check_card(self):
        if not self.card:
            return
        cards_file = self.CARDS_FILE
        if not cards_file.exists():
            logger.warning(f"audio.card is '{self.card}' but {cards_file} does not exist "
                           f"— cannot check the card is there")
            return
        ids = [card_id for card_id, _ in parse_asound_cards(cards_file.read_text()).values()]
        if self.card not in ids:
            raise DeterrentCardMissing(
                f"audio.card is '{self.card}' but there is no such sound card — the "
                f"deterrent would fire silently.\n"
                f"  Cards present: {', '.join(ids) or 'none'}  (see: aplay -l)\n"
                f"  Fix: edit config.yaml -> audio.card (null = ALSA's default device)"
            )

    def _load_sound_files(self):
        # Refuse to start silent: the whole point of the device is the noise.
        if not self.sound_path.exists():
            raise NoDeterrentSoundError(
                f"Sound directory does not exist: {self.sound_path}\n"
                f"  Fix: mkdir -p {self.sound_path} and put at least one .wav in it."
            )

        self.sound_files = sorted(self.sound_path.glob('*.wav'))
        if not self.sound_files:
            raise NoDeterrentSoundError(
                f"No .wav files in {self.sound_path} — the deterrent would fire silently.\n"
                f"  Fix: copy at least one WAV in, e.g.\n"
                f"    cp /path/to/heron_distress.wav {self.sound_path}\n"
                f"  The repo ships sounds/startle_default.wav as a floor; if it is missing, "
                f"restore it with: git checkout sounds/"
            )

        logger.info(
            f"Loaded {len(self.sound_files)} deterrent sound(s): "
            f"{', '.join(f.name for f in self.sound_files)}"
        )
        logger.info(f"Deterrent output: "
                    f"{'card ' + self.card if self.card else 'ALSA default device'}")

    def play_random(self):
        sound_file = random.choice(self.sound_files)
        self.play(sound_file)

    def play(self, sound_file: Path):
        try:
            logger.info(f"Playing deterrent sound: {sound_file.name}")

            volume_percent = min(100, max(0, self.volume))

            # Pi 4 uses 'PCM' or 'Headphone'; other systems may use 'Master'.
            # Volume goes to the same card the sound does.
            card_args = ['-c', self.card] if self.card else []
            for control in ['Master', 'PCM', 'Headphone']:
                result = subprocess.run(
                    ['amixer', *card_args, 'set', control, f'{volume_percent}%'],
                    capture_output=True,
                    check=False,
                    timeout=5
                )
                if result.returncode == 0:
                    break

            device_args = ['-D', f'plughw:CARD={self.card}'] if self.card else []
            subprocess.run(
                ['aplay', *device_args, str(sound_file)],
                capture_output=True,
                check=True,
                timeout=30  # WAV files should never take more than 30 seconds
            )

            logger.info("Sound played successfully")

        except subprocess.TimeoutExpired:
            logger.error("Audio playback timed out — killing hung process")
        except FileNotFoundError:
            logger.error("aplay not found. Install alsa-utils: sudo apt-get install alsa-utils")
        except subprocess.CalledProcessError as e:
            logger.error(f"aplay returned non-zero exit code: {e.returncode}")
        except Exception as e:
            logger.error(f"Failed to play sound: {e}")
