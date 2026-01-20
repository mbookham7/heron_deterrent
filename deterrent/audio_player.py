# ============================================================
# FILE: deterrent/audio_player.py
# ============================================================

import os
import random
from pathlib import Path
from typing import List
import logging
import subprocess

logger = logging.getLogger(__name__)

class AudioPlayer:
    def __init__(self, sound_path: str, volume: int = 80):
        self.sound_path = Path(sound_path)
        self.volume = volume
        self.sound_files = []
        self._load_sound_files()
    
    def _load_sound_files(self):
        if not self.sound_path.exists():
            logger.warning(f"Sound directory not found: {self.sound_path}")
            self.sound_path.mkdir(parents=True, exist_ok=True)
            return
        
        self.sound_files = list(self.sound_path.glob('*.wav'))
        logger.info(f"Loaded {len(self.sound_files)} sound files")
    
    def play_random(self):
        if not self.sound_files:
            logger.warning("No sound files available")
            return
        
        sound_file = random.choice(self.sound_files)
        self.play(sound_file)
    
    def play(self, sound_file: Path):
        try:
            logger.info(f"Playing deterrent sound: {sound_file.name}")
            
            # Use aplay on Linux (Raspberry Pi / Coral)
            # Adjust volume (0-100)
            volume_percent = min(100, max(0, self.volume))
            
            # Set volume using amixer
            subprocess.run(
                ['amixer', 'set', 'Master', f'{volume_percent}%'],
                capture_output=True,
                check=False
            )
            
            # Play sound
            subprocess.run(
                ['aplay', str(sound_file)],
                capture_output=True,
                check=True
            )
            
            logger.info("Sound played successfully")
            
        except FileNotFoundError:
            logger.error("aplay not found. Install alsa-utils")
        except Exception as e:
            logger.error(f"Failed to play sound: {e}")