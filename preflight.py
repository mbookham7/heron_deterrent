#!/usr/bin/env python3
"""Pre-start checks for the Heron Deterrent System.

Run this on the board before starting the service. Every check prints PASS or
FAIL; every failure prints the command that fixes it.

    ./preflight.py               # all checks, plays a deterrent sound
    ./preflight.py --skip-audio  # same, but stay quiet

Exit status is 0 only if every check passed.
"""

import argparse
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Disk floor is relative, because partition sizes vary wildly: Mendel puts
# /home on its own 2 GB partition, so a fixed 2 GB floor could never pass.
MIN_FREE_GB = 0.5
MIN_FREE_FRACTION = 0.10

_COLOUR = sys.stdout.isatty() and os.getenv('NO_COLOR') is None
GREEN = '\033[32m' if _COLOUR else ''
RED = '\033[31m' if _COLOUR else ''
DIM = '\033[2m' if _COLOUR else ''
BOLD = '\033[1m' if _COLOUR else ''
OFF = '\033[0m' if _COLOUR else ''

results = []


def check(name, ok, detail, *hints):
    """Record and print one result. hints are printed only on failure."""
    tag = f"{GREEN}[PASS]{OFF}" if ok else f"{RED}[FAIL]{OFF}"
    print(f"{tag} {name:<12} {detail}")
    if not ok:
        for hint in hints:
            for line in str(hint).splitlines():
                print(f"       {DIM}->{OFF} {line}")
    results.append((name, ok))
    return ok


# --------------------------------------------------------------------------
# Checks. Each returns whatever later checks need, or None on failure.
# --------------------------------------------------------------------------

def check_config():
    try:
        from utils.config_loader import Config
        config = Config('config.yaml')
    except Exception as e:
        check('config', False, f"cannot load config.yaml: {e}",
              "Run preflight from the repo root: cd ~/heron_deterrent && ./preflight.py")
        return None

    from ai.model_loader import TARGETS
    target = config.get('ai.target')
    if target not in TARGETS:
        check('config', False, f"ai.target is {target!r}",
              f"Valid targets: {', '.join(sorted(TARGETS))}",
              "Fix: edit config.yaml -> ai.target")
        return None

    labels = config.get('ai.labels', {})
    check('config', True,
          f"ai.target={target}  confidence={config.get('ai.confidence_threshold')}  "
          f"classes={len(labels)}")
    return config


def check_model(config):
    from ai.model_loader import TARGETS, ModelLoadError, ModelLoader
    target = config.get('ai.target')
    filename, needs_tpu, source = TARGETS[target]

    try:
        loader = ModelLoader(target, config.get('ai.num_threads', 4))
    except ModelLoadError as e:
        check('model', False, f"{filename} will not load on target '{target}'", e)
        return None
    except Exception as e:
        check('model', False, f"unexpected error loading {filename}: {e}")
        return None

    shape = 'x'.join(str(int(v)) for v in loader.input_details[0]['shape'])
    backend = 'Edge TPU' if needs_tpu else f"CPU/{config.get('ai.num_threads', 4)} threads"
    check('model', True, f"{filename} loaded on {backend}, input {shape}")
    return loader


def check_edge_tpu(config, model_loaded):
    """Only meaningful when the Coral target is selected."""
    if config.get('ai.target') != 'edgetpu':
        print(f"{DIM}[skip] edgetpu      ai.target is not 'edgetpu'{OFF}")
        return True

    try:
        from pycoral.utils.edgetpu import list_edge_tpus
        tpus = list_edge_tpus()
        if tpus:
            return check('edgetpu', True, f"{len(tpus)} device(s): {tpus}")
        return check('edgetpu', False, "no Edge TPU devices enumerated",
                     "Is the TPU visible?  lspci -nn | grep 089a",
                     "Is the driver up?    dmesg | grep apex",
                     "Reinstall runtime:   sudo apt-get install --reinstall libedgetpu1-std")
    except ImportError:
        # No pycoral, so we cannot enumerate devices independently. The model
        # check is then the only evidence — never claim more than it showed.
        if model_loaded:
            return check('edgetpu', True,
                         "pycoral absent, but the delegate loaded during the model check")
        return check('edgetpu', False,
                     "cannot verify: pycoral absent and the model failed to load",
                     "Install the Coral runtime:  "
                     "sudo apt-get install -y python3-pycoral libedgetpu1-std")


def check_sounds(config):
    sound_dir = Path(config.get('audio.sound_path', './sounds/'))
    wavs = sorted(sound_dir.glob('*.wav')) if sound_dir.exists() else []
    if not wavs:
        check('sounds', False, f"no .wav files in {sound_dir}",
              f"The deterrent would fire silently. Copy at least one WAV:",
              f"  cp /path/to/heron_distress.wav {sound_dir}",
              f"The repo ships a default: git checkout sounds/")
        return None
    total_kb = sum(w.stat().st_size for w in wavs) // 1024
    check('sounds', True,
          f"{len(wavs)} file(s), {total_kb} KB: {', '.join(w.name for w in wavs[:3])}"
          + (" ..." if len(wavs) > 3 else ""))
    return wavs


def check_audio(wavs, skip_audio):
    if not shutil.which('aplay'):
        return check('audio', False, "aplay not found",
                     "sudo apt-get install -y alsa-utils")

    try:
        listing = subprocess.run(['aplay', '-l'], capture_output=True, text=True, timeout=10)
    except (subprocess.TimeoutExpired, OSError) as e:
        return check('audio', False, f"aplay -l did not respond: {e}",
                     "A wedged ALSA device usually needs a reboot, or:",
                     "  sudo fuser -v /dev/snd/*")

    if listing.returncode != 0 or 'card' not in listing.stdout:
        return check('audio', False, "no ALSA playback device",
                     "Plug in the speaker, then check:  aplay -l",
                     "On a Pi, force the 3.5mm jack:     sudo raspi-config nonint do_audio 1")

    cards = [ln.split(':', 1)[1].strip() for ln in listing.stdout.splitlines()
             if ln.startswith('card')]
    if skip_audio or not wavs:
        return check('audio', True, f"device present ({cards[0] if cards else '?'}), "
                                    f"playback not tested (--skip-audio)")

    try:
        played = subprocess.run(['aplay', str(wavs[0])], capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        return check('audio', False, f"playing {wavs[0].name} hung for 30s",
                     "The WAV may be malformed, or the device is blocked:",
                     f"  file {wavs[0]}",
                     "  sudo fuser -v /dev/snd/*")

    if played.returncode != 0:
        return check('audio', False,
                     f"device present but playing {wavs[0].name} failed",
                     played.stderr.decode(errors='replace').strip()[:200],
                     "Test directly:  speaker-test -t wav -c 2")
    return check('audio', True,
                 f"played {wavs[0].name} on {cards[0] if cards else 'default'} "
                 f"— did you hear it?")


def _service_is_running():
    """True if the heron-deterrent service is active — it holds the camera."""
    try:
        return subprocess.run(['systemctl', 'is-active', '--quiet', 'heron-deterrent'],
                              timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def check_camera(config):
    try:
        import cv2
    except ImportError:
        return check('camera', False, "opencv not installed",
                     "pip install opencv-python-headless")

    # OpenCV writes V4L2 probe failures straight to stderr, which buries the
    # PASS/FAIL lines this tool exists to show.
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass

    device_id = config.get('system.camera_device_id', 0)
    cap = cv2.VideoCapture(device_id)
    try:
        if not cap.isOpened():
            # Only one process can hold a V4L2 device. If our own service has
            # it, the camera is fine and this is not a fault — saying "FAIL"
            # here would send you hunting a problem that does not exist.
            if _service_is_running():
                print(f"{DIM}[skip] camera       device {device_id} is held by the "
                      f"running heron-deterrent service{OFF}")
                print(f"       {DIM}->{OFF} That is expected. To test the device itself:")
                print(f"       {DIM}->{OFF}   sudo systemctl stop heron-deterrent "
                      f"&& ./preflight.py")
                print(f"       {DIM}->{OFF} Or check the live feed: "
                      f"curl -s localhost:5000/api/system-status")
                return True

            return check('camera', False, f"cannot open device {device_id}",
                         "List devices:      v4l2-ctl --list-devices",
                         "Or check nodes:     ls /dev/video*",
                         "Something else using it?  sudo fuser -v /dev/video*",
                         "Wrong number? edit config.yaml -> system.camera_device_id")

        ok, frame = cap.read()
        if not ok or frame is None:
            return check('camera', False, f"device {device_id} opened but returned no frame",
                         "Often a USB power problem — try a powered hub",
                         "Or a busy device:  sudo fuser -v /dev/video*")

        h, w = frame.shape[:2]
        want = (config.get('system.resolution.width', 640),
                config.get('system.resolution.height', 480))
        note = "" if (w, h) == want else f"  (config asks for {want[0]}x{want[1]})"
        return check('camera', True, f"device {device_id} returned a {w}x{h} frame{note}")
    finally:
        cap.release()


def check_video_encoder():
    """Can this board write clips a browser can actually play?

    Worth a dedicated check because the old failure was completely silent:
    cv2.VideoWriter(*'mp4v') reported success and produced a well-formed .mp4
    that no browser can decode. Nothing downstream noticed — the dashboard drew
    playback controls over a black frame. Only opening a clip in a browser, or
    checking the codec here, reveals it.
    """
    from storage.video_encoder import encoder_status

    ok, detail = encoder_status()
    if ok:
        return check('video', True, f"H.264 available — {detail}")
    return check('video', False, detail,
                 "Clips would be unplayable in the dashboard gallery.",
                 "On Mendel/Debian:  sudo apt-get install -y ffmpeg",
                 "Verify afterwards: ffmpeg -hide_banner -encoders | grep libx264")


def check_disk(config):
    media = Path(config.get('storage.media_path', './media/'))
    probe = media if media.exists() else Path('.')
    stat = os.statvfs(probe)
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
    total_gb = (stat.f_blocks * stat.f_frsize) / (1024 ** 3)
    used_pct = 100 * (1 - stat.f_bavail / stat.f_blocks) if stat.f_blocks else 0

    # Report the filesystem the media path is really on. On Mendel that is
    # /home, not /, and confusing the two hides how little headroom there is.
    mount = _mount_point(probe)
    floor_gb = max(MIN_FREE_GB, MIN_FREE_FRACTION * total_gb)
    detail = (f"{free_gb:.1f} GB free of {total_gb:.1f} GB on {mount} "
              f"({used_pct:.0f}% used)")

    if free_gb < floor_gb:
        return check('disk', False, detail,
                     f"Below the {floor_gb:.1f} GB floor for this filesystem. "
                     f"A full card kills the device.",
                     f"Largest media first:  du -sh {media}/* | sort -h | tail",
                     f"Then reduce storage.retention_days (currently "
                     f"{config.get('storage.retention_days')}) and restart.")
    return check('disk', True, detail)


def _mount_point(path):
    """Deepest mount point containing path, so we name the real filesystem."""
    p = Path(path).resolve()
    while not p.is_mount() and p != p.parent:
        p = p.parent
    return str(p)


def check_database(config):
    import sqlite3
    db_path = Path(config.get('storage.db_path', './data/heron.db'))
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return check('database', False, f"cannot create {db_path.parent}: {e}",
                     f"sudo chown -R $USER {db_path.parent}")

    try:
        conn = sqlite3.connect(db_path)
        conn.execute('CREATE TABLE IF NOT EXISTS _preflight (x INTEGER)')
        conn.execute('INSERT INTO _preflight VALUES (1)')
        conn.execute('DROP TABLE _preflight')
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        return check('database', False, f"{db_path} is not writable: {e}",
                     f"Check ownership:  ls -la {db_path.parent}",
                     f"Fix:              sudo chown -R $USER {db_path.parent}")

    size_kb = db_path.stat().st_size // 1024 if db_path.exists() else 0
    return check('database', True, f"{db_path} writable ({size_kb} KB)")


def check_twilio(config):
    if not config.get('alert.sms_enabled', False):
        print(f"{DIM}[skip] sms         alert.sms_enabled is false{OFF}")
        return True

    names = ('TWILIO_ACCOUNT_SID', 'TWILIO_AUTH_TOKEN',
             'TWILIO_FROM_NUMBER', 'TWILIO_TO_NUMBER')
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        return check('sms', False,
                     f"sms_enabled is true but {len(missing)} variable(s) unset: "
                     f"{', '.join(missing)}",
                     "Credentials belong in a root-owned file outside git:",
                     "  sudo install -m 600 -o root -g root /dev/null /etc/heron-deterrent.env",
                     "  sudo nano /etc/heron-deterrent.env     # TWILIO_ACCOUNT_SID=... etc",
                     "  sudo systemctl restart heron-deterrent",
                     "Running by hand? export them in this shell first.",
                     "Not using SMS? set alert.sms_enabled: false in config.yaml.")

    to = os.getenv('TWILIO_TO_NUMBER')
    masked = to[:3] + '*' * max(0, len(to) - 5) + to[-2:] if to and len(to) > 5 else '(set)'
    return check('sms', True, f"all four variables set, alerts go to {masked}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--skip-audio', action='store_true',
                        help='check the audio device but do not play a sound')
    args = parser.parse_args()

    print(f"\n{BOLD}Heron Deterrent — preflight{OFF}")
    print(f"{DIM}{socket.gethostname()}  {datetime.now():%Y-%m-%d %H:%M:%S}  "
          f"cwd {Path.cwd()}{OFF}\n")

    config = check_config()
    if config is None:
        print(f"\n{RED}Cannot continue without a valid config.{OFF}\n")
        return 1

    loader = check_model(config)
    check_edge_tpu(config, model_loaded=loader is not None)
    wavs = check_sounds(config)
    check_audio(wavs, args.skip_audio)
    check_camera(config)
    check_video_encoder()
    check_disk(config)
    check_database(config)
    check_twilio(config)

    passed = sum(1 for _, ok in results if ok)
    failed = [name for name, ok in results if not ok]

    print()
    if failed:
        print(f"{RED}{BOLD}{passed} passed, {len(failed)} FAILED{OFF} "
              f"({', '.join(failed)})")
        print(f"{RED}Do not start the service until the failures above are fixed.{OFF}\n")
        return 1

    print(f"{GREEN}{BOLD}All {passed} checks passed.{OFF}")
    if _service_is_running():
        print(f"{DIM}Already running.  Watch it:  "
              f"sudo journalctl -u heron-deterrent -f{OFF}\n")
    else:
        print(f"{DIM}Start it:  sudo systemctl start heron-deterrent{OFF}")
        print(f"{DIM}Watch it:  sudo journalctl -u heron-deterrent -f{OFF}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
