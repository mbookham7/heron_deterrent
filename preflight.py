#!/usr/bin/env python3
"""Pre-start checks for the Heron Deterrent System.

Run this on the board before starting the service. Every check prints PASS or
FAIL; every failure prints the command that fixes it.

    ./preflight.py               # all checks, plays a deterrent sound
    ./preflight.py --skip-audio  # same, but stay quiet

Exit status is 0 only if every check passed. [WARN] lines do not fail the
run, but each one names something worth fixing before the box is sealed.
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
YELLOW = '\033[33m' if _COLOUR else ''
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


def warn(name, detail, *hints):
    """Record and print a warning: worth acting on, not a reason to refuse to
    start. Hints are always printed."""
    print(f"{YELLOW}[WARN]{OFF} {name:<12} {detail}")
    for hint in hints:
        for line in str(hint).splitlines():
            print(f"       {DIM}->{OFF} {line}")
    warnings.append(name)
    return True


warnings = []

# Pi 4 firmware starts soft-throttling at 80 C and hard-throttles at 85 C.
# 70 C leaves headroom for a sealed box warming up through a summer afternoon.
CPU_TEMP_WARN_C = 70.0
CPU_TEMP_FAIL_C = 80.0

# vcgencmd get_throttled bit meanings (Raspberry Pi firmware).
THROTTLE_NOW = {0x1: 'under-voltage', 0x2: 'ARM frequency capped',
                0x4: 'throttled', 0x8: 'soft temperature limit'}
THROTTLE_SINCE_BOOT = {0x10000: 'under-voltage', 0x20000: 'ARM frequency capped',
                       0x40000: 'throttled', 0x80000: 'soft temperature limit'}

# Frames grabbed to judge the frame type. The first few after opening a USB
# camera are often mid auto-exposure, so take a handful and use the majority.
FRAME_TYPE_SAMPLE = 15


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
    check('config', True, f"ai.target={target}  classes={len(labels)}")
    return config


def check_modes(config):
    """Per-frame-type modes: coherent, and every named model file present."""
    from utils.modes import (ModeConfigError, load_frame_type_config, load_modes,
                             missing_model_files)
    try:
        modes = load_modes(config)
        load_frame_type_config(config)
    except ModeConfigError as e:
        lines = str(e).splitlines()
        check('modes', False, lines[0], *lines[1:])
        return None

    for mode in modes.values():
        print(f"       {DIM}{mode.describe()}{OFF}")

    missing = missing_model_files(modes, config.get('ai.target'))
    if missing:
        check('modes', False, "a mode names a model file that does not exist", *missing,
              "Copy the model into models/, or set that mode's model: null "
              "(motion-only) in config.yaml")
        return None

    armed = [m.name for m in modes.values() if m.deterrent]
    check('modes', True, f"{len(modes)} modes coherent; deterrent "
          + (f"ARMED in {', '.join(armed)}" if armed else "silent in every mode"))
    shadows = [m.name for m in modes.values() if m.shadow_model]
    if shadows:
        warn('modes', f"shadow inference on in {', '.join(shadows)}: one extra "
             f"inference per trigger (logged, never acted on)")
    return modes


def check_model(config, modes):
    """Load every model a mode names, once each. Motion-only modes need none."""
    from ai.model_loader import MODELS, ModelLoadError, ModelLoader
    target = config.get('ai.target')
    names = sorted({n for m in (modes or {}).values()
                    for n in (m.model, m.shadow_model) if n})
    if not names:
        print(f"{DIM}[skip] model        every mode is motion-only{OFF}")
        return True

    loaded = True
    for name in names:
        filename, needs_tpu, _source = MODELS[name][target]
        try:
            loader = ModelLoader(target, config.get('ai.num_threads', 4), model=name)
        except ModelLoadError as e:
            loaded = check('model', False, f"{filename} will not load on target '{target}'", e)
            continue
        except Exception as e:
            loaded = check('model', False, f"unexpected error loading {filename}: {e}")
            continue
        shape = 'x'.join(str(int(v)) for v in loader.input_details[0]['shape'])
        backend = 'Edge TPU' if needs_tpu else f"CPU/{config.get('ai.num_threads', 4)} threads"
        check('model', True, f"{name}: {filename} loaded on {backend}, input {shape}")
    return loaded


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
                print(f"       {DIM}->{OFF} Or check the live feed and frame type: "
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
        check('camera', True, f"device {device_id} returned a {w}x{h} frame{note}")

        frames = [frame]
        for _ in range(FRAME_TYPE_SAMPLE - 1):
            ok, f = cap.read()
            if ok and f is not None:
                frames.append(f)
        return check_frame_type(config, frames)
    finally:
        cap.release()


def check_frame_type(config, frames):
    """Report what kind of image the camera is producing right now, with the
    raw numbers behind it — those are what you tune frame_type: against.

    Run it once in daylight and once after dark: the two readings should land
    clearly either side of the thresholds. If a night reading sits in the dead
    band, the camera is probably tinting its IR frames rather than sending
    true monochrome; raise mono_max_chroma above the tint.
    """
    import statistics
    from capture.frame_classifier import AMBIGUOUS, classify_measured, measure
    from utils.modes import ModeConfigError, load_frame_type_config, load_modes
    try:
        cfg = load_frame_type_config(config)
        modes = load_modes(config)
    except ModeConfigError:
        print(f"{DIM}[skip] frame type   frame_type:/modes: invalid (see above){OFF}")
        return True

    readings = [measure(f, cfg.sample_step) for f in frames]
    types = [classify_measured(c, l, cfg) or AMBIGUOUS for c, l in readings]
    majority = max(set(types), key=types.count)
    chroma = statistics.median(c for c, _ in readings)
    luma = statistics.median(l for _, l in readings)
    tally = ', '.join(f"{t} {types.count(t)}" for t in sorted(set(types)))
    detail = (f"{majority} ({tally} of {len(types)} frames)  "
              f"chroma {chroma:.2f}  luma {luma:.1f}")
    thresholds = (f"Thresholds: IR below chroma {cfg.mono_max_chroma}, dead band to "
                  f"{cfg.ambiguous_chroma}; dark below luma {cfg.dark_max_luma}")

    if majority == AMBIGUOUS or types.count(majority) < len(types) * 0.8:
        return warn('frame type', detail,
                    "Frames are ambiguous or disagree: the system would stay in "
                    "establishing/switching, recording but never classifying.",
                    thresholds,
                    "If this is an IR scene, the camera may tint IR frames: raise "
                    "frame_type.mono_max_chroma above the chroma shown.")
    mode = modes.get(majority)
    return check('frame type', True,
                 detail + (f"  -> {mode.describe()}" if mode else ""))


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


def check_storage_caps(config):
    """The two clip caps must fit on the filesystem with room to spare.

    They are sized for the Pi's card. Left at those values on a Coral, whose
    /home is a 2 GB partition, the caps would never trigger and the card would
    fill — the cap that cannot be reached protects nothing.
    """
    media = Path(config.get('storage.media_path', './media/'))
    probe = media if media.exists() else Path('.')
    stat = os.statvfs(probe)
    total_gb = (stat.f_blocks * stat.f_frsize) / (1024 ** 3)
    media_gb = config.get('storage.max_media_gb', 1.0)
    motion_gb = config.get('storage.max_motion_gb', 1.0)
    caps = media_gb + motion_gb
    detail = (f"clip caps {media_gb:g} GB classified + {motion_gb:g} GB motion = "
              f"{caps:g} GB on a {total_gb:.1f} GB filesystem")
    if caps > 0.8 * total_gb:
        return check('storage', False, detail,
                     "The caps can never be reached before the disk fills.",
                     "Lower storage.max_media_gb and storage.max_motion_gb so together "
                     f"they stay under {0.6 * total_gb:.1f} GB (60% of this filesystem).")
    return check('storage', True, detail)


def check_labeled(config):
    """Per-domain labelled data, and anything stranded in the old flat layout."""
    media = Path(config.get('storage.media_path', './media/'))
    counts = []
    for ft in ('day', 'ir', 'dark'):
        d = media / 'labeled' / ft / 'images'
        counts.append(f"{ft} {len(list(d.glob('*.jpg'))) if d.exists() else 0}")
    legacy = [f for sub in ('images', 'labels')
              for f in ((media / 'labeled' / sub).glob('*'))
              if f.is_file()] if (media / 'labeled').exists() else []
    if legacy:
        return warn('labeled', f"images by domain: {', '.join(counts)}; "
                    f"{len(legacy)} file(s) in the pre-split labeled/images|labels",
                    "Those have no recorded domain. Sort them by eye into "
                    "labeled/day|ir|dark/{images,labels} — keep each .jpg with its .txt.",
                    "Nothing deletes them meanwhile, but no training run should read them.")
    return check('labeled', True, f"images by domain: {', '.join(counts)}")


def cpu_temp_verdict(temp_c, throttle_bits):
    """('pass'|'warn'|'fail', active conditions, conditions since boot).
    throttle_bits is vcgencmd get_throttled's value, or None if unavailable."""
    bits = throttle_bits or 0
    now = [name for bit, name in THROTTLE_NOW.items() if bits & bit]
    past = [name for bit, name in THROTTLE_SINCE_BOOT.items() if bits & bit]
    if temp_c >= CPU_TEMP_FAIL_C or now:
        return 'fail', now, past
    if temp_c >= CPU_TEMP_WARN_C or past:
        return 'warn', now, past
    return 'pass', now, past


def check_cpu_temp():
    """CPU temperature and throttling history.

    The Pi lives in a sealed IP box in a filter house. A board quietly
    throttling looks exactly like "inference got slower for no reason", so
    report it here rather than leave it to be inferred from timings.
    """
    zone = Path('/sys/class/thermal/thermal_zone0/temp')
    if not zone.exists():
        print(f"{DIM}[skip] cpu temp     no /sys thermal zone (not a Pi/Linux board){OFF}")
        return True
    try:
        temp_c = int(zone.read_text().strip()) / 1000.0
    except (OSError, ValueError) as e:
        return warn('cpu temp', f"cannot read {zone}: {e}")

    bits = None
    vcgencmd = shutil.which('vcgencmd')
    if vcgencmd:
        try:
            out = subprocess.run([vcgencmd, 'get_throttled'], capture_output=True,
                                 text=True, timeout=5).stdout.strip()
            bits = int(out.split('=')[1], 16)   # "throttled=0x50000"
        except (OSError, subprocess.TimeoutExpired, IndexError, ValueError):
            pass
    level, now, past = cpu_temp_verdict(temp_c, bits)

    detail = f"{temp_c:.1f} C" + ("" if vcgencmd else "  (no vcgencmd: throttle state unknown)")
    cooling = ("Improve cooling: heatsink or fan on the SoC, a vented or shaded "
               "enclosure, and keep the box out of direct sun.")
    if level == 'fail':
        return check('cpu temp', False,
                     detail + (f"  NOW: {', '.join(now)}" if now else ""),
                     f"At {CPU_TEMP_FAIL_C:.0f} C the Pi 4 throttles; inference "
                     f"slows with no error anywhere.",
                     "Under-voltage means the PSU or cable is too weak: use the "
                     "official 5.1 V / 3 A supply." if 'under-voltage' in now else cooling)
    if level == 'warn':
        return warn('cpu temp',
                    detail + (f"  since boot: {', '.join(past)}" if past else ""),
                    cooling if temp_c >= CPU_TEMP_WARN_C or 'soft temperature limit' in past
                    or 'throttled' in past else
                    "Under-voltage since boot: check the PSU and cable.",
                    "Watch it live:  watch -n 5 'vcgencmd measure_temp; vcgencmd get_throttled'")
    return check('cpu temp', True, detail + ("  never throttled since boot" if vcgencmd else ""))


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

    modes = check_modes(config)
    loaded = check_model(config, modes)
    check_edge_tpu(config, model_loaded=bool(loaded))
    wavs = check_sounds(config)
    check_audio(wavs, args.skip_audio)
    check_camera(config)
    check_video_encoder()
    check_disk(config)
    check_storage_caps(config)
    check_labeled(config)
    check_database(config)
    check_cpu_temp()
    check_twilio(config)

    passed = sum(1 for _, ok in results if ok)
    failed = [name for name, ok in results if not ok]

    print()
    if failed:
        print(f"{RED}{BOLD}{passed} passed, {len(failed)} FAILED{OFF} "
              f"({', '.join(failed)})")
        print(f"{RED}Do not start the service until the failures above are fixed.{OFF}\n")
        return 1

    print(f"{GREEN}{BOLD}All {passed} checks passed.{OFF}"
          + (f" {YELLOW}{len(warnings)} warning(s): {', '.join(warnings)}{OFF}" if warnings else ""))
    if _service_is_running():
        print(f"{DIM}Already running.  Watch it:  "
              f"sudo journalctl -u heron-deterrent -f{OFF}\n")
    else:
        print(f"{DIM}Start it:  sudo systemctl start heron-deterrent{OFF}")
        print(f"{DIM}Watch it:  sudo journalctl -u heron-deterrent -f{OFF}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
