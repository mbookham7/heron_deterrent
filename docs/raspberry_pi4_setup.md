# Raspberry Pi 4 Setup Guide

This guide covers setting up the Heron Deterrent System on a **Raspberry Pi 4**.

> The primary target platform is the **Coral Dev Board** (built-in Edge TPU, faster inference).
> For that setup, return to the [main README](../README.md).

---

## How Pi 4 Differs from Coral Dev Board

| | Coral Dev Board | Raspberry Pi 4 |
|---|---|---|
| Edge TPU | Built-in (M.2) | None built-in |
| AI inference | Edge TPU (~10 ms/frame) | CPU only (~200–400 ms/frame) |
| TFLite model | `_edgetpu.tflite` required | Standard `_int8.tflite` |
| OS | Mendel Linux | Raspberry Pi OS (Bookworm) |
| Audio mixer | Varies | `PCM` or `Headphone` (3.5mm jack) |
| Config flag | `use_edge_tpu: true` | `use_edge_tpu: false` |

Because the Pi 4 has no Edge TPU, inference runs on the 4-core ARM CPU. Expect roughly 2–5 fps of inference throughput on a typical YOLOv8n int8 model. The system compensates by running at 5 fps capture and analysing only key frames from each clip.

---

## Hardware

| Component | Notes |
|---|---|
| Raspberry Pi 4 (2 GB RAM minimum, 4 GB recommended) | |
| MicroSD card ≥ 16 GB (32 GB recommended) | Class 10 / A1 or better |
| USB webcam 720p or higher | Or Pi Camera Module v2/v3 via CSI |
| Powered USB speaker or 3.5mm amplified speaker | |
| USB-C power supply 5V/3A | Official Pi PSU recommended |
| Case with cooling | CPU runs warm during sustained inference |

---

## Step 1 — Flash Raspberry Pi OS

1. Download **Raspberry Pi Imager** from [raspberrypi.com/software](https://www.raspberrypi.com/software/)
2. Choose **Raspberry Pi OS Lite (64-bit)** — the headless image is sufficient
3. Click the gear icon in Imager to pre-configure:
   - Hostname (e.g. `heron-pi`)
   - SSH enabled
   - Wi-Fi SSID and password
   - Username and password (default: `pi`)
4. Flash to MicroSD, insert into Pi, power on
5. SSH in once the Pi has booted:

```bash
ssh pi@heron-pi.local
# or use the IP address if mDNS isn't working
```

---

## Step 2 — System Configuration

```bash
# Run raspi-config to enable camera and set audio output
sudo raspi-config
```

Inside raspi-config:
- **Interface Options → Camera** → Enable (if using Pi Camera Module via CSI)
- **System Options → Audio** → select **Headphones** (3.5mm jack) or **HDMI**

Or non-interactively:

```bash
# Enable CSI camera
sudo raspi-config nonint do_camera 0

# Set audio to 3.5mm jack (1 = headphones, 2 = HDMI, 0 = auto)
sudo raspi-config nonint do_audio 1
```

---

## Step 3 — Install System Dependencies

```bash
sudo apt-get update && sudo apt-get upgrade -y

sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-opencv \
    alsa-utils \
    v4l-utils \
    libportaudio2 \
    libatlas-base-dev \
    libjpeg-dev \
    libopenjp2-7 \
    sqlite3 \
    git
```

---

## Step 4 — Clone the Repository

```bash
cd ~
git clone https://github.com/mbookham7/heron_deterrent.git
cd heron_deterrent
```

---

## Step 5 — Create a Virtual Environment and Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Install tflite-runtime

The correct method depends on your Python version:

```bash
python3 --version
```

**Python 3.9** (Raspberry Pi OS Bullseye):

```bash
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl
```

**Python 3.10 or 3.11** (Raspberry Pi OS Bookworm — the current default):

```bash
pip install tflite-runtime
```

Verify it installed:

```bash
python3 -c "import tflite_runtime.interpreter; print('tflite-runtime OK')"
```

---

## Step 6 — Add Your Model

The Pi 4 uses the **standard int8 TFLite model**, not the Edge TPU compiled variant. If you exported using Ultralytics:

```bash
# On a machine with GPU:
yolo export model=best.pt format=tflite int8=True
# This produces best_int8.tflite — copy to the Pi
scp best_int8.tflite pi@heron-pi.local:~/heron_deterrent/models/
```

```bash
mkdir -p models
# File should now be at:  models/best_int8.tflite
```

---

## Step 7 — Add Deterrent Sounds

```bash
mkdir -p sounds
# Copy your .wav files
scp /path/to/sounds/*.wav pi@heron-pi.local:~/heron_deterrent/sounds/
```

---

## Step 8 — Configure for Raspberry Pi 4

Edit `config.yaml`. The critical Pi-specific settings are:

```yaml
ai:
  confidence_threshold: 0.6
  model_path: "./models/best_int8.tflite"   # Standard int8, NOT _edgetpu
  use_edge_tpu: false                        # No built-in TPU on Pi 4
  num_threads: 4                             # Use all 4 cores
  labels:
    0: "heron"
    1: "cat"
    2: "dog"
    3: "bird"
    4: "person"

system:
  running_on_raspberry_pi: true             # Disables Edge TPU automatically
  frame_rate: 5                             # Keep low — CPU inference is slower
  camera_device_id: 0
  resolution:
    width: 640
    height: 480
  active_hours:
    start: "06:00"
    end: "20:00"
```

**Twilio credentials** — use environment variables, not the config file:

```bash
export TWILIO_ACCOUNT_SID="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export TWILIO_AUTH_TOKEN="your_auth_token"
export TWILIO_FROM_NUMBER="+441234567890"
export TWILIO_TO_NUMBER="+441234567890"
```

---

## Step 9 — Test Camera and Audio

```bash
source venv/bin/activate

# Check camera is detected
v4l2-ctl --list-devices

# Capture a test frame
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imwrite('test_frame.jpg', frame)
cap.release()
print('OK' if ret else 'FAILED — check camera_device_id in config.yaml')
"

# Test speaker
aplay -l                        # list audio devices
speaker-test -t wav -c 2       # play test tone
aplay sounds/your_sound.wav    # test a deterrent file
```

If `amixer set Master` fails (common on Pi OS), the audio player automatically tries `PCM` and `Headphone` as fallbacks. You can confirm which control your device uses:

```bash
amixer           # lists all controls and their names
```

---

## Step 10 — Run Manually First

```bash
source venv/bin/activate
START_WEB_UI=true python main.py
```

Open the dashboard at `http://heron-pi.local:5000` (or use the Pi's IP address).

Watch the log for startup errors:

```bash
tail -f heron_deterrent.log
```

You should see:
```
INFO  Model loaded on CPU
INFO  YOLO Detector initialized:
INFO    - Input size: 640x640
INFO    - Quantized: True
INFO  Loaded N sound files
INFO  System started successfully
```

---

## Step 11 — Install as a Systemd Service

```bash
# Edit the service file to match Pi paths
nano heron-deterrent.service
```

Change `User` and `WorkingDirectory` to match your Pi setup:

```ini
[Service]
User=pi
WorkingDirectory=/home/pi/heron_deterrent
ExecStart=/home/pi/heron_deterrent/venv/bin/python main.py
Environment="START_WEB_UI=true"

# Uncomment and fill in Twilio credentials:
# Environment="TWILIO_ACCOUNT_SID=ACxxxx"
# Environment="TWILIO_AUTH_TOKEN=xxxx"
# Environment="TWILIO_FROM_NUMBER=+441234567890"
# Environment="TWILIO_TO_NUMBER=+441234567890"
```

Install and enable:

```bash
sudo cp heron-deterrent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable heron-deterrent
sudo systemctl start heron-deterrent

# Verify
sudo systemctl status heron-deterrent
sudo journalctl -u heron-deterrent -f
```

The service will now start automatically on every boot and restart itself if it crashes.

---

## Optional: Using a Coral USB Accelerator

If you have a [Coral USB Accelerator](https://coral.ai/products/accelerator/), you can plug it into a Pi 4 USB 3.0 port for Edge TPU speed. You'll need the Edge TPU compiled model.

```bash
# Install Edge TPU runtime for the USB accelerator
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y libedgetpu1-std python3-pycoral

# Compile your model for Edge TPU (do this on a desktop machine)
# edgetpu_compiler best_int8.tflite
# Copy best_int8_edgetpu.tflite to the Pi
```

Then update `config.yaml`:

```yaml
ai:
  model_path: "./models/best_int8_edgetpu.tflite"
  use_edge_tpu: true
```

---

## Performance Expectations

| Config | Inference speed | Notes |
|---|---|---|
| Pi 4 CPU, 4 threads, YOLOv8n int8 | ~200–400 ms | Suitable for 5 fps capture |
| Pi 4 + Coral USB Accelerator | ~10–30 ms | Near Coral Dev Board performance |
| Coral Dev Board (built-in TPU) | ~5–10 ms | Fastest — recommended primary target |

At 5 fps capture with a 5-second clip and 5 key frames analysed, each detection event takes around 1–2 seconds to process on CPU — well within the time a heron would spend at a pond.

---

## Troubleshooting

### `tflite_runtime` ImportError

```bash
python3 -c "import tflite_runtime.interpreter"
# If this fails:
python3 --version   # check exact version
pip install tflite-runtime   # for Python 3.10+
# or use the Google wheel URL above for Python 3.9
```

### Slow inference / high CPU temperature

- Reduce `frame_rate` to `3` in `config.yaml`
- Add a heatsink and fan — sustained inference generates significant heat
- Consider a Coral USB Accelerator

### Camera not found

```bash
# USB webcam
v4l2-ctl --list-devices
ls /dev/video*

# Pi Camera Module (CSI)
vcgencmd get_camera
# Should show: supported=1 detected=1
# If detected=0, re-enable in raspi-config and reboot
```

### Audio: `amixer set Master` fails

```bash
amixer    # note the control names listed
# Common Pi controls: PCM, Headphone, Master
# The audio player tries all three automatically
```

### Service won't start

```bash
sudo journalctl -u heron-deterrent -n 50
# Look for: model not found, camera not opened, permission denied
```

Common causes:
- `WorkingDirectory` in the service file doesn't match where you cloned the repo
- Virtual environment path is wrong in `ExecStart`
- Model file missing from `models/`
- No WAV files in `sounds/`

---

## Back to Main README

[README.md](../README.md) — Coral Dev Board setup (primary platform)
