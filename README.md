# Heron Deterrent System

An edge AI system that watches a live camera feed, detects herons, plays a deterrent sound to scare them away, and sends SMS alerts — all running locally on a Coral Dev Board.

> **Running on a Raspberry Pi 4 instead?** See the [Raspberry Pi 4 Setup Guide](docs/raspberry_pi4_setup.md).

---

## How It Works

```
Camera → Motion Detection → AI Inference (Edge TPU) → Deterrent Sound
                                      ↓
                              SQLite Database
                                      ↓
                         Web Dashboard  +  SMS Alert
```

1. The camera runs continuously at 5 fps
2. Background subtraction filters frames for motion
3. When motion is detected and the cooldown has elapsed, a 5-second clip is captured
4. The clip's key frames are run through a YOLOv8 TFLite model on the Edge TPU
5. If a heron is detected above the confidence threshold, the deterrent sound plays and an alert is created
6. The web dashboard shows the live stream, detection history, and gallery of heron clips

---

## Features

- Real-time motion detection with configurable sensitivity
- YOLOv8 TFLite inference accelerated by the Coral Edge TPU
- Random deterrent WAV file playback via ALSA
- Web dashboard: live stream, stats, alert feed, heron gallery
- SMS notifications via Twilio (throttled to avoid spam)
- Optional MQTT publishing for IoT integrations
- Manual labelling tool to build a training dataset
- Automatic cleanup of old videos and database records
- Auto-start on boot via systemd

---

## Hardware

| Component | Recommended |
|---|---|
| Board | Google Coral Dev Board (built-in Edge TPU) |
| Camera | USB webcam 720p or higher |
| Speaker | USB or 3.5mm powered speaker |
| Storage | MicroSD ≥ 16 GB (32 GB recommended) |
| Power | USB-C 5V/3A supply |
| Network | Ethernet or Wi-Fi |

---

## Step 1 — Flash the Coral Dev Board

If your board is freshly unboxed or needs a clean OS, flash Mendel Linux first.

**Full flashing instructions:** [docs/coral_dev_board_flash_guide.md](docs/coral_dev_board_flash_guide.md)

In brief:

```bash
# On your host machine — download the Mendel Linux flash package
mkdir ~/coral-flash && cd ~/coral-flash
# Visit https://coral.ai/software/#mendel-linux and download the latest Enterprise Eagle zip

# Short the two boot pins on the board, connect USB-C, apply power, remove short
fastboot devices          # should list your board

# Extract the downloaded zip, then:
chmod +x flash.sh
sudo ./flash.sh           # takes 5–10 minutes
```

After flashing, connect with MDT:

```bash
pip3 install mendel-development-tool
mdt devices               # wait ~3 min for first boot
mdt shell                 # opens a shell on the board
```

---

## Step 2 — First-Boot Board Setup

Inside the board shell:

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Set your timezone
sudo timedatectl set-timezone Europe/London

# Connect to Wi-Fi (or use Ethernet — skip if already connected)
nmcli device wifi connect "YOUR_SSID" password "YOUR_PASSWORD"

# Verify Edge TPU is present
lspci -nn | grep 089a
# Expected: 01:00.0 System peripheral: Global Unichip Corp. Coral Edge TPU [1ac1:089a]
```

---

## Step 3 — Install System Dependencies

```bash
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

# Install Edge TPU runtime and PyCoral
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y libedgetpu1-std python3-pycoral
```

---

## Step 4 — Clone the Repository

```bash
cd ~
git clone https://github.com/mbookham7/heron_deterrent.git
cd heron_deterrent
```

---

## Step 5 — Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Install tflite-runtime (PyCoral provides this on Mendel Linux)
# If not already available via python3-pycoral:
pip install tflite-runtime
```

---

## Step 6 — Add Your Model

Place your trained YOLOv8 TFLite model in the `models/` directory.
The Coral requires the **Edge TPU compiled** variant (the `_edgetpu.tflite` file):

```bash
mkdir -p models
cp /path/to/best_int8_edgetpu.tflite models/best_int8_edgetpu.tflite
```

Update `config.yaml` to point at it:

```yaml
ai:
  model_path: "./models/best_int8_edgetpu.tflite"
  use_edge_tpu: true
```

> **Don't have a model yet?** See [Training Your Model](#training-your-model) below.

---

## Step 7 — Add Deterrent Sounds

Place `.wav` audio files in the `sounds/` directory. The system picks one at random each time a heron is detected.

```bash
mkdir -p sounds
cp /path/to/your/*.wav sounds/
```

Any WAV file works — predator calls, distress sounds, loud bangs, etc.

---

## Step 8 — Configure the System

Edit `config.yaml`. Key settings:

```yaml
motion:
  sensitivity: 5000          # Lower = more sensitive. Start here and tune.
  cooldown_seconds: 10       # Minimum gap between triggers

ai:
  confidence_threshold: 0.6  # Raise this if you get false positives
  model_path: "./models/best_int8_edgetpu.tflite"
  use_edge_tpu: true
  labels:
    0: "heron"               # Must match your model's class order
    1: "cat"
    2: "dog"
    3: "bird"
    4: "person"

system:
  active_hours:
    start: "06:00"           # Only monitor between these hours
    end: "20:00"
  camera_device_id: 0        # Run `v4l2-ctl --list-devices` to find yours
  frame_rate: 5
  resolution:
    width: 640
    height: 480
  running_on_raspberry_pi: false

storage:
  retention_days: 365
  media_path: "./media/"
  db_path: "./data/heron.db"
```

**Do not store Twilio credentials in `config.yaml`** — use environment variables instead:

```bash
export TWILIO_ACCOUNT_SID="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export TWILIO_AUTH_TOKEN="your_auth_token"
export TWILIO_FROM_NUMBER="+441234567890"
export TWILIO_TO_NUMBER="+441234567890"
```

Then enable SMS in `config.yaml`:

```yaml
alert:
  sms_enabled: true
  sms_throttle_minutes: 15
```

---

## Step 9 — Test the Camera and Audio

```bash
# List connected cameras
v4l2-ctl --list-devices

# Preview a frame (saves to disk)
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imwrite('test_frame.jpg', frame)
cap.release()
print('Frame saved' if ret else 'FAILED — check camera_device_id')
"

# List audio devices
aplay -l

# Test speaker output
speaker-test -t wav -c 2

# Test a WAV file directly
aplay sounds/your_sound.wav
```

---

## Step 10 — Run Manually (Test First)

```bash
source venv/bin/activate
START_WEB_UI=true python main.py
```

Open the web dashboard in your browser at `http://<board-ip>:5000`

Check the logs for any errors:

```
tail -f heron_deterrent.log
```

---

## Step 11 — Run as a Service (Auto-start on Boot)

Once you're happy the system works, install it as a systemd service so it starts automatically and restarts if it crashes.

```bash
# Edit the service file and verify the paths match your setup
nano heron-deterrent.service

# The key lines to check:
#   User=mendel                             (Mendel Linux default user)
#   WorkingDirectory=/home/mendel/heron_deterrent
#   ExecStart=/home/mendel/heron_deterrent/venv/bin/python main.py

# Install and enable
sudo cp heron-deterrent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable heron-deterrent
sudo systemctl start heron-deterrent

# Check it started correctly
sudo systemctl status heron-deterrent

# Follow logs in real time
sudo journalctl -u heron-deterrent -f
```

To pass environment variables (e.g. Twilio) to the service, uncomment and fill in the `Environment=` lines in `heron-deterrent.service`:

```ini
Environment="TWILIO_ACCOUNT_SID=ACxxxx"
Environment="TWILIO_AUTH_TOKEN=xxxx"
Environment="TWILIO_FROM_NUMBER=+441234567890"
Environment="TWILIO_TO_NUMBER=+441234567890"
```

Then reload:

```bash
sudo systemctl daemon-reload
sudo systemctl restart heron-deterrent
```

---

## Web Dashboard

Access at `http://<board-ip>:5000`

| Section | Description |
|---|---|
| Stats strip | Herons today, total detections, last detection time, alert count |
| Live Feed | MJPEG stream from the camera with system status overlay |
| Recent Alerts | Feed of CRITICAL / WARNING alerts with timestamps |
| Heron Detections | Detection history with confidence bars |
| Unclassified | Objects the model returned "unknown" for — label them to build a dataset |
| Heron Gallery | Video clips of confirmed heron detections |
| Test Upload | Upload a video file to run inference against — useful for tuning the model |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/live-stream` | MJPEG video stream |
| GET | `/api/system-status` | System health JSON |
| GET | `/api/alerts` | Recent alerts (param: `limit`) |
| GET | `/api/detections` | Detection history (params: `label`, `limit`) |
| GET | `/api/heron-detections` | Heron-only detections (param: `limit`) |
| GET | `/api/unclassified` | Unlabelled detections |
| POST | `/api/label` | Submit a manual label `{detection_id, label, bboxes}` |
| POST | `/api/trigger-deterrent` | Manually fire the deterrent sound |
| POST | `/api/upload-test-video` | Run inference on an uploaded video file |
| GET | `/api/media/<filename>` | Serve a stored video clip |

---

## Configuration Reference

| Key | Default | Description |
|---|---|---|
| `motion.sensitivity` | `5000` | Minimum contour area to trigger motion. Lower = more sensitive. |
| `motion.cooldown_seconds` | `10` | Seconds between detection events |
| `ai.confidence_threshold` | `0.6` | Minimum confidence score to accept a detection |
| `ai.use_edge_tpu` | `true` | Use the Coral Edge TPU (set `false` for Raspberry Pi CPU) |
| `ai.num_threads` | `4` | CPU threads for TFLite (ignored when Edge TPU is used) |
| `ai.labels` | see config | Map of class index → label name — must match your model |
| `alert.sms_enabled` | `false` | Enable Twilio SMS alerts |
| `alert.sms_throttle_minutes` | `15` | Minimum gap between SMS messages |
| `system.active_hours.start` | `06:00` | Start of monitoring window |
| `system.active_hours.end` | `20:00` | End of monitoring window |
| `system.camera_device_id` | `0` | V4L2 device number |
| `system.frame_rate` | `5` | Camera capture rate (fps) |
| `storage.retention_days` | `365` | Days before old records and videos are deleted |

---

## Training Your Model

The system includes a manual labelling workflow to build a training dataset from footage captured in the field.

### 1. Collect data

Run the system for a few days. Any detection saved as "unknown" is available in the **Unclassified** panel on the dashboard. Label each one using the form — this writes YOLO-format annotation files to `media/labeled/`.

### 2. Train

```bash
# On a machine with a GPU
pip install ultralytics

# Train YOLOv8 nano (fast) — adjust data path and epochs as needed
yolo detect train \
    data=dataset.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640
```

### 3. Export for Edge TPU (Coral Dev Board)

```bash
# Export to int8 TFLite first
yolo export \
    model=runs/detect/train/weights/best.pt \
    format=tflite \
    int8=True

# Then compile for Edge TPU using the Coral compiler
# Install: https://coral.ai/docs/edgetpu/compiler/
edgetpu_compiler best_int8.tflite

# This produces best_int8_edgetpu.tflite — copy to the board
scp best_int8_edgetpu.tflite mendel@<board-ip>:~/heron_deterrent/models/
```

### 4. Export for CPU (Raspberry Pi 4)

If you're deploying to a Pi 4 instead, skip the `edgetpu_compiler` step and use `best_int8.tflite` directly.

---

## Project Structure

```
heron_deterrent/
├── main.py                       # System orchestrator
├── config.yaml                   # All configuration
├── requirements.txt              # Python dependencies
├── heron-deterrent.service       # Systemd service unit
├── setup.sh                      # Automated setup script
│
├── ai/
│   ├── detector.py               # YOLOv8/v5 TFLite inference + NMS
│   └── model_loader.py           # TFLite + Edge TPU model loader
│
├── capture/
│   ├── camera.py                 # OpenCV camera wrapper
│   └── motion_detector.py        # MOG2 background subtraction
│
├── deterrent/
│   └── audio_player.py           # ALSA audio playback
│
├── storage/
│   ├── database.py               # SQLite (thread-safe)
│   └── media_store.py            # Video/image file management
│
├── alerts/
│   ├── alert_manager.py          # SMS + MQTT orchestration
│   └── mqtt_client.py            # Paho MQTT client with reconnection
│
├── ui/
│   ├── app.py                    # Flask REST API
│   └── templates/
│       └── index.html            # Web dashboard
│
├── utils/
│   └── config_loader.py          # YAML config loader
│
├── models/                       # Place your .tflite model here
├── sounds/                       # Place your .wav deterrent files here
├── media/                        # Auto-created: heron/, other/, unknown/, labeled/
├── data/                         # Auto-created: heron.db
│
└── docs/
    ├── coral_dev_board_flash_guide.md
    └── raspberry_pi4_setup.md
```

---

## Troubleshooting

### Camera not detected

```bash
# List all V4L2 devices
v4l2-ctl --list-devices

# Test opening device 0
python3 -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened()); cap.release()"

# If the wrong device is used, update camera_device_id in config.yaml
```

### No sound / audio silent

```bash
# List audio output devices
aplay -l

# Test speaker
speaker-test -t wav -c 2

# Check which mixer control applies (Coral may use a different name)
amixer
# Note the control name shown, update audio player if needed
```

### Edge TPU not detected

```bash
# Check PCIe
lspci -nn | grep 089a

# Check kernel driver
dmesg | grep apex

# Reinstall runtime
sudo apt-get install --reinstall libedgetpu1-std

# Verify Python can load it
python3 -c "from pycoral.utils.edgetpu import list_edge_tpus; print(list_edge_tpus())"
```

### Model outputs wrong classes

Edit the `ai.labels` section in `config.yaml` to match the class order your model was trained with. The indices must map exactly to what's in your `data.yaml` training file.

### System starts but no detections

1. Check `active_hours` — is the current time within the window?
2. Lower `motion.sensitivity` (e.g. try `2000`)
3. Lower `ai.confidence_threshold` (e.g. try `0.4`)
4. Watch the logs: `sudo journalctl -u heron-deterrent -f`
5. Upload a test video via the dashboard to confirm inference is working

### Disk space warning

```bash
df -h
# The system warns when free space drops below 1 GB

# Manually trigger a cleanup
sqlite3 data/heron.db "DELETE FROM detections WHERE timestamp < datetime('now', '-30 days');"

# Or reduce retention_days in config.yaml and restart
```

---

## Raspberry Pi 4

If you need to run this on a Raspberry Pi 4 (no built-in Edge TPU), see the dedicated setup guide:

**[docs/raspberry_pi4_setup.md](docs/raspberry_pi4_setup.md)**

Key differences are: `use_edge_tpu: false`, `running_on_raspberry_pi: true`, a different tflite-runtime install method, and a slower inference rate due to CPU-only execution.

---

## License

Open source — free for personal and commercial use.

## Contributing

Pull requests and issues are welcome.
