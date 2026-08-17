# Heron Deterrent System

An edge AI system that watches a live camera feed, detects herons with a YOLOv8
TFLite model, plays a deterrent sound to scare them away, and raises SMS and
MQTT alerts — all running locally on a Coral Dev Board or a Raspberry Pi 4.

**Deploying at the pond? Go straight to [DEPLOY.md](DEPLOY.md).** This file is
the reference; that one is the checklist.

---

## How It Works

```
Camera (5 fps) → Motion detection → Inference on the trigger frame
                                            │
                        heron ──────────────┤────────────── not heron
                          │                                     │
                  Deterrent sound FIRST                  Capture clip,
                          │                              re-check 5 key frames
                  Capture clip, save                            │
                          │                                     │
                  SQLite ── Dashboard ── SMS / MQTT      "Unclassified" panel
```

The ordering matters. Inference runs on the frame that triggered the motion, so
the speaker fires about **1 second** after the bird appears. An earlier version
captured a five-second clip first and took **9.6 seconds** — by which time the
heron had usually left.

If the trigger frame is inconclusive, the system falls back to capturing a clip
and scoring five key frames, so a bird caught mid-wingbeat is not written off.

---

## Measured performance

All figures measured on a Coral Dev Board (Mendel 5, quad Cortex-A53), CPU
inference, 640×640 model:

| Metric | Value |
|---|---|
| Inference | 920 ms/frame |
| Motion → deterrent sound | **1.02 s** |
| Recall @ `confidence_threshold: 0.75` | 96% over 80 dataset images |
| Non-heron frames | mostly 0.15–0.33, occasional spikes to ~0.70 |
| Saved clip size | 117 KB (5 s, 640×480) |
| Resident memory | ~110 MB |

---

## Edge TPU status — read this before buying hardware for it

**The Edge TPU is not currently usable, and it is the model's fault, not the
board's.** `ai.target: edgetpu` loads and runs at 87 ms/frame, but detects
nothing: full-integer quantisation destroyed the model's classification head, so
every one of the 8,400 anchors scores about `1e-5`. The bounding-box outputs also
come back in pixel units rather than normalised, unlike the CPU model.

The hardware itself is healthy — an older compiled model runs on the same TPU at
3 ms/frame.

Fixing it requires re-quantising with better calibration and recompiling with
`edgetpu_compiler`, which only runs on x86-64 Linux. Until someone does that,
**use `ai.target: cpu` on both boards.** A Coral runs CPU inference perfectly
well; it is simply not using its accelerator.

---

## Hardware

| Component | Notes |
|---|---|
| Board | Coral Dev Board or Raspberry Pi 4 (2 GB+) |
| Camera | USB webcam, 640×480 is plenty |
| Speaker | Powered USB or 3.5 mm speaker — the deterrent is the whole point |
| Storage | MicroSD ≥ 16 GB |
| Power | The board's rated supply; brownouts show up as camera dropouts |
| Network | Wi-Fi or Ethernet. Optional — detection works offline |

---

## Configuration

`config.yaml` in full. **`ai.target` is the only hardware switch**, and it names
the inference *backend*, not the board:

```yaml
ai:
  target: cpu                  # "cpu" or "edgetpu" — see Edge TPU status above
  confidence_threshold: 0.75
  num_threads: 4               # CPU inference only; ignored when target is edgetpu
  labels:
    0: "heron"                 # single-class model; more entries fail at startup
```

| `ai.target` | Model file | Backend |
|---|---|---|
| `cpu` | `models/heron_cpu_int8.tflite` | `tflite-runtime`, either board |
| `edgetpu` | `models/heron_edgetpu.tflite` | Edge TPU delegate, Coral only |

There is deliberately **no `model_path` key**. The filename is derived from the
target, so config cannot point at a file that does not exist. To trial a
different model, rename a file in `models/`.

### The model is single-class

Both training configs are `nc: 1, names: [heron]`, and every exported model
emits `[1, 5, N]` — that is 4 bbox coordinates plus **one** class score. An
earlier config declared five classes (`heron, cat, dog, bird, person`), which
was silently unreachable: `argmax` over a length-1 array is always 0.

`ai.labels` must therefore have exactly one entry. A mismatch raises
`ClassCountMismatch` at startup rather than mislabelling detections.

The dashboard's manual-labelling form is a **separate vocabulary** — it lets you
tag an "unknown" clip as cat, dog, bird or person to triage footage. Those are
human triage labels, not model classes. Only heron-tagged images get a YOLO
annotation; anything else is written as a background example (empty label file).

### Full configuration reference

| Key | Default | Description |
|---|---|---|
| `motion.sensitivity` | `5000` | Minimum contour area to trigger. Lower = more sensitive |
| `motion.cooldown_seconds` | `10` | Minimum gap between detection events |
| `ai.target` | `cpu` | `cpu` or `edgetpu` — the only hardware switch |
| `ai.confidence_threshold` | `0.75` | Measured: 96% recall, clear of non-heron spikes |
| `ai.num_threads` | `4` | CPU threads; ignored on `edgetpu` |
| `ai.labels` | `{0: heron}` | Must match the model's class count exactly |
| `alert.sms_enabled` | `false` | Needs `/etc/heron-deterrent.env` (see below) |
| `alert.sms_throttle_minutes` | `15` | Minimum gap between SMS messages |
| `audio.volume` | `80` | Percent, applied via `amixer` |
| `audio.sound_path` | `./sounds/` | Startup **fails** if this has no `.wav` files |
| `storage.retention_days` | `365` | Age-based deletion |
| `storage.max_media_gb` | `1.0` | **Hard size cap.** Oldest clips deleted first |
| `storage.media_path` | `./media/` | Clips, grouped by label |
| `storage.db_path` | `./data/heron.db` | SQLite database |
| `mqtt.enabled` | `false` | Optional MQTT publishing |
| `mqtt.broker` | `localhost` | Broker hostname. Connects asynchronously, retries forever |
| `mqtt.port` | `1883` | Broker port |
| `system.active_hours.start/end` | `06:00`/`20:00` | Handles windows spanning midnight |
| `system.camera_device_id` | `0` | V4L2 device number |
| `system.frame_rate` | `5` | Capture rate, and the rate clips are written at |

### Why there is a size cap as well as retention

Retention by age cannot protect a small card. Clips are 117 KB, so at 100 events
per day a 2 GB partition fills in about four months — long before a 365-day
cutoff deletes anything. A windy pond with a 10-second cooldown can generate far
more than 100 events a day.

`storage.max_media_gb` is the backstop that actually bounds growth: oldest files
go first, checked every five minutes, covering `labeled/` as well as the clip
directories.

> On the Coral, `/home` is a **separate 2 GB partition** and the repo lives
> there. `df -h /` showing plenty of space is misleading — check `df -h /home`.

### Secrets

Twilio credentials are read **only** from environment variables. There are no
`twilio_*` keys in `config.yaml`, so they cannot be committed by accident.

Put them in a root-owned file outside the repo, which the service unit reads:

```bash
sudo install -m 600 -o root -g root /dev/null /etc/heron-deterrent.env
sudo tee /etc/heron-deterrent.env >/dev/null <<'EOF'
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_FROM_NUMBER=+441234567890
TWILIO_TO_NUMBER=+441234567890
EOF
sudo systemctl restart heron-deterrent
```

Then set `alert.sms_enabled: true`. Do **not** put credentials in
`Environment=` lines in `heron-deterrent.service` — that file is tracked in git.

If SMS is enabled but the variables are missing, the system logs a warning and
runs without SMS. Losing alerts is bad; losing the deterrent because alerting is
misconfigured would be worse.

---

## preflight.py

One command that checks everything before you start the service. Run it on the
board, from the repo root:

```bash
./preflight.py                # all checks, plays a deterrent sound
./preflight.py --skip-audio   # same, but stay quiet
```

```
[PASS] config       ai.target=cpu  confidence=0.75  classes=1
[PASS] model        heron_cpu_int8.tflite loaded on CPU/4 threads, input 1x640x640x3
[skip] edgetpu      ai.target is not 'edgetpu'
[PASS] sounds       1 file(s), 172 KB: startle_default.wav
[PASS] audio        played startle_default.wav on edgetpu-audio-card — did you hear it?
[PASS] camera       device 0 returned a 640x480 frame
[PASS] disk         1.5 GB free of 1.9 GB on /home (23% used)
[PASS] database     data/heron.db writable (24 KB)
[skip] sms          alert.sms_enabled is false
All 7 checks passed.
```

Every failure prints the command that fixes it. Exit status is 0 only when
everything passes, so it can gate a deploy script.

Note that `preflight.py` run by hand will **not** see credentials from
`/etc/heron-deterrent.env` — only the service reads that. `export` them in your
shell to check the SMS leg manually.

---

## Deterrent sounds

`sounds/` must contain at least one `.wav` file or **the system refuses to
start**. A silent deterrent is the one failure mode that defeats the whole build
while looking perfectly healthy, so it is a hard error rather than a warning.

The repo ships `sounds/startle_default.wav` as a floor — three sharp broadband
bursts with a 2 ms attack plus a descending screech, generated with the Python
standard library. A sudden broadband transient is what actually startles birds.
Replace or supplement it with real heron distress calls or an air horn; one file
is picked at random per trigger.

---

## Field robustness

Things that were specifically hardened for running unattended outdoors:

- **Camera dropouts.** Health is judged on consecutive read failures, not on
  `isOpened()`, which commonly stays `True` through a USB brownout while every
  read fails. Reconnection backs off from 1 s to 60 s.
- **Thread safety.** The TFLite interpreter and the camera are each behind a
  lock. Both are reachable from the detection threads and the dashboard at once,
  and concurrent access to either is undefined behaviour.
- **Memory.** Motion events are dropped, not queued, when both detection slots
  are busy. Each queued event holds ~22 MB of frames.
- **Network.** Twilio calls time out after 10 seconds. MQTT connects
  asynchronously and retries indefinitely with backoff, so an unreachable broker
  never blocks startup or the detection loop.
- **Logs.** `heron_deterrent.log` rotates at 10 MB × 3 — a 40 MB hard ceiling.
- **Restart.** systemd restarts on failure after 10 s, giving up after 5 failures
  in 2 minutes so a config error is visible in `systemctl status` rather than
  looping forever in the journal.

---

## Web Dashboard

At `http://<board-ip>:5000`.

| Section | Description |
|---|---|
| Stats strip | Herons today, total detections, last detection, alert count |
| Live Feed | MJPEG stream, throttled to 2 fps so it cannot starve detection |
| Recent Alerts | CRITICAL and WARNING alerts with timestamps |
| Heron Detections | History with confidence bars |
| Unclassified | Clips the model returned "unknown" for — label them to build a dataset |
| Heron Gallery | Confirmed heron clips |
| Test Upload | Upload a video to run inference against — the quickest end-to-end proof |

> Every inconclusive motion event creates a WARNING "unknown" alert and saves a
> clip. That is deliberate — it feeds the Unclassified panel — but at a busy pond
> the alert feed gets noisy. Raise `motion.sensitivity` if so.

The Flask server is the development server. That is fine on a private LAN; do
not expose it to the internet.

### API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/live-stream` | MJPEG video stream (2 fps) |
| GET | `/api/system-status` | System health JSON |
| GET | `/api/alerts` | Recent alerts (param: `limit`) |
| GET | `/api/detections` | Detection history (params: `label`, `limit`) |
| GET | `/api/heron-detections` | Heron-only detections (param: `limit`) |
| GET | `/api/unclassified` | Unlabelled detections |
| POST | `/api/label` | Submit a manual label `{detection_id, label, bboxes}` |
| POST | `/api/trigger-deterrent` | Manually fire the deterrent sound |
| POST | `/api/upload-test-video` | Run inference on an uploaded video |
| GET | `/api/media/<filename>` | Serve a stored clip |

> `POST /api/config` exists but is **not safe to call**: it rewrites
> `config.yaml` from the posted JSON, and since the GET strips the `alert`
> section, a round trip deletes it along with every comment. The dashboard never
> calls it. Edit `config.yaml` directly instead.

---

## Installation

Full step-by-step for both boards is in **[DEPLOY.md](DEPLOY.md)**. The short
version:

```bash
git clone https://github.com/mbookham7/heron_deterrent.git
cd heron_deterrent
./setup.sh                    # system packages, venv, default config

# Models are gitignored — copy them in by hand
cp /path/to/heron_cpu_int8.tflite models/

./preflight.py                # must pass before you start
START_WEB_UI=true python main.py
```

### Python environment notes

**Coral / Mendel 5 (Eagle)** is Debian 10 with **Python 3.7**, OpenCV 4.5.4 and
numpy 1.19. `requirements.txt` floors are set low deliberately — demanding newer
versions triggers long source builds on the board for no functional gain.

The venv **must** be created with `--system-site-packages` on Mendel:

```bash
python3 -m venv --system-site-packages venv
```

Mendel's `python3-tflite-runtime` (2.5.0.post1) is version-matched to
`libedgetpu 16.0`. **Do not `pip install tflite-runtime` on Mendel** — the pip
build targets a newer TFLite and segfaults when the Edge TPU delegate constructs
an Interpreter. `setup.sh` handles this automatically.

**Raspberry Pi 4** uses a plain venv plus `pip install tflite-runtime`.

`paho-mqtt` is pinned below 2.0: version 2.x requires a `CallbackAPIVersion`
argument that `alerts/mqtt_client.py` does not pass, and the resulting
`ValueError` gets swallowed, silently disabling MQTT.

---

## Project Structure

```
heron_deterrent/
├── main.py                       # Orchestrator: pipeline, health, cleanup
├── preflight.py                  # Pre-start checks — run this at the pond
├── config.yaml                   # All configuration
├── requirements.txt              # Python dependencies
├── heron-deterrent.service       # Systemd unit
├── setup.sh                      # Automated setup
│
├── ai/
│   ├── detector.py               # YOLOv8 TFLite inference + NMS
│   └── model_loader.py           # Target → model file → backend
│
├── capture/
│   ├── camera.py                 # Locked, self-healing camera wrapper
│   └── motion_detector.py        # MOG2 background subtraction
│
├── deterrent/
│   └── audio_player.py           # ALSA playback; fails loudly if silent
│
├── storage/
│   ├── database.py               # SQLite (thread-safe)
│   └── media_store.py            # Clips, retention, size cap
│
├── alerts/
│   ├── alert_manager.py          # SMS + MQTT orchestration
│   └── mqtt_client.py            # Paho client with indefinite backoff
│
├── ui/
│   ├── app.py                    # Flask REST API
│   └── templates/index.html      # Dashboard
│
├── utils/config_loader.py        # YAML config loader
│
├── models/                       # *.tflite — gitignored, copy in by hand
├── sounds/                       # *.wav — at least one required
├── media/                        # Auto-created: heron/, other/, unknown/, labeled/
├── data/                         # Auto-created: heron.db
└── docs/
    ├── coral_dev_board_flash_guide.md
    └── raspberry_pi4_setup.md
```

---

## Training and exporting a model

The dataset and training pipeline live in sibling repositories
(`heron-data`, `heron-detector`). Current weights:
`heron-detector/runs/detect/heron_v8_model-4/weights/best.pt` (May 2026).

### Export for CPU (both boards)

```bash
yolo export model=best.pt format=tflite int8=True imgsz=640 data=your_data.yaml
# produces best_saved_model/best_int8.tflite
cp best_saved_model/best_int8.tflite models/heron_cpu_int8.tflite
```

`int8=True` gives int8 weights with float32 input and output. That is what the
CPU target expects, and it is ~3.3 MB versus 12 MB for float32.

### Export for Edge TPU (currently broken — see above)

Full-integer quantisation (uint8 in and out) is required by
`edgetpu_compiler`, and it is that step which destroyed the classification head
in the current model. If you attempt it, **validate detection scores on real
images before deploying**, not just that the file loads:

```bash
edgetpu_compiler -s best_full_integer_quant.tflite
cp best_full_integer_quant_edgetpu.tflite models/heron_edgetpu.tflite
```

### A note on resolution

A 320×320 export runs 4× faster (220 ms vs 920 ms on the Coral) but measured
**10 points lower recall** — 86% vs 96% at threshold 0.75, with distant birds
worst affected. Since inference now happens on the trigger frame, 640 is fast
enough, so the resolution is not worth trading.

### Collecting more training data

The model was trained on ~180 close-up heron images with **no negative
examples**, which is why non-heron frames occasionally spike to 0.70. The most
valuable thing you can add is negatives from your own pond: water ripple,
ducks, foliage, moving shade. Label them via the dashboard's Unclassified panel
and they are written to `media/labeled/` in YOLO format.

---

## Troubleshooting

Run `./preflight.py` first — it diagnoses most of this and prints the fix.

### No detections at all

1. Is the current time inside `active_hours`?
2. Lower `motion.sensitivity` (try `2000`)
3. Lower `ai.confidence_threshold` (try `0.6`, accepting more false positives)
4. Upload a clip via the dashboard's Test Upload to confirm inference works
5. `sudo journalctl -u heron-deterrent -f`

### False triggers

Raise `ai.confidence_threshold` toward `0.85`. Recall falls to about 72% at that
point, so prefer collecting negative training images instead.

### Model will not load

The error names the fix. Most often the model is missing — they are gitignored,
so a fresh clone has none. On Mendel, a segfault here means pip's
`tflite-runtime` is shadowing the system one: `pip uninstall tflite-runtime` and
ensure the venv has `include-system-site-packages = true`.

### Camera not detected

```bash
v4l2-ctl --list-devices
ls /dev/video*
sudo fuser -v /dev/video0        # something else holding it?
```

Repeated reconnects usually mean USB power, not a bad camera. Use a powered hub.

### No sound

```bash
aplay -l                          # list devices
speaker-test -t wav -c 2
amixer                            # find the right control name
```

### Edge TPU not detected

```bash
lspci -nn | grep 089a
dmesg | grep apex
python3 -c "from pycoral.utils.edgetpu import list_edge_tpus; print(list_edge_tpus())"
```

### Disk filling

```bash
df -h /home                       # on Coral this is a separate 2 GB partition
du -sh media/* | sort -h | tail
```

Lower `storage.max_media_gb`. The cap is enforced every five minutes.

---

## License

Open source — free for personal and commercial use.

## Contributing

Pull requests and issues are welcome.
