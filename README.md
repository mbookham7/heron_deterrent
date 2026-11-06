# Heron Deterrent System

An edge AI system that watches a live camera feed, detects herons with a YOLOv8
TFLite model, plays a deterrent sound to scare them away, and raises SMS and
MQTT alerts — all running locally on a Coral Dev Board or a Raspberry Pi 4.

**Deploying at the pond? Go straight to [DEPLOY.md](DEPLOY.md).** This file is
the reference; that one is the checklist.

---

## How It Works

```
Camera (5 fps) → Frame type (day / IR / dark / unsettled) → Motion detection
                                                                 │
              ┌──────────────────── day ─────────────────────────┤
              │                                                  │ IR, dark,
     Inference on the trigger frame                              │ establishing,
              │                                                  │ switching
  heron ──────┤────── not heron                                  │
    │                     │                              Record clip, NO inference
  Deterrent (if the       Capture clip,                  label "motion" (unclassified)
  mode allows) FIRST      re-check 5 key frames                  │
    │                     │                              "To Triage" panel
  Capture clip, save      "To Triage" panel
    │
  SQLite ── Dashboard ── SMS / MQTT
```

The ordering matters. Inference runs on the frame that triggered the motion, so
the speaker fires about **1 second** after the bird appears. An earlier version
captured a five-second clip first and took **9.6 seconds** — by which time the
heron had usually left.

If the trigger frame is inconclusive, the system falls back to capturing a clip
and scoring five key frames, so a bird caught mid-wingbeat is not written off.

The system runs **round the clock**. There is no time-of-day window: what it does
depends on what kind of image the camera is producing, below.

## Day and night

Herons feed hardest at first light, which in a UK November is well before
sunrise. So night is a first-class mode, not an afterthought — and it is a
different problem, not a darker version of the same one.

### Why an infrared frame is not just a dim daylight frame

When it gets dark the camera switches its own infrared LEDs on and starts
sending black-and-white frames. Those frames are lit by a torch strapped to the
camera, in a colour your eye cannot see. That changes what things look like:

- **Brightness falls off with distance.** Near things are bright, the far bank
  is dark. In daylight the sun lights everything evenly.
- **Water absorbs infrared**, so the pond reads almost black.
- **Leaves reflect infrared strongly** (the *Wood effect*), so a planted margin
  glows white — exactly the background a heron stands against.

A grey heron against a dark-green bank has decent contrast by day. Under IR it
may be a mid-grey bird against a glowing white bank: different, possibly worse.
A model trained only on daylight photos has never seen that, and will not
transfer reliably.

Two rules follow, and the code enforces both:

1. **Separate models, separate training data, never mixed.** At night there will
   be thousands of empty-pond IR frames and very few with a heron in them. Put
   them in one training set with the daylight photos and the model learns the
   shortcut "black-and-white means no heron" — confidently, and exactly
   backwards. So labelled images go to `media/labeled/day/`, `ir/` or `dark/`,
   and nothing can file one in the wrong tree.
2. **No fake IR.** Greyscaling a daylight photo gives a plausible-looking image
   that is physically wrong: it is how the scene reflects *visible* light, with
   the colour thrown away, not how it reflects infrared. Water would not go dark
   and leaves would not glow. There is no such conversion anywhere, and there
   should not be one.

Until real IR heron footage exists to train on, **night runs motion-only**: any
movement records a clip, labelled "motion — unclassified", and raises a
dashboard alert. At four in the morning, movement at the pond is very probably
wildlife, so that is a reasonable signal on its own — and it saves running a
~1 s inference on every moth.

### How the frame type is decided

From the pixels, never the clock — the camera decides when it is night, and a
clock-based guess drifts with the seasons.

| Frame type | What it means | How it is recognised |
|---|---|---|
| `day` | Colour, adequately lit | Channels differ (there is colour) and it is bright enough |
| `ir` | The camera has cut to infrared | Red, green and blue are effectively identical |
| `dark` | Colour but very dim — a camera with **no** IR LEDs, at night | Some colour noise, low brightness |

`dark` is treated like night: the daylight model does as badly on a murky colour
frame as on an IR one.

At dawn and dusk the camera can flick between IR and colour, and frames caught
mid-switch are neither. So the mode only changes after **ten consecutive frames
agree** (about 2 s) — the same idea as a readiness probe's `successThreshold`.
Until then the state is `establishing` (just started) or `switching`. Motion in
those windows is **still recorded**, unclassified, because dawn is exactly when
the bird turns up; it just is not judged by any model, since nobody knows which
one applies.

When labelling, the dashboard says which training set the current frame goes
into. That decision uses each frame's type **as recorded live when it was
captured**, not a re-check of the saved video: video compression puts a faint
colour cast on a black-and-white frame, enough to make an IR frame look like
daylight.

### What runs in each mode

Set per mode in `config.yaml` under `modes:` — see Configuration. Today:

| Mode | Model | Deterrent | Cooldown |
|---|---|---|---|
| `day` | `heron` (the daylight model) | **silent** | 10 s |
| `ir` | none — motion-only | silent | 45 s |
| `dark` | none — motion-only | silent | 45 s |

The night cooldown is longer on purpose: moth clips ten seconds apart are
near-duplicates, which bloat the dataset and leak across a train/validation
split.

Optionally, `modes.ir.shadow_model: heron` runs the daylight model on IR
triggers and **logs** its confidence in the clip's metadata without ever acting
on it — a cheap way to measure how badly it transfers.

---

## Measured performance

All figures measured on a Coral Dev Board (Mendel 5, quad Cortex-A53), CPU
inference, 640×640 model:

| Metric | Value |
|---|---|
| Inference | 920 ms/frame |
| Motion → deterrent sound | **1.02 s** |
| Saved clip size | 117 KB (5 s, 640×480) |
| Resident memory | ~110 MB |

### Accuracy — read the caveat, and re-derive with the script

Reproduce every number here with `python tools/measure_thresholds.py`. That
script simulates the camera before scoring positives (centre-crop to 4:3,
downsample to 640×480) so the harness matches deployment; see
`tools/camera_sim.py` for why not doing so overstates recall by roughly a
third. Numbers below come from that script on the current media/ and
heron-data/val/ snapshots.

| Metric | Value |
|---|---|
| Positives median (per-frame) | 0.887 |
| Recall @ threshold 0.75 | 0.84 |
| Recall @ current `deter_threshold` = 0.90 | **0.41 (18/44)** |
| Precision @ 0.75 (positives) | 1.00 |
| Real camera frames with no heron scoring ≥ 0.75 | 18% (50 / 275) |
| Real camera frames with no heron scoring ≥ 0.90 | **0% (0 / 275)** |
| Worst false score on those frames | 0.894 |

Positives are the 44 validation images from `heron-data`, centre-cropped to
4:3 and scaled to the camera's 640×480. **They are the same stock-photo
domain as the training set, so they flatter the model and say nothing about
performance at a pond** — even with the camera simulation applied. Negatives
are 275 frames of real camera footage from `media/` — an indoor bench test,
not the deployment scene.

The single most important fact: **at the shipped `deter_threshold` of 0.90 no
negative frame clears the bar, but recall on stock positives is 0.41 (18/44).**
The model catches fewer than half the herons that a stock camera would show
it. The distributions overlap — positives median 0.887, negatives p99 0.863 —
so no threshold separates them cleanly. The fix is negative training data, not
a different threshold. The 0.90 setpoint chooses to miss more than half the
herons rather than fire the deterrent at nothing, which is the right trade
for a bang whose false alarm is nearly free.

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
  alert_vote_k: 3              # need k of the last m key frames above alert
  alert_vote_m: 5
  num_threads: 4               # CPU inference only; ignored when target is edgetpu
  labels:
    0: "heron"                 # single-class model; more entries fail at startup

modes:                         # one per frame type — see "Day and night"
  day:
    model: heron               # registered model name, or null = motion-only
    deter_threshold: 0.90      # fires the deterrent (fast path, single frame)
    alert_threshold: 0.95      # governs recording + SMS (voted across frames)
    record_threshold: 0.30     # collect mode only
    deterrent: false           # THE switch for the sound; nothing else gates it
    cooldown_seconds: 10
  ir:
    model: null                # no IR model yet
    deterrent: false           # true here is a startup error: no model, no deterrent
    cooldown_seconds: 45
  dark:
    model: null
    deterrent: false
    cooldown_seconds: 45
```

Incoherent mode config is a **startup error** that lists every problem at once:
a motion-only mode with the deterrent armed, an unregistered model, thresholds on
a model-less mode, or leftover keys from the old layout (`system.active_hours`,
`ai.deter_threshold`, `ai.alert_threshold`, `ai.confidence_threshold`,
`collect_mode.record_threshold`, `collect_mode.deterrent_enabled`).

Two thresholds, because one number cannot do both jobs. A false BANG is nearly
free — the sound is a bang, not a bill — and being late to a bird is expensive,
so `deter_threshold` is optimistic and acts on a single frame. A false ALERT
costs your attention and your trust in the dashboard, so `alert_threshold` is
strict and confirmed across frames (`k` of the last `m` key frames must clear
it). Re-derive both from footage with `tools/measure_thresholds.py`; see
"Accuracy" for what those numbers mean today.

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

### Frames are letterboxed, not stretched

The other half of the contract with the training repo, and the half that used to
be broken. Three things have to agree between the two repos: the model file, the
class count, and the **preprocessing convention**.

The camera gives 640×480 (4:3). The model input is 640×640 (square). Ultralytics
trained the model by *letterboxing* — scale by one ratio so the aspect ratio is
preserved, then pad the remainder to square with grey `114`. This app used to
`cv2.resize()` straight to 640×640 instead, stretching every frame vertically by
1.33× and showing the model a distortion it never saw in training. On a tall thin
bird that is the worst axis to get wrong.

`ai/letterbox.py` is now the single definition of that convention, and the boxes
the model returns are mapped back out of the padded square before they leave
`infer()` — so a `Detection.bbox` always describes the frame that was passed in,
not the square the model saw. This matters beyond drawing: the dashboard's
labelling path writes YOLO annotation files, and a box left in square
coordinates would be wrong by the height of the padding in every one of them.

The class count is enforced at startup by `ClassCountMismatch`. The
preprocessing convention is enforced by `tests/test_letterbox.py`, which pins it
against Ultralytics' own `LetterBox` and `scale_boxes`:

```bash
python tests/test_letterbox.py         # no pytest needed; runs on the board
```

The two Ultralytics cross-checks skip if `ultralytics` is not installed, which it
is not on a board. **Run them on a dev machine** (`pip install ultralytics`)
before trusting a change to preprocessing — without them the file only asserts
that the code agrees with itself.

### Full configuration reference

| Key | Default | Description |
|---|---|---|
| `motion.sensitivity` | `5000` | Minimum contour area to trigger. Lower = more sensitive |
| `motion.cooldown_seconds` | `10` | Gap between triggers while the frame type is establishing/switching. Settled modes use `modes.<mode>.cooldown_seconds` |
| `ai.target` | `cpu` | `cpu` or `edgetpu` — the only hardware switch |
| `ai.alert_vote_k` / `alert_vote_m` | `3` / `5` | N-of-M voting on the alert path (see `ai/detector.py`) |
| `ai.num_threads` | `4` | CPU threads; ignored on `edgetpu` |
| `ai.labels` | `{0: heron}` | Must match the model's class count exactly |
| `alert.sms_enabled` | `false` | Needs `/etc/heron-deterrent.env` (see below) |
| `alert.sms_throttle_minutes` | `15` | Minimum gap between SMS messages |
| `audio.volume` | `80` | Percent, applied via `amixer` |
| `audio.sound_path` | `./sounds/` | Startup **fails** if this has no `.wav` files |
| `storage.retention_days` | `365` | Age-based deletion |
| `storage.max_media_gb` | `15.0` | **Hard size cap** on clips a model looked at (`heron/`, `other/`, `unknown/`). Oldest first |
| `storage.max_motion_gb` | `5.0` | **Separate hard cap** on unclassified clips (`motion/`), so a noisy night only evicts night clips |
| `storage.max_labeled_gb` | `5.0` | Soft warning on `labeled/`. Never deletes |
| `storage.media_path` | `./media/` | Clips, grouped by label |
| `storage.db_path` | `./data/heron.db` | SQLite database |
| `mqtt.enabled` | `false` | Optional MQTT publishing |
| `mqtt.broker` | `localhost` | Broker hostname. Connects asynchronously, retries forever |
| `mqtt.port` | `1883` | Broker port |
| `frame_type.*` | see `config.yaml` | Thresholds for day/IR/dark. Tune with the numbers `./preflight.py` prints |
| `modes.<mode>.model` | `heron` / `null` | Registered model name, or `null` = motion-only (no inference) |
| `modes.<mode>.deter_threshold` | `0.90` (day) | Deterrent fires above this. At 0.90: recall 0.41 (18/44) on stock positives with camera simulation, zero false-positive frames. See "Accuracy" |
| `modes.<mode>.alert_threshold` | `0.95` (day) | Recording + SMS require this AND `k`-of-`m` voting |
| `modes.<mode>.record_threshold` | `0.30` (day) | Collect mode records anything above this |
| `modes.<mode>.deterrent` | `false` | Whether the sound may fire in this mode. The only switch for it |
| `modes.<mode>.cooldown_seconds` | `10` / `45` / `45` | Minimum gap between triggers |
| `modes.<mode>.shadow_model` | `null` | Motion-only modes: log this model's confidence, never act on it |
| `collect_mode.enabled` | `false` | Record borderline day events above `record_threshold`. Does not touch the deterrent |
| `system.camera_device_id` | `0` | V4L2 device number |
| `system.frame_rate` | `5` | Capture rate, and the rate clips are written at |

### Why there is a size cap as well as retention

Retention by age cannot protect a card. At 100 events per day a 2 GB partition
fills in about four months — long before a 365-day cutoff deletes anything. A
windy pond, or moths in front of IR LEDs, can generate far more than that.

The caps are the backstop that actually bounds growth, checked every five
minutes, oldest files first. There are **two pools, each with its own cap**:

| Pool | Directories | Cap |
|---|---|---|
| classified | `heron/`, `other/`, `unknown/` | `storage.max_media_gb` |
| motion | `motion/` | `storage.max_motion_gb` |

With one shared pool, a night of moth clips would evict the unlabelled daylight
footage — the clips most likely to contain a heron. Separate pools mean a noisy
night can only evict other night clips.

`labeled/` is in **neither pool** and is never deleted by retention or a cap —
deleting half of an image/annotation pair silently turns a labelled heron into a
"no heron here" example. It only triggers the `max_labeled_gb` warning. The
protection is recursive: any directory under `labeled/`, including ones added
later, is exempt.

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
[PASS] config       ai.target=cpu  classes=1
       day: model=heron deter=0.9 alert=0.95 record=0.3, silent, cooldown 10s
       ir: motion-only (no model), silent, cooldown 45s
       dark: motion-only (no model), silent, cooldown 45s
[PASS] modes        3 modes coherent; deterrent silent in every mode
[PASS] model        heron: heron_cpu_int8.tflite loaded on CPU/4 threads, input 1x640x640x3
[skip] edgetpu      ai.target is not 'edgetpu'
[PASS] sounds       1 file(s), 172 KB: startle_default.wav
[PASS] audio        device present (...), playback not tested (--skip-audio)
[PASS] camera       device 0 returned a 640x480 frame
[PASS] frame type   ir (ir 15 of 15 frames)  chroma 0.31  luma 88.2  -> ir: motion-only ...
[PASS] video        H.264 available — ...
[PASS] disk         ...
[PASS] storage      clip caps 15 GB classified + 5 GB motion = 20 GB on a 58.0 GB filesystem
[PASS] labeled      images by domain: day 0, ir 0, dark 0
[PASS] database     data/heron.db writable (24 KB)
[WARN] cpu temp     71.2 C
       -> Improve cooling: heatsink or fan on the SoC, ...
[skip] sms          alert.sms_enabled is false
```

(Illustrative, not a captured run.) Every failure prints the command that fixes
it. Exit status is 0 only when everything passes, so it can gate a deploy
script. `[WARN]` lines do not fail the run.

**Frame type** prints the raw chroma (how much colour) and luma (brightness)
behind its verdict. Run preflight once in daylight and once after dark: the two
readings should land clearly on either side of the `frame_type:` thresholds. If
the night reading lands in the dead band, the camera is probably tinting its IR
frames rather than sending pure black-and-white; raise `mono_max_chroma` above
the tint.

**CPU temperature** matters because the Pi lives in a sealed box. A throttling
board looks exactly like "inference got slower for no reason". It FAILs at 80 °C
or if the board is throttling or under-voltage *now*, and WARNs at 70 °C or if
either has happened since boot.

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
| Mode pill | Day / IR night / Dark night / establishing / switching; model or motion-only; deterrent armed or silent. Hover for live chroma/luma |
| Stats strip | Herons today, model detections, last heron, model alerts, and — separately — unclassified motion. Motion is never counted as a detection |
| Frame-type filter | Day / IR / Dark / Unsettled, across the lists and gallery. Every clip carries a frame-type badge |
| Live Feed | MJPEG stream, throttled to 2 fps so it cannot starve detection |
| Recent Alerts | CRITICAL and WARNING alerts with timestamps |
| Heron Detections | History with confidence bars |
| To Triage | Clips the model returned "unknown" for, plus unclassified motion clips — label them to build a dataset. The label dialog shows which training set (day/IR/dark) the frame goes to, and blocks saving until there is one |
| Heron Gallery | Confirmed heron clips |
| Test Upload | Upload a video to run inference against — the quickest end-to-end proof |

> Every inconclusive motion event creates a WARNING "unknown" alert and saves a
> clip. That is deliberate — it feeds the To Triage panel — but at a busy pond
> the alert feed gets noisy. Raise `motion.sensitivity` if so.

The Flask server is the development server. That is fine on a private LAN; do
not expose it to the internet.

### API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/live-stream` | MJPEG video stream (2 fps) |
| GET | `/api/system-status` | System health, frame state and mode JSON |
| GET | `/api/stats` | Counts for the stats strip (motion reported separately) |
| GET | `/api/alerts` | Recent alerts (param: `limit`) |
| GET | `/api/detections` | Detection history (params: `label`, `limit`, `frame_type` = day/ir/dark/unknown) |
| GET | `/api/heron-detections` | Heron-only detections (params: `limit`, `frame_type`) |
| GET | `/api/unclassified` | Unlabelled "unknown" and "motion" detections |
| GET | `/api/detection/<id>/frame/<i>/domain` | Which training set frame `i` would be filed under (param: `requested`) |
| POST | `/api/label` | Submit a manual label `{detection_id, label, bboxes, frame_index, domain?}`. 409 if the frame cannot be placed in a domain |
| POST | `/api/trigger-deterrent` | Manual speaker test. Plays regardless of mode |
| POST | `/api/upload-test-video` | Run inference on an uploaded video. Refused unless every key frame is a day frame |
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
│   ├── letterbox.py              # The preprocessing convention. Do not bypass
│   └── model_loader.py           # Target → model file → backend
│
├── tests/                        # pytest, or run any file directly on the board
│
├── capture/
│   ├── camera.py                 # Locked, self-healing camera wrapper
│   ├── frame_classifier.py       # day / IR / dark, with hysteresis
│   └── motion_detector.py        # MOG2 background subtraction
│
├── deterrent/
│   └── audio_player.py           # ALSA playback; fails loudly if silent
│
├── storage/
│   ├── database.py               # SQLite (thread-safe)
│   └── media_store.py            # Clips, retention, two size-capped pools, labeled/
│
├── alerts/
│   ├── alert_manager.py          # SMS + MQTT orchestration
│   └── mqtt_client.py            # Paho client with indefinite backoff
│
├── ui/
│   ├── app.py                    # Flask REST API
│   └── templates/index.html      # Dashboard
│
├── utils/
│   ├── config_loader.py          # YAML config loader
│   └── modes.py                  # Per-frame-type modes; validates modes:
│
├── models/                       # *.tflite — gitignored, copy in by hand
├── sounds/                       # *.wav — at least one required
├── media/                        # Auto-created: heron/ other/ unknown/ motion/ (clips)
│                                 #   labeled/{day,ir,dark}/{images,labels}/ (training data)
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
ducks, foliage, moving shade. Label them via the dashboard's To Triage panel
and they are written in YOLO format to `media/labeled/day/`, `ir/` or `dark/` —
one tree per image domain. **Point a training run at exactly one of them.** The
pre-split `media/labeled/images|labels` layout is no longer written; preflight
warns if anything is still there.

---

## Troubleshooting

Run `./preflight.py` first — it diagnoses most of this and prints the fix.

### No detections at all

1. What mode is it in? The dashboard's mode pill, or `./preflight.py`. At night,
   or while `establishing`/`switching`, no model runs and nothing is classified
   as a heron by design — look in To Triage for "motion" clips instead
2. Lower `motion.sensitivity` (try `2000`)
3. Lower `modes.day.deter_threshold` toward `0.75` (accepting more false positives, since
   at 0.75 recall on stock positives is 0.84 vs 0.41 at the shipped 0.90 — but 18%
   of your bench footage also clears that bar, so the deterrent will fire more)
4. Upload a clip via the dashboard's Test Upload to confirm inference works
5. `sudo journalctl -u heron-deterrent -f`

### False triggers

At the shipped `deter_threshold=0.90`, no frame of the current bench footage
clears the bar — so if the deterrent is firing at nothing, the trigger frames
are worse than what is already recorded, and you should look at
`media/heron/` for evidence before changing thresholds. Lowering `deter_threshold`
toward 0.75 raises recall on positives (0.41 → 0.84) but lets 18% of bench-style
frames through. The two distributions overlap — positives median 0.887,
negatives p99 0.863 — so no setting gives you both. **Collect negative training
images from your own scene instead:** use collect mode (`collect_mode.enabled:
true`) to have the device manufacture borderline clips for you to label through
the dashboard.

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

Lower `storage.max_media_gb` and `storage.max_motion_gb`; `./preflight.py`
fails if together they cannot fit on the filesystem. The caps are enforced every
five minutes. If `motion/` is the one growing, lengthen
`modes.ir.cooldown_seconds` — moths in front of IR LEDs are the usual cause.

---

## License

Open source — free for personal and commercial use.

## Contributing

Pull requests and issues are welcome.
