# Heron Deterrent System

An edge AI system that watches a pond, detects herons with a YOLOv8 TFLite
model, plays a deterrent sound to scare them away, and raises SMS and MQTT
alerts — all running locally on a **Raspberry Pi 4**, in a sealed box inside the
pond filter house, with an IP66 USB camera. It runs round the clock, in
different modes by day and by night.

**Deploying at the pond? Go straight to [DEPLOY.md](DEPLOY.md).** This file is
the reference; that one is the checklist. Building a fresh Pi from a blank card
is [docs/raspberry_pi4_setup.md](docs/raspberry_pi4_setup.md).

> **Coming from the Coral Dev Board?** It is legacy now — Google has archived
> the Coral software, and the Pi runs the same model about 3× faster. Its notes
> live in [docs/legacy/coral/](docs/legacy/coral/README.md).

### What has and has not been verified

Every Pi command in these docs ran on the build Pi (Raspberry Pi 4 Model B
Rev 1.5, 8 GB, Raspberry Pi OS 13 "Trixie", Python 3.13.5) on 2026-11-25.
Some things could not be checked that day, and the text says so wherever they
come up rather than stating them as fact. The main ones:

- **Nothing has been measured inside the sealed box** — every temperature here
  is from the bench, in open air.
- **Nothing has been measured at the pond** — no real heron, no outdoor scene,
  no outdoor night. Night was tested indoors only, by switching the room
  lights off: the camera cut to IR, the mode followed, and night motion was
  recorded unclassified.
- **The camera specification is the vendor's**, not measured. In particular
  whether the 96° field of view is horizontal or diagonal is open — see
  "Camera and mounting". (Whether 640×480 crops or squashes the sensor *was*
  measured: it crops.)

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
heron had usually left. (Both figures were measured on the Coral, where
inference took 920 ms. The Pi's inference is about 600 ms faster, so it should
be quicker still, but motion-to-sound has not been re-measured on the Pi.)

If the trigger frame is inconclusive, the system falls back to capturing a clip
and scoring five key frames, so a bird caught mid-wingbeat is not written off.

The system runs **round the clock**. There is no time-of-day window — the old
`system.active_hours` key is gone, and leaving it in `config.yaml` is a startup
error. What it does depends on what kind of image the camera is producing,
below.

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
wildlife, so that is a reasonable signal on its own — and it saves running an
inference on every moth.

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

On the Pi, settling took about 2 s after a service restart but **26 s after a
cold boot** — most likely the camera's auto-exposure finding its level. Motion
in that window is recorded as `establishing`, as designed. With the room lights
switched off, the ELP cut to IR and the mode read `day -> ir` 3.6 s after the
motion the lights caused.

**A switch can land in the middle of an event**, and on the bench it did: the
lights going off was itself motion, caught in `day` mode. The trigger frame was
inconclusive, so a clip was captured for the voting pass — and the mode
switched to `ir` partway through it. The clip was saved as `unknown_day_…` and
its database row says `frame_type: day`, but its per-frame record
(`clip_frame_types` in the metadata) correctly shows 7 `day` frames then 18
`ir`, and labelling goes by that per-frame record. The voting pass used to score
those mostly-IR key frames with the daylight model — the one place it judged
infrared frames. It now uses the same per-frame record to skip any key frame
whose type differs from the event's mode, logging `Voting pass: 3 of 5 key
frames are not day frames … still need 3 votes`. `alert_vote_k` does not
shrink, so a mixed clip needs as many agreeing daylight frames as a clean one:
a switch mid-clip can turn an event into `unknown`, never into an alert.
(Pinned by `tests/test_vote_frame_types.py`, which replays this bench event;
not yet seen live, since it needs the lights to change mid-event again.)

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

On the build Pi (Raspberry Pi 4, 8 GB, Raspberry Pi OS 13, LiteRT 2.4.0 with
its XNNPACK CPU delegate, 4 threads, 640×640 model), on the bench:

| Metric | Value | How it was measured |
|---|---|---|
| Inference | **314 ms** median (313–343) | 20 runs after 3 warm-ups, on random input — a fair timing, but it says nothing about the scores |
| Motion → trigger-frame verdict | 373 ms | One real event: `Motion detected` to `Trigger frame inconclusive` in the log, so it includes letterboxing and post-processing. A single sample |
| Motion → deterrent sound | **not measured on the Pi** | 1.02 s on the Coral. The deterrent is silent in every mode as shipped, so no real event has played it |
| Saved clip size | 170 KB mean (5 s, 640×480, 5 fps, H.264) | 16 indoor clips on the bench; outdoor clips will differ |
| Saved clip size, IR | 18–278 KB | 4 indoor IR clips. 278 KB is a person walking through; the small ones are near-static dark scenes |
| Resident memory | 206 MB | The running service, `ps` RSS |
| CPU, idle pond | ~55% of one core | The running service with no inference: motion detection, frame typing, dashboard |

For comparison, the Coral Dev Board (Mendel 5, quad Cortex-A53, CPU inference)
measured 920 ms per inference, 1.02 s motion-to-sound, 117 KB clips and ~110 MB
resident — full table in [docs/legacy/coral/](docs/legacy/coral/README.md).

### Accuracy — read the caveat, and re-derive with the script

The table below is the original stock-photo measurement; the alert threshold
was re-measured on field footage in 2026-11 (next section). The script now
needs a negatives set named explicitly (`--negatives DIR` or
`--negatives-labelled data/heron.db`) — `media/` is no longer one. That
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
are 275 frames of real camera footage from `media/` — an indoor bench test on
the Coral, not the deployment scene.

The single most important fact: **at the shipped `deter_threshold` of 0.90 no
negative frame clears the bar, but recall on stock positives is 0.41 (18/44).**
The model catches fewer than half the herons that a stock camera would show
it. The distributions overlap — positives median 0.887, negatives p99 0.863 —
so no threshold separates them cleanly. The fix is negative training data, not
a different threshold. The 0.90 setpoint chooses to miss more than half the
herons rather than fire the deterrent at nothing, which is the right trade
for a bang whose false alarm is nearly free.

These scores were measured with `tflite-runtime` and TensorFlow; the Pi runs
the same model file under LiteRT. The output shapes match, but **the scores
have not been re-derived under LiteRT on the Pi**. One spot check since: the
key frames of two heron videos scored on the Pi (LiteRT) and on a Mac
(TensorFlow) agreed to within 0.002.

### Alert threshold, measured (2026-11-27): 0.95 → 0.90

Two phone videos of a wild heron, uploaded through the dashboard, came back
"no alert": the model scored their key frames 0.92–0.95, and the alert needed
3 of 5 at 0.95. The threshold was re-measured with `tools/measure_thresholds.py`
after fixing it to vote the way the device does (5 s clips, `main.py`'s own 5
key frames, 3 must clear — it used to count any 3 frames of a whole file):

```bash
python tools/measure_thresholds.py --negatives <62 labelled clips> \
    --positives <the two videos> --positives-source camera
```

- **Negatives:** the 62 daylight clips labelled "background" or "person" in the
  dashboard — 1,550 frames, worst 0.845.
- **Positives:** the two videos, one heron, cut into 27 live-length clips; 16
  have the bird in view throughout (in the rest the phone pans off it,
  checked by eye).

| Alert threshold, 3 of 5 | False alerts (62) | Heron-in-view clips alerting (16) | Above worst negative |
|---|---|---|---|
| 0.95 (old) | 0 | **0** | +0.105 |
| 0.94 | 0 | 1 | +0.095 |
| 0.92 | 0 | 12 | +0.075 |
| **0.90 (now)** | **0** | **13** | **+0.055** |
| 0.88 | 0 | 15 | +0.035 |
| 0.85 | 0 | 16 | +0.005 |

The script picks the lowest threshold with no false alerts that also clears
the worst negative frame by `--margin` (0.05): 0.90, equal to
`deter_threshold`. It misses the heron when it is further off and scores
0.83–0.90.

**How far to trust it:** one bird, filmed on a phone rather than the ELP
camera, and windows of the same video are not independent. The negatives are
only the scenes this camera has seen so far; ducks, ripples or reeds at the
pond may score above 0.845 and eat the margin. Re-run it when there is pond
footage — `--negatives-labelled data/heron.db` uses every daylight clip you
have labelled as not a heron.

---

## Hardware

| Component | What was built | Notes |
|---|---|---|
| Board | Raspberry Pi 4 Model B, 8 GB | The service uses ~210 MB, so a 2 GB board should fit — not tried |
| OS | Raspberry Pi OS 13 (Trixie), 64-bit | Boot to the console, not the desktop — see "Sound output" |
| Camera | ELP USBFHD05MT-DL36 | IP66 dome, UVC. See "Camera and mounting" |
| USB hub | **Powered** hub between Pi and camera | Recommended, not fitted in the bench build. See below |
| Speaker | **Powered** speaker on the 3.5 mm jack | The jack is line-level; it cannot drive a passive speaker |
| Storage | microSD; the build used 128 GB | The clip caps (20 GB) need ≥ 32 GB. `preflight` fails if they cannot fit |
| Power | Official 5 V / 3 A USB-C supply | Under-voltage shows as `throttled=` bits in `vcgencmd get_throttled` |
| Network | Wi-Fi or Ethernet | Optional — detection works offline |

### Why the camera goes through a powered hub

A Pi 4's USB ports share a power budget of about **1.2 A** across all four. The
camera's infrared LED array only switches on at night, and an LED array is
exactly the kind of load that can push a shared budget over. When it does, the
voltage sags and the camera browns out.

The trap is *when* that happens: **only ever at night**. A brownout drops the
camera, the code reconnects, frames stop and start — and it looks exactly like a
bug in the night path, which is the newest and least-tested code. A powered hub
takes the camera off the Pi's budget and removes the question. If night
reconnects show up without one, suspect power before code.

The 1.2 A figure is Raspberry Pi's documented limit for all USB ports combined
(not measured here), and the camera's IR current has not been measured either — so this is a precaution against a plausible
failure, not a diagnosed one. The bench build had no hub. Its IR LEDs ran for
a few minutes indoors with the camera straight on the Pi: no camera read
failures, no USB errors in the kernel log, `throttled=0x0`. That is too short,
and too warm and dry, to rule a brownout out on a cold night.

### The filter-house enclosure

The Pi lives in a **sealed IP-rated box inside the pond filter house**. Two
choices follow from "sealed":

- **Bare board on standoffs with heatsinks**, not the official Pi case. The
  official case is closed plastic around a board that runs warm; inside a
  second sealed box it is a box in a box. Standoffs and heatsinks give the SoC
  the whole volume of air to shed heat into.
- **Desiccant in the box.** A filter house is damp, and a sealed box traps
  whatever moisture was in it when it was closed, which then condenses every
  cold night.

**How warm it actually gets in there has not been measured.** On the bench, in
open air: 55 °C idle at boot, ~60–64 °C with the service running — later the
same evening, 68–69 °C — and 68.6 °C after only eight seconds of back-to-back
inference. **Half a minute of continuous inference in open air reached 79.8 °C
and tripped the firmware's soft temperature limit** (`get_throttled` went to
`0x80000`, and `0xe0008` — throttling *now* — at 82.7 °C on a second run).
Real detection only runs inference on motion, not continuously, but a busy
night of triggers is closer to that test than to idle. That is before the box
adds anything: heatsinks are not optional, and the box may need more.
Once it is in the box, check it after a few hours, and again on a warm day:

```bash
vcgencmd measure_temp; vcgencmd get_throttled    # throttled=0x0 is what you want
```

`preflight` reports both. A throttling board looks exactly like "inference got
slower for no reason".

A single reading only catches one moment. To get the whole history — overnight,
a warm afternoon, the day the box was closed — log it every five minutes with a
cron job for `pi` (no sudo; installed on the build Pi and confirmed writing):

```bash
crontab -l     # check pi has no crontab first — the next line replaces it
cat > /tmp/heron.cron <<"EOF"
*/5 * * * * echo "$(date "+\%F \%T"),$(vcgencmd measure_temp | cut -d= -f2),$(vcgencmd get_throttled | cut -d= -f2)" >> /home/pi/heron_temps.csv
EOF
crontab /tmp/heron.cron && rm /tmp/heron.cron
```

Each line is `time,temperature,throttle state`, e.g.
`2026-11-25 18:45:01,57.4'C,0x0`. Anything but `0x0` in the last column means
it has throttled or seen under-voltage. The `\%` escapes are needed: cron treats
a bare `%` as a newline. Remove it with `crontab -r`. At one ~35-byte line every
five minutes the file grows about 4 MB a year.

---

## Camera and mounting

### The camera

**ELP USBFHD05MT-DL36.** Vendor specification — not measured here:

| | |
|---|---|
| Housing | IP66 dome |
| Sensor | OmniVision OV2710, 1080p |
| Lens | M12 mount, fixed focus, 3.6 mm |
| Field of view | 96° horizontal (vendor figure) |
| Night | Infrared LEDs with an IR-cut filter |
| Interface | USB, UVC — no driver needed |

What was observed on the Pi: it enumerates as `32e4:9230 HD USB Camera` and
appears as `/dev/video0`. It offers MJPG and YUYV at 1920×1080, 1280×1024,
1280×720, 1024×768, 800×600, 640×480 and 320×240. Through OpenCV's default
(YUYV) it delivers 640×480 at 30 fps but **1920×1080 at only 6 fps** — worth
knowing before reaching for a higher resolution.

### Capture resolution: 640×480, unchanged

The build kept `system.resolution` at **640×480**. That choice matters more than
it looks, because **it fixes the image domain of every clip the system ever
collects** — and those clips are the future training data. Change it later and
everything already labelled was captured differently.

Reasons to keep 640×480:

- **Every threshold in this README was measured at 640×480.**
  `tools/camera_sim.py` simulates exactly this camera output, so changing it
  invalidates the numbers above.
- **The model sees 640×640 regardless.** A 640×480 frame letterboxes into it
  with no downscaling. A 1080p frame is shrunk to fit, so the extra pixels never
  reach the model.
- **It is cheap**: 30 fps available natively, small clips, little CPU.

**It crops, it does not squash.** The OV2710 is a 16:9 sensor and 640×480 is
4:3. A UVC camera can get there by *cropping* the sides of the sensor (the image
stays undistorted, the horizontal view narrows) or by *squashing* the whole
sensor into 4:3 (full view, but everything is horizontally compressed — and a
heron is a tall thin bird). On the bench, the same lit scene was captured at
1920×1080 and at 640×480 and every candidate region of the large frame was
scored against the small one. The best match was exactly the **centred
1440×1080 window** — a 4:3 crop — at correlation 0.982, against 0.589 for the
whole frame squashed. So:

- **No distortion.** The model sees the bird in its true proportions.
- **The vertical view is the sensor's full ~64°**, which the mounting
  arithmetic below relies on.
- **The horizontal view at 640×480 is about 80°**, not 96°:
  `2 × atan(tan 48° × 3/4) ≈ 80°` — still assuming the vendor's 96° is the
  horizontal figure for the full sensor.

### Mounting distance — and how to redo the sum for another lens

The model can only find a heron at roughly the sizes it was trained on. So the
question is: at what distance does a heron fill the frame the way the training
herons did?

**1. How tall the training herons are.** Box heights in all 224 heron labels in
`heron-data` (train and val), as a fraction of frame height after the same 4:3
centre-crop the camera simulation applies:

| p10 | p25 | median | p75 | smallest |
|---|---|---|---|---|
| 0.30 | 0.42 | 0.59 | 0.77 | 0.16 |

**2. How much scene the camera sees.** Vertical field of view from horizontal,
for a 16:9 sensor:

```
vertical = 2 × atan( tan(horizontal / 2) × 9/16 )
         = 2 × atan( tan(48°) × 0.5625 )
         = 2 × atan( 0.625 )
         ≈ 64°
```

At distance *d*, the visible scene height is `2 × d × tan(32°) ≈ 1.25 × d`.

**3. How big the heron is in the frame.** A standing grey heron is roughly
**0.9 m** tall (an assumption — hunched, it is less). So its fraction of the
frame height is `0.9 / (1.25 × d)`, and the distance that gives a fraction *f*
is:

```
d ≈ 0.72 / f
```

| Heron fills | Matches the training set's | Distance |
|---|---|---|
| 0.77 of frame height | p75 | ~0.9 m |
| 0.59 | median | **~1.2 m** |
| 0.48 | ~p30 | 1.5 m |
| 0.42 | p25 | ~1.7 m |
| 0.30 | p10 | ~2.4 m |
| 0.16 | the single smallest box | 4.5 m |

**So: mount 1.5–4.5 m from where the heron stands, and nearer the bottom of
that range.** The table is why "nearer the bottom" is not a nicety. At 1.5 m
the heron is the size of a typical training bird. By 2.4 m it is smaller than
nine in ten of them. At 4.5 m it matches only the single smallest box in the
set — that is the edge of what the model has seen, not a comfortable working
distance. Distant birds were already the ones the model missed on the Coral.

**Two assumptions, either of which moves the answer:**

- **96° is horizontal.** Camera vendors often quote the *diagonal*. If it is
  diagonal, the vertical view is narrower, herons look bigger, and every
  distance stretches.
- **0.9 m heron.**

(A third used to be here — that 640×480 crops the sensor rather than
squashing it. That has since been measured: it crops. See "Capture
resolution".)

**Redoing it for a different lens:** replace 96° in step 2 with the new
horizontal field of view (and 9/16 with the sensor's aspect ratio, if it is
not 16:9), get the new `scene height ÷ distance` factor, and divide 0.9 by it.
Then divide by the fractions in step 1.

### The dome: IR washout

**The failure.** At night the image goes milky or washed-out white, sometimes
with a bright haze or ring. The camera's own IR LEDs are shining back into the
lens: a dome camera has its LEDs and lens behind the same clear dome, and a
foam gasket around the lens is what keeps the LED light from leaking sideways
through the dome material into it.

**The cause** was that gasket **not seated flush against the inside of the
dome**. Light from the LEDs got past it and reflected straight into the lens.
(Found on this camera by the owner. On the bench, with the gasket seated, the
IR image indoors was clean — no haze, no ring — so there was nothing to
diagnose. Rain and dirt on the dome outdoors are untested.)

**The diagnostic.** If the image improves with the IR LEDs **off** (or covered),
it is internal reflection, not the scene. A dirty or wet dome *outside* scatters
light too, but it stays bad either way.

For the software, this matters twice over. A washed-out frame has no colour, so
it reads as `ir` (on the bench, a white-out frame measured chroma 0.00 and luma
255 — classified `ir`). And at night nothing is classified anyway, so a washed-
out camera produces an endless stream of "motion" clips that look perfectly
normal in the dashboard list. Look at a night clip, not just the counts.

---

## Sound output

The deterrent plays to the card named in **`audio.card`** — `Headphones`, the
Pi's 3.5 mm jack, as shipped — passing it to both `aplay` (`-D
plughw:CARD=Headphones`) and `amixer` (`-c Headphones`). A card that is not
there is a startup error, listing the cards that are. Set `audio.card: null`
to fall back to ALSA's default device instead.

It is named because the default is a trap. On the build Pi, with the default
forced to HDMI, the named card still played through the jack. Before
`audio.card` existed, the deterrent played to the default, and three things on
the Pi made that harder than it sounds — all three hit on the build. They still
matter with `audio.card: null`, and for anything else that uses the default
(`aplay` by hand, `speaker-test`), which is why `setup.sh` still writes
`/etc/asound.conf`:

1. **ALSA's default is HDMI**, not the 3.5 mm jack. Out of the box the
   deterrent plays to a monitor that is not there.
2. **The fix has to be system-wide.** The usual per-user file, `~/.asoundrc`,
   is removed at login by the desktop's volume plugin (`libvolumepulse.so` —
   it builds that path, and the file vanished across the first reboot). So it
   works until the first reboot and then silently stops. `setup.sh` writes
   `/etc/asound.conf` instead, naming the jack as `Headphones`.
3. **ALSA will not accept the obvious form.** `defaults.pcm.card "Headphones"`
   is rejected (`card is not a string`), and the error breaks ALSA entirely
   until the file is removed. The working form is a `pcm.!default` block — see
   `setup.sh`.

**`preflight` now catches the routing, not the speaker.** Its audio check
watches which card is actually playing and **FAILs if it is HDMI** — with
`--skip-audio` it plays one second of silence to find out. (It used to pass
whenever `aplay` succeeded, which it does when playing faithfully to HDMI, and
named the first card in the list rather than the one that played.) What it
still cannot know is whether the speaker on the jack is powered, turned up and
connected: on the build, the jack was proven working with headphones while
the speaker was silent. **You still have to listen.** To see which output is
playing by hand:

```bash
aplay -q sounds/startle_default.wav & sleep 0.4
head -1 /proc/asound/card*/pcm0p/sub0/status      # the jack is the Headphones card: want RUNNING
```

**Boot to the console, not the desktop.** It is what removes `~/.asoundrc`, it
costs heat in a sealed box, and nothing here needs it:

```bash
sudo systemctl set-default multi-user.target && sudo reboot
```

**`audio.volume` is not what it looks like.** The app sets it with `amixer` as a
percentage, and the scale is not linear: on the Pi's jack, the shipped **80% is
−17.3 dB** and 100% is +4 dB — about 21 dB louder. A deterrent is meant to
startle, so judge it from where the heron stands, outdoors, and use the
speaker's own gain as well. (The shipped value has not been changed.)

---

## Configuration

`config.yaml` in full. **`ai.target` is the only hardware switch**, and it names
the inference *backend*, not the board:

```yaml
ai:
  target: cpu                  # "cpu" or "edgetpu" — see Edge TPU status below
  alert_vote_k: 3              # need k of the last m key frames above alert
  alert_vote_m: 5
  num_threads: 4               # CPU inference only; ignored when target is edgetpu
  labels:
    0: "heron"                 # single-class model; more entries fail at startup

modes:                         # one per frame type — see "Day and night"
  day:
    model: heron               # registered model name, or null = motion-only
    deter_threshold: 0.90      # fires the deterrent (fast path, single frame)
    alert_threshold: 0.90      # governs recording + SMS (voted across frames)
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
| `cpu` | `models/heron_cpu_int8.tflite` | `ai-edge-litert` on the Pi; `tflite-runtime` on the legacy Coral |
| `edgetpu` | `models/heron_edgetpu.tflite` | Edge TPU delegate — legacy Coral only, and broken; see below |

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
venv/bin/python tests/test_letterbox.py    # no pytest needed; runs on the board
```

The two Ultralytics cross-checks skip if `ultralytics` is not installed, which it
is not on a board. **Run them on a dev machine** (`pip install ultralytics`)
before trusting a change to preprocessing — without them the file only asserts
that the code agrees with itself.

The same "squash vs letterbox" question arises one step earlier, at the camera:
there, 640×480 is a crop of the sensor, so nothing is squashed either — see
"Capture resolution" above.

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
| `audio.volume` | `80` | Percent, applied via `amixer` before every play. Not linear: 80% is −17.3 dB on the Pi's jack — see "Sound output" |
| `audio.sound_path` | `./sounds/` | Startup **fails** if this has no `.wav` files |
| `audio.card` | `Headphones` | ALSA card id the deterrent plays to (`aplay -l`); `Headphones` is the Pi's 3.5 mm jack. Startup **fails** if it is not present. `null` = ALSA's default device — see "Sound output" |
| `storage.retention_days` | `365` | Age-based deletion |
| `storage.max_media_gb` | `15.0` | **Hard size cap** on clips a model looked at (`heron/`, `other/`, `unknown/`). Oldest first |
| `storage.max_motion_gb` | `5.0` | **Separate hard cap** on unclassified clips (`motion/`), so a noisy night only evicts night clips |
| `storage.max_labeled_gb` | `5.0` | Soft warning on `labeled/`. Never deletes |
| `storage.media_path` | `./media/` | Clips, grouped by label |
| `storage.db_path` | `./data/heron.db` | SQLite database |
| `mqtt.enabled` | `false` | Optional MQTT publishing |
| `mqtt.broker` | `localhost` | Broker hostname. Connects asynchronously, retries forever |
| `mqtt.port` | `1883` | Broker port |
| `frame_type.*` | see `config.yaml` | Thresholds for day/IR/dark. Tune with the numbers `preflight` prints |
| `modes.<mode>.model` | `heron` / `null` | Registered model name, or `null` = motion-only (no inference) |
| `modes.<mode>.deter_threshold` | `0.90` (day) | Deterrent fires above this. At 0.90: recall 0.41 (18/44) on stock positives with camera simulation, zero false-positive frames. See "Accuracy" |
| `modes.<mode>.alert_threshold` | `0.90` (day) | Recording + SMS require this AND `k`-of-`m` voting |
| `modes.<mode>.record_threshold` | `0.30` (day) | Collect mode records anything above this |
| `modes.<mode>.deterrent` | `false` | Whether the sound may fire in this mode. The only switch for it |
| `modes.<mode>.cooldown_seconds` | `10` / `45` / `45` | Minimum gap between triggers |
| `modes.<mode>.shadow_model` | `null` | Motion-only modes: log this model's confidence, never act on it |
| `collect_mode.enabled` | `false` | Record borderline day events above `record_threshold`. Does not touch the deterrent |
| `system.camera_device_id` | `0` | V4L2 device number |
| `system.frame_rate` | `5` | Capture rate, and the rate clips are written at |
| `system.resolution` | `640×480` | Fixes the image domain of all collected data — see "Capture resolution" |

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

The default caps (15 + 5 GB) are sized for a Pi's card. On the Pi the repo is
on the root filesystem, so `df -h /` is the right check. (On the legacy Coral
it was not — see its notes.)

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
misconfigured would be worse. (SMS has never been switched on — there is no
Twilio account yet — so none of this has been exercised on the Pi.)

---

## preflight.py

One command that checks everything before you start the service. Run it on the
board, from the repo root, **through the venv**:

```bash
venv/bin/python preflight.py                # all checks, plays a deterrent sound
venv/bin/python preflight.py --skip-audio   # same, but stay quiet
```

Not `./preflight.py`. Its first line runs the *system* Python, which has none of
the venv's packages — so on the Pi it reports `[FAIL] model` and
`[FAIL] camera ... opencv not installed` on a perfectly healthy install.

A real run on the build Pi, with the service running:

```
[PASS] config       ai.target=cpu  classes=1
       day: model=heron deter=0.9 alert=0.9 record=0.3, silent, cooldown 10s
       ir: motion-only (no model), silent, cooldown 45s
       dark: motion-only (no model), silent, cooldown 45s
[PASS] modes        3 modes coherent; deterrent silent in every mode
[PASS] model        heron: heron_cpu_int8.tflite loaded on CPU/4 threads, input 1x640x640x3
[skip] edgetpu      ai.target is not 'edgetpu'
[PASS] sounds       1 file(s), 172 KB: startle_default.wav
[PASS] audio        deterrent plays to bcm2835 Headphones (audio.card: Headphones); not played (--skip-audio)
[skip] camera       device 0 is held by the running heron-deterrent service
       -> That is expected. To test the device itself:
       ->   sudo systemctl stop heron-deterrent && venv/bin/python preflight.py
       -> Or check the live feed and frame type: curl -s localhost:5000/api/system-status
[PASS] video        H.264 available — /usr/bin/ffmpeg with libx264 (preset veryfast)
[PASS] disk         104.9 GB free of 116.8 GB on / (10% used)
[PASS] storage      clip caps 15 GB classified + 5 GB motion = 20 GB on a 116.8 GB filesystem
[PASS] labeled      images by domain: day 0, ir 0, dark 0
[PASS] database     data/heron.db writable (8 KB)
[PASS] cpu temp     62.3 C  never throttled since boot
[skip] sms         alert.sms_enabled is false

All 11 checks passed.
```

`bcm2835 Headphones` is the 3.5 mm jack. The check plays exactly as the
deterrent does — to `audio.card` — and watches which card actually runs. If
that turns out to be HDMI (`audio.card: null` with ALSA's default on HDMI, as
tested on the build), it reports `[FAIL] audio  sound goes to HDMI
(vc4-hdmi-0), not the speaker` with the fix; a named card that does not exist
FAILs with the list of cards that do. With the service stopped the camera and
frame-type checks run too, for 13 checks in all.

Every failure prints the command that fixes it. Exit status is 0 only when
everything passes, so it can gate a deploy script. `[WARN]` lines do not fail
the run.

**Frame type** prints the raw chroma (how much colour) and luma (brightness)
behind its verdict. Run preflight once in daylight and once after dark: the two
readings should land clearly on either side of the `frame_type:` thresholds. If
the night reading lands in the dead band, the camera is probably tinting its IR
frames rather than sending pure black-and-white; raise `mono_max_chroma` above
the tint. On the bench the ELP read chroma 33.5 / luma 93.5 lit indoors,
chroma 5.8 / luma 11 in a dim room before its IR cut in (`dark`), and **chroma
0.0 / luma ~63 on its own IR** with the room lights off — pure black-and-white,
well clear of `mono_max_chroma: 1.5`. Outdoors, with a pond and a far bank,
expect a lower luma.

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
(A missing file is only one way to be silent — the wrong output device is
another, and that one starts fine. See "Sound output".)

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
  looping forever in the journal. Verified on the Pi: after `kill -9` systemd
  restarted it (a fresh startup in the app log, before the next reboot); after
  each of two reboots it was running 3–6 s after boot.

**The system journal does not survive a reboot on the Pi.** Raspberry Pi OS
ships `/usr/lib/systemd/journald.conf.d/40-rpi-volatile-storage.conf`
(`Storage=volatile`) to spare the SD card, so `journalctl -b -1` finds nothing.
The app's own `heron_deterrent.log` in the repo directory *does* survive — read
that for what happened before a crash-and-reboot.

---

## Web Dashboard

At `http://<pi-ip>:5000` (or `http://heron-pi.local:5000` from a Mac or iPhone).

Space follows content. **Triage owns the page**: it is the work, and at a pond
there will be hundreds of clips to look at and, mostly, no herons. A Herons
panel appears above it only when there are herons; otherwise that is one line.
Everything you glance at sits in a narrow side column.

| Section | Description |
|---|---|
| Mode pill | Day / IR night / Dark night / establishing / switching; model or motion-only; deterrent armed or silent. Hover for live chroma/luma |
| Stats strip | Herons today, last heron, **to triage**, unclassified motion, model detections. Motion is never counted as a detection |
| Herons | Only when there are any: history with confidence, and the clips (thumbnail as poster). Never filtered |
| To Triage | A contact sheet — see below |
| Live Feed | MJPEG stream, throttled to 2 fps so it cannot starve detection. Test Speaker plays through the running service |
| Recent Alerts | Identical consecutive alerts are one row with a count and time span; expand for each one's thumbnail and whether it has been labelled. A 0.00 confidence is never shown |
| Tools | Collapsed: test video upload — the quickest end-to-end proof |

### Triage: a contact sheet

Every triage row used to read "unknown object" — the signal was in an image you
had to open a dialog to see. Now each clip is a thumbnail, grouped by day and
image domain ("Today · IR night"), showing the whole frame (never cropped — a
heron at the edge must stay visible), its time, frame-type badge, and a
confidence only where a model produced one. A filter above it narrows to Day /
IR / Dark / Unsettled, with counts.

**Bulk "no heron" is the main way training data gets made.** The model has
never seen a single negative example — which is why non-heron frames score as
high as 0.894. Select clips (click; shift-click a range; a group's checkbox;
`A` for all shown) and press `N`:

- Each clip's **thumbnail frame — frame 1, the only one you looked at** — is
  saved as a background example (an empty annotation): *no heron in this
  frame*. Not the other frames of the clip, which you have not seen.
- Each clip goes to **its own** domain's set, `labeled/day/`, `ir/` or `dark/`,
  by its frame type recorded at capture — through exactly the same code path
  as a single label. **A bulk action never chooses a domain**: a clip caught
  mid-switch, with no capture-time record, or whose frame 1 is a transition
  frame stays in triage, still selected, for you to deal with one at a time.
- A confirmation shows the split first: "2 → `labeled/day/`, 5 →
  `labeled/ir/`, 1 cannot be filed in bulk".
- **Undo** (`U`) deletes exactly the files the last batch wrote and puts its
  clips back — sparing any clip you have relabelled by hand since. It
  survives a service restart.
- "Heron" is never a bulk label: a heron needs a box drawn by hand.

Keys on the sheet: arrows move, `Space` selects, `Enter` opens the labeller,
`A` all, `N` no heron, `U` undo, `Esc` clears. While you have a selection, new
clips arriving are held behind an "N new" pill rather than moving tiles under
the cursor.

### The labeller

Double-click a tile (or `Enter`). The frame fills the screen at its own aspect
ratio — on a 1568 px window, 1136×852 for the camera's 640×480 — with the label
choices beside it. Draw with a drag (mouse, finger or pen); drawing a box
selects "heron". Keys: `←`/`→` frame (`Shift` ×5, `Home`/`End`); `H` heron,
`N` no heron, `B` `C` `D` `P` other bird / cat / dog / person, `U` can't tell;
`Enter` save and open the next clip; `Shift+Enter` save and close; `]`/`[`
next/previous clip without saving; `Ctrl/Cmd+Z` undo a box, `Delete` clear;
`Esc` close.

**The coordinate guard.** The frame and the drawing layer over it are sized
together, in whole pixels, by the page's script — not by CSS — because box
coordinates are scaled by exactly that size. A stylesheet change that moved one
without the other (a padding, a border, `object-fit`, a transform) would not
cause an error; it would quietly put every box in the wrong place in the
training data. So before a heron is saved the page checks that the drawing
layer sits on the image's pixels within 1 px, and **refuses to save** if not,
naming what is wrong. Each of those five was injected in a real browser and
refused. The arithmetic itself is pinned by `tests/test_label_coords.py`,
which runs the page's own JavaScript in Node (on a machine that has Node — not
the Pi, where those tests skip).

### What the dashboard costs the Pi

Measured on the build Pi with the service running, 38–42 clips in triage, the
browser fetching six thumbnails at a time, and an inference probe running the
real model once every 0.6 s alongside (the A/B runs alternated, from the same
board temperature):

| | Wall | Pi CPU | Data |
|---|---|---|---|
| Page HTML, once | — | — | 101 KB |
| Triage list (`/api/unclassified`), every 5 s | — | — | 25 KB |
| Contact sheet, **first view** (thumbnails generated) | 2.2–2.4 s | **1.4 s** | ~200 KB, 4.8 KB each |
| Contact sheet, **every view after** (from disk) | 0.8 s | 0.4 s | ~200 KB |

| Inference while a first view loads | Median |
|---|---|
| Before | 381–390 ms |
| During — thumbnails merely one at a time (first version) | **666–673 ms (+70–77%)** |
| During — one decoder thread, lowest priority (shipped) | **545–572 ms (+41–49%)** |
| After | 382–392 ms |

**Thumbnails are now written when the clip is saved**, from frame 0 while it
is still in memory: a resize and a JPEG encode, no video decode. Measured on
the Pi over 60 real frames: median 1.7 ms, p95 4.0 ms, max 9.9 ms, on the
detection thread *after* the clip is safely on disk (encoding the clip itself
takes far longer). So for any clip recorded by this version, a first view costs
what "every view after" costs above.

The decode path in the table is now only a fallback — clips recorded before
this change, or a save-time write that failed (it logs a warning and never
stops detection). That fallback still slows inference while it runs: serialising
generation was not enough, because the video decoder starts a thread per core
and the model's four threads wait for one another, so losing part of one core
stalls all four. One decoder thread at the lowest scheduling priority roughly
halves the effect. Once the older clips have been viewed, or have aged out,
nothing decodes for thumbnails any more.

> Every inconclusive motion event creates a WARNING "unknown" alert and saves a
> clip. That is deliberate — it feeds triage — and the grouping keeps the
> alert feed readable. Raise `motion.sensitivity` if there are too many.

The Flask server is the development server. That is fine on a private LAN; do
not expose it to the internet. It listens on every interface, so if the Pi has
both Ethernet and Wi-Fi connected it answers on both addresses.

### API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/live-stream` | MJPEG video stream (2 fps) |
| GET | `/api/system-status` | System health, frame state and mode JSON |
| GET | `/api/stats` | Counts for the stats strip (motion reported separately) |
| GET | `/api/alerts` | Recent alerts, each with its detection's `det_label`, `det_confidence`, `det_frame_type`, `det_user_label` (param: `limit`) |
| GET | `/api/detections` | Detection history (params: `label`, `limit`, `frame_type` = day/ir/dark/unknown) |
| GET | `/api/heron-detections` | Heron-only detections (params: `limit`, `frame_type`) |
| GET | `/api/unclassified` | Unlabelled "unknown" and "motion" detections |
| GET | `/api/detection/<id>/frame/<i>` | One frame at native resolution — what the labeller draws on |
| GET | `/api/detection/<id>/thumb` | 320-wide JPEG of a frame (param: `frame`, default 0) for the contact sheet. Cached in `data/thumbs/`; `X-Thumb: hit\|miss`. Display only — never used for coordinates |
| GET | `/api/detection/<id>/frame/<i>/domain` | Which training set frame `i` would be filed under (param: `requested`) |
| POST | `/api/label` | Submit a manual label `{detection_id, label, bboxes, frame_index, domain?}`. Labels: `heron` (needs boxes), `background` (no heron), `bird`, `cat`, `dog`, `person`, `unknown`. 409 if the frame cannot be placed in a domain |
| POST | `/api/label/bulk` | `{detection_ids, label: "background", preview?}` — frame 0 of each clip, each to its own domain; returns `filed` per domain, `left` (with reasons) and a `batch_id`. `preview` writes nothing |
| POST | `/api/label/bulk/undo` | `{batch_id}` of the latest batch: delete exactly its files, return its clips to triage |
| POST | `/api/trigger-deterrent` | Manual speaker test. Plays regardless of mode |
| POST | `/api/upload-test-video` | Run the alert vote on an uploaded video, decoded as the camera would see it (≤ 640 px, the camera's frame rate, first 120 s). Returns every key frame's score, the votes, and how many key frames clear the deterrent threshold — uploads never sound it. Refused unless every key frame is a day frame |
| GET | `/api/media/<filename>` | Serve a stored clip |

`/api/trigger-deterrent` is the best speaker test there is: it plays through the
running service itself — its user, its sound device, its volume. From the Pi:
`curl -s -X POST http://localhost:5000/api/trigger-deterrent`.

> `POST /api/config` exists but is **not safe to call**: it rewrites
> `config.yaml` from the posted JSON, and since the GET strips the `alert`
> section, a round trip deletes it along with every comment. The dashboard never
> calls it. Edit `config.yaml` directly instead.

---

## Installation

Step by step, every command run on the build Pi, is in
**[DEPLOY.md](DEPLOY.md)** (and [docs/raspberry_pi4_setup.md](docs/raspberry_pi4_setup.md)
for flashing the card). The short version, on the Pi:

```bash
git clone https://github.com/mbookham7/heron_deterrent.git
cd heron_deterrent
./setup.sh                    # apt packages, sound to the 3.5 mm jack, venv, LiteRT

# Models are gitignored — copy from your laptop:
#   scp models/heron_cpu_int8.tflite pi@<pi-ip>:~/heron_deterrent/models/

venv/bin/python preflight.py  # must pass before you start
START_WEB_UI=true venv/bin/python main.py
```

### Python environment notes

**Raspberry Pi OS 13 (Trixie)** is Debian 13 with **Python 3.13**. Use a plain
venv — no `--system-site-packages`:

- **TFLite runtime: `ai-edge-litert`, not `tflite-runtime`.** `pip install
  tflite-runtime` finds no versions at all for Python 3.13 (neither PyPI nor
  piwheels has one). `ai-edge-litert` is its successor — TensorFlow Lite was
  renamed LiteRT — and 2.4.0 installs cleanly and runs the model.
  `ai/model_loader.py` tries `tflite_runtime` first and `ai_edge_litert`
  second.
- **No BLAS package.** numpy's wheel bundles its own OpenBLAS. In particular,
  `libatlas-base-dev` was removed in Debian 13 and no longer exists — do not
  hunt for an old `.deb`. `libopenblas-dev` does exist but is not needed.
- **numpy 2 and OpenCV 5.** `requirements.txt` has floors, not ceilings, so pip
  resolves current wheels: on the build that was **numpy 2.5.5 and OpenCV
  5.1** — both major versions newer than this code was written against. 11 of
  the 12 test files passed on them; the twelfth (`test_camera_sim.py`) needs
  `pytest`, which is not installed on the board.
- **`paho-mqtt` is pinned below 2.0**: version 2.x requires a
  `CallbackAPIVersion` argument that `alerts/mqtt_client.py` does not pass, and
  the resulting `ValueError` gets swallowed, silently disabling MQTT.

The legacy Coral's environment is completely different (Python 3.7, a venv that
*must* see system packages, and a pip package that segfaults) — see
[docs/legacy/coral/](docs/legacy/coral/README.md).

---

## Project Structure

```
heron_deterrent/
├── main.py                       # Orchestrator: pipeline, health, cleanup
├── preflight.py                  # Pre-start checks — run this at the pond
├── config.yaml                   # All configuration
├── requirements.txt              # Python dependencies (not the TFLite runtime)
├── heron-deterrent.service       # Systemd unit (runs as pi)
├── setup.sh                      # Automated setup
│
├── ai/
│   ├── detector.py               # YOLOv8 TFLite inference + NMS
│   ├── letterbox.py              # The preprocessing convention. Do not bypass
│   └── model_loader.py           # Target → model file → backend
│
├── tests/                        # pytest, or run most files directly on the board
│
├── capture/
│   ├── camera.py                 # Locked, self-healing camera wrapper
│   ├── frame_classifier.py       # day / IR / dark, with hysteresis
│   └── motion_detector.py        # MOG2 background subtraction
│
├── deterrent/
│   └── audio_player.py           # ALSA playback; fails loudly if no sounds
│
├── storage/
│   ├── database.py               # SQLite (thread-safe)
│   ├── media_store.py            # Clips, retention, two size-capped pools, labeled/
│   └── thumbnails.py             # Triage thumbnails: cached, one at a time, low priority
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
├── tools/                        # Threshold measurement, camera simulation — dev only
├── models/                       # *.tflite — gitignored, copy in by hand
├── sounds/                       # *.wav — at least one required
├── media/                        # Auto-created: heron/ other/ unknown/ motion/ (clips)
│                                 #   labeled/{day,ir,dark}/{images,labels}/ (training data)
├── data/                         # Auto-created: heron.db, thumbs/ (triage thumbnails),
│                                 #   last_bulk_label.json (what Undo removes)
└── docs/
    ├── raspberry_pi4_setup.md    # Flashing and preparing the Pi
    └── legacy/coral/             # Coral Dev Board + Edge TPU — end-of-life
```

---

## Training and exporting a model

The dataset and training pipeline live in sibling repositories
(`heron-data`, `heron-detector`). Current weights:
`heron-detector/runs/detect/heron_v8_model-4/weights/best.pt` (May 2026).

### Export for CPU

```bash
yolo export model=best.pt format=tflite int8=True imgsz=640 data=your_data.yaml
# produces best_saved_model/best_int8.tflite
cp best_saved_model/best_int8.tflite models/heron_cpu_int8.tflite
```

`int8=True` gives int8 weights with float32 input and output. That is what the
CPU target expects, and it is ~3.3 MB versus 12 MB for float32.

The 640 input is deliberate. A 320 export ran 4× faster on the Coral but lost
recall, worst on distant birds; at 314 ms on the Pi, 640 is affordable. Details
in [docs/legacy/coral/](docs/legacy/coral/README.md).

### Collecting more training data

The model was trained on ~180 close-up heron images with **no negative
examples**, which is why non-heron frames score as high as **0.894** on real
camera footage (see "Accuracy"). The most valuable thing you can add is
negatives from your own pond: water ripple, ducks, foliage, moving shade. Select
them on the dashboard's triage sheet and mark them **no heron** in one go (see
"Triage: a contact sheet"), and they are written in YOLO format to
`media/labeled/day/`, `ir/` or `dark/` — one tree per image domain. **Point a
training run at exactly one of them.** The pre-split
`media/labeled/images|labels` layout is no longer written; preflight warns if
anything is still there.

Mounting distance matters here too: the training herons are big in the frame
(median 0.59 of its height). Footage of distant birds is exactly what the
dataset lacks — see "Mounting distance".

---

## Edge TPU status

**Not usable today, and not relevant to the Pi.** The compiled Edge TPU model
detects nothing — quantisation crushed its class scores — and on the Pi the
Edge TPU path stops at `libedgetpu.so.1: cannot open shared object file`: no
Edge TPU library has been found that works there.
Two known workarounds for the model are **untried, not unavailable**. The full
diagnosis, and why a Coral USB Accelerator on this Pi would not work as things
stand, is in [docs/legacy/coral/README.md](docs/legacy/coral/README.md).

---

## Troubleshooting

Run `venv/bin/python preflight.py` first — it diagnoses most of this and prints
the fix.

### No detections at all

1. What mode is it in? The dashboard's mode pill, or `preflight`. At night,
   or while `establishing`/`switching`, no model runs and nothing is classified
   as a heron by design — look in To Triage for "motion" clips instead
2. Lower `motion.sensitivity` (try `2000`)
3. Lower `modes.day.deter_threshold` toward `0.75` (accepting more false positives, since
   at 0.75 recall on stock positives is 0.84 vs 0.41 at the shipped 0.90 — but 18%
   of your bench footage also clears that bar, so the deterrent will fire more)
4. Is the heron too far away? See "Mounting distance"
5. Upload a clip via the dashboard's Test Upload to confirm inference works. It runs the **alert** vote (k of m key frames at `alert_threshold`), which is stricter than the deterrent's single frame at `deter_threshold`: "No alert" with key frames at 0.92–0.94 means the model saw the heron and the alert threshold was not reached, not that it was missed
6. `sudo journalctl -u heron-deterrent -f`

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
so a fresh clone has none (and no `models/` directory either, until `setup.sh`
creates it). Otherwise, check the runtime is in the venv:

```bash
venv/bin/python -c "from ai_edge_litert.interpreter import Interpreter; print('OK')"
venv/bin/pip install ai-edge-litert      # if not
```

### Camera not detected

```bash
v4l2-ctl --list-devices
ls /dev/video*                    # the Pi has many; the USB camera is normally video0
sudo fuser -v /dev/video0         # something else holding it?
```

Only one process can hold the camera; while the service runs, `preflight` skips
its camera check rather than fail. Repeated reconnects — especially only at
night — usually mean USB power, not a bad camera. Use a powered hub.

### Night image washed out

The dome IR reflection — see "The dome: IR washout". Check the foam gasket is
seated flush against the dome.

### No sound

`preflight` fails if sound is going to HDMI, but its PASS cannot tell you the
speaker is powered and turned up — see "Sound output".

```bash
aplay -l                                          # the jack is the "Headphones" card
cat /etc/asound.conf                              # should name Headphones
aplay -q sounds/startle_default.wav & sleep 0.4; head -1 /proc/asound/card*/pcm0p/sub0/status
amixer sget PCM                                   # the jack's volume control
```

If the jack shows `RUNNING` and you hear nothing, it is the speaker: power, its
own volume, its lead. Headphones in the jack are the quickest way to split "Pi"
from "speaker".

### Disk filling

```bash
df -h /
du -sh media/* | sort -h | tail
```

Lower `storage.max_media_gb` and `storage.max_motion_gb`; `preflight` fails if
together they cannot fit on the filesystem. The caps are enforced every five
minutes. If `motion/` is the one growing, lengthen
`modes.ir.cooldown_seconds` — moths in front of IR LEDs are the usual cause.

---

## License

Open source — free for personal and commercial use.

## Contributing

Pull requests and issues are welcome.
