# DEPLOY — pond checklist

Ordered steps from a flashed Raspberry Pi to a running service. Tick straight
down. Rollback is at the bottom.

Flashing the card comes first: [docs/raspberry_pi4_setup.md](docs/raspberry_pi4_setup.md).
For *why* any of this is the way it is, see [README.md](README.md).

**Every command here ran on the build Pi** (Pi 4 Model B 8 GB, Raspberry Pi OS 13
"Trixie", Python 3.13.5, user `pi`) on 2026-11-25, except where a step says it
was not. That build was on the bench — not in the box, not at the pond, not at
night.

**Reference values for this build:** `ai.target: cpu`,
`models/heron_cpu_int8.tflite`, `deter_threshold: 0.90` / `alert_threshold: 0.95`,
service `heron-deterrent`, user `pi`, SSH key `~/.ssh/heron_pi`.

---

## 0. Before you leave the house

- [ ] Pi 4, **powered** speaker (3.5 mm lead), ELP camera, **powered USB hub**
      for the camera, SD card, official 5 V / 3 A supply, cables
- [ ] The sealed box: standoffs, heatsinks on the SoC, **desiccant** — not the
      official Pi case (README, "The filter-house enclosure")
- [ ] Laptop can SSH to the Pi with a key (step 1) — sort this out at home, not
      at the pond
- [ ] `models/heron_cpu_int8.tflite` on your laptop — **it is gitignored, a clone
      does not include it**
- [ ] A real `.wav` if you want better than the built-in startle burst
- [ ] Mounting: camera **1.5–4.5 m from where the heron stands, nearer 1.5 m**,
      speaker aimed at the landing spot. Distant birds are missed — README,
      "Mounting distance", has the arithmetic
- [ ] Camera dome: check the foam gasket inside sits flush against the dome
      (README, "The dome: IR washout")

---

## 1. Get a shell on the Pi

```bash
ssh -i ~/.ssh/heron_pi -o IdentitiesOnly=yes pi@<pi-ip>
```

If the key is not on the Pi yet (you set a password in Imager), install it once
**from a normal terminal** — it prompts for the Pi's password:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/heron_pi -N "" -C "heron-pi-deploy"   # only if you have no key
ssh-copy-id -i ~/.ssh/heron_pi.pub pi@<pi-ip>                          # want: Number of key(s) added: 1
```

It has to be a real terminal. Run from a tool with no terminal attached (such
as the `!` prefix in Claude Code), the first-contact host key question cannot be
answered and SSH fails with `Host key verification failed` — on a host it has
never seen, not one that changed. The password prompt fails the same way.

**The Pi is on DHCP, so the address can move.** If it stops answering, find it
again (`ping`, or the router's device list) before assuming a fault. When it
does move, SSH refuses the new address with `Host key verification failed` —
expected. Confirm it is the same Pi rather than blindly clearing the entry: the
host key should be *identical* to the one already trusted.

```bash
ssh-keyscan -t ed25519 <new-ip> | ssh-keygen -lf -      # compare the fingerprint
```

A **re-flashed** Pi has a new host key, so a mismatch after re-flashing is
expected too.

**Two addresses.** With Ethernet and Wi-Fi both connected, the Pi has one on
each (the build had `eth0` and `wlan0`), and SSH and the dashboard answer on
both. `ip -4 -br addr` lists them. Note the Wi-Fi one — that is what the filter
house will have.

**`sudo` asks for a password** on a Pi set up with Imager. Steps marked
**(sudo)** need you at a real terminal for that reason.

---

## 2. Install (sudo)

```bash
git clone https://github.com/mbookham7/heron_deterrent.git
cd heron_deterrent
./setup.sh
```

`setup.sh`:

- installs `python3-venv`, `python3-pip`, `git`, `ffmpeg`, `alsa-utils`,
  `v4l-utils`, `sqlite3`. On the build image all but `sqlite3` were already
  there
- writes **`/etc/asound.conf`** so sound goes to the 3.5 mm jack — without it
  the deterrent plays to HDMI and is silent (README, "Sound output")
- creates a plain venv and installs `requirements.txt` (about 40 s, all
  prebuilt wheels) plus **`ai-edge-litert`**, the TFLite runtime
- creates `models/`, `sounds/`, `media/`, `data/`
- warns if the Pi boots to the desktop

Two things it deliberately does **not** do, because they fail on Trixie:

- `pip install tflite-runtime` — there is no version for Python 3.13
  (`from versions: none`)
- `apt-get install libatlas-base-dev` — removed in Debian 13
  (`Candidate: (none)`). numpy brings its own BLAS; nothing replaces it

If the Pi boots to the desktop, switch it to the console now — the desktop is
heat in a sealed box and nothing needs it:

```bash
sudo systemctl set-default multi-user.target && sudo reboot
```

---

## 3. Copy the model and sounds across

From your laptop:

```bash
scp -i ~/.ssh/heron_pi models/heron_cpu_int8.tflite pi@<pi-ip>:~/heron_deterrent/models/
scp -i ~/.ssh/heron_pi my_heron_call.wav            pi@<pi-ip>:~/heron_deterrent/sounds/
```

If `scp` says `dest open "heron_deterrent/models/": Failure`, the directory does
not exist — `setup.sh` has not run. `models/` is gitignored, so a fresh clone
has none.

Check the copy arrived intact — the two hashes must match:

```bash
shasum -a 256 models/heron_cpu_int8.tflite                       # laptop
ssh -i ~/.ssh/heron_pi pi@<pi-ip> 'sha256sum heron_deterrent/models/heron_cpu_int8.tflite'
```

The repo already ships `sounds/startle_default.wav`, so the sounds copy is
optional — but the model copy is not. Without it, startup fails.

---

## 4. Check `config.yaml`

```bash
nano config.yaml
```

- [ ] `ai.target: cpu`
- [ ] `system.camera_device_id: 0` — `v4l2-ctl --list-devices` shows which
      `/dev/video*` belongs to the camera. The Pi has a dozen `/dev/video*`
      nodes of its own (the codec and ISP); the USB camera was `video0` on the
      build
- [ ] `system.resolution` 640×480 — leave it. It fixes the image domain of every
      clip you collect (README, "Capture resolution")
- [ ] No `system.active_hours` — it is gone; the system runs round the clock and
      branches on frame type. A leftover block is a startup error.
- [ ] `modes.<mode>.deterrent` — the **only** switch for the sound, and `false`
      in every mode as shipped. Turning the sound on is `modes.day.deterrent:
      true`; a motion-only mode (`model: null`) with `deterrent: true` refuses
      to start
- [ ] `audio.volume` — 80 as shipped, which is −17.3 dB on the jack, not "80%
      loud" (README, "Sound output")
- [ ] `storage.max_media_gb` + `storage.max_motion_gb` (15 + 5 GB) — need a
      32 GB card or larger. Preflight fails if they cannot fit
- [ ] `alert.sms_enabled: false` unless you have done step 5
- [ ] `collect_mode.enabled` — `false` normally. Set `true` only for a
      data-collection run, and read the storage warning in step 10 first: it
      records every day event above `modes.day.record_threshold` (0.30), which
      on a bench is near-continuous. It does **not** affect the deterrent.

---

## 5. SMS — skip if you have no Twilio account (sudo)

Not exercised on the Pi — there is no Twilio account yet.

```bash
sudo install -m 600 -o root -g root /dev/null /etc/heron-deterrent.env
sudo nano /etc/heron-deterrent.env
```

```
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_FROM_NUMBER=+441234567890
TWILIO_TO_NUMBER=+441234567890
```

Then set `alert.sms_enabled: true` in `config.yaml`.

Never put credentials in `heron-deterrent.service` — that file is in git.

---

## 6. Preflight — do not skip this

```bash
cd ~/heron_deterrent
venv/bin/python preflight.py          # NOT ./preflight.py — see below
```

**Always run it through the venv.** `./preflight.py` runs the system Python,
which has none of the venv's packages. On the Pi it gets as far as the config
and then reports, on a perfectly healthy install:

```
[FAIL] model        heron_cpu_int8.tflite will not load on target 'cpu'
[FAIL] camera       opencv not installed
```

**Listen for the speaker.** The audio check plays a real WAV, watches which
output plays it, and **FAILs if it went to HDMI** — the line should name
`bcm2835 Headphones`, the 3.5 mm jack. But a PASS only proves the sound left
the jack: whether the speaker is powered, turned up and connected, only your
ears can tell. To see which output played by hand:

```bash
aplay -q sounds/startle_default.wav & sleep 0.4
head -1 /proc/asound/card*/pcm0p/sub0/status      # want the Headphones card (card2 on the build) RUNNING
```

If the jack is `RUNNING` and you hear nothing, it is the speaker side: is it
powered, is its own volume up, is the lead good? Headphones plugged into the
jack split "Pi" from "speaker" in seconds — that is how the build found its
silent speaker.

Running remotely, where nobody can hear? Use `--skip-audio` and treat the
speaker as unverified.

Every failure prints the command that fixes it. Do not continue until you see:

```
All 13 checks passed.
```

(**11** if the service is already running — see below. The number is the count
of checks that ran, so it varies; what matters is no FAILED.)

Two things preflight cannot see when run by hand:

- **The camera, if the service is already running** — only one process can hold
  `/dev/video0`, so the check reports `[skip] camera ... held by the running
  heron-deterrent service`. That is not a fault. For a full camera test:
  `sudo systemctl stop heron-deterrent && venv/bin/python preflight.py`
- **`/etc/heron-deterrent.env`** — only the service reads it. `export` the
  variables in your shell if you want to verify SMS manually.

---

## 7. Run it by hand once

```bash
START_WEB_UI=true venv/bin/python main.py
```

On the build this step was skipped and the service (step 8) ran the identical
command; the lines below are from that service's log on the Pi. Expect, within
a couple of seconds:

```
Video encoder: /usr/bin/ffmpeg with libx264 (preset veryfast)
Mode day: model=heron deter=0.9 alert=0.95 record=0.3, silent, cooldown 10s
Mode ir: motion-only (no model), silent, cooldown 45s
Mode dark: motion-only (no model), silent, cooldown 45s
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Model loaded: heron_cpu_int8.tflite | target=cpu | backend=CPU (4 threads) | input=1x640x640x3 float32 | output=1x5x8400
Detector ready: input 640x640 float32 | classes=1 | confidence>=0.9 | iou=0.45
day: vote=3/5 anchor_floor=0.9
Loaded 1 deterrent sound(s): startle_default.wav
Camera 0 open at 640x480
System started successfully
Motion detected (establishing)                          <- motion before the frame type settles is still recorded
Frame type: startup -> day                              <- ~2 s later (up to ~26 s after a cold boot); ir / dark at night
Mode is now day: model=heron ...
```

The `XNNPACK delegate` line is LiteRT announcing its CPU backend — normal, not
an error.

With `collect_mode.enabled: true` the detector's floor drops — this is
correct, not a fault:

```
Detector ready: ... | confidence>=0.3 | iou=0.45          <- anchor floor drops to record_threshold
day: vote=3/5 anchor_floor=0.3
COLLECT MODE ON: recording day events above 0.3
```

- [ ] Open `http://<pi-ip>:5000` — dashboard loads, live feed moves, the
      mode pill shows Day (or IR night / Dark night)
- [ ] Walk in front of the camera → the log shows `Motion detected (day)`.
      At night it shows `Motion detected (ir)` then
      `Motion (motion-only mode, raw=ir) — recording unclassified`, and no
      inference runs. Seen on the Pi with the room lights off — the quickest
      night test there is, short of waiting for dark
- [ ] Press **Test Speaker** on the dashboard → you hear the sound. This is a
      manual test and plays in every mode. Automatic firing is governed only
      by `modes.<mode>.deterrent`; when that is `false` a confident heron logs
      `Deterrent SILENCED (modes.day.deterrent=false)` and is still recorded
      and alerted
- [ ] Optional, strongest test: upload a heron clip via **Test Upload**

**Two traps in the Test Upload check.** It is a weaker test than it looks:

1. **It only accepts video** — `.mp4/.mov/.avi/.mkv`. A heron *photo* is
   rejected with `Unsupported file type`. Wrap stills into a short clip first.
2. **`herons_detected` is not scoped to your upload.** The endpoint returns the
   heron rows among the last 10 detections in the database, so it can report
   `herons_detected: 1` from an *earlier* detection while the clip you just
   uploaded was actually filed as `unknown`. Confirm against the database
   before believing it:
   `sqlite3 data/heron.db "select id,timestamp,label,confidence from detections order by id desc limit 5"`

Also note Test Upload runs the **alert** path (`alert_threshold`, k-of-m
voting), never the deter path — its docstring says so explicitly. It therefore
cannot validate `deter_threshold`. Since only 1 of the 44 stock positives
clears 0.95, a genuine heron clip will usually come back `unknown` here even
though the model is working perfectly.

(Both traps were found on the Coral; Test Upload has not been run on the Pi.)

`Ctrl-C` when satisfied.

---

## 8. Install as a service (sudo)

The unit runs as `pi` from `/home/pi/heron_deterrent`. If you chose a different
username in Imager, edit `User`, `Group` and both paths in
`heron-deterrent.service` first.

From your laptop, in one go — it asks for the Pi's password once:

```bash
ssh -t -i ~/.ssh/heron_pi pi@<pi-ip> 'cd ~/heron_deterrent && sudo cp heron-deterrent.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now heron-deterrent && sleep 10 && systemctl is-active heron-deterrent'
```

On the build this printed:

```
Created symlink '/etc/systemd/system/multi-user.target.wants/heron-deterrent.service' → '/etc/systemd/system/heron-deterrent.service'.
active
```

- [ ] `active`
- [ ] `journalctl -u heron-deterrent -f` shows the startup lines from step 7
      (`pi` can read the journal without sudo — it is in the `adm` group)
- [ ] Dashboard still responds on `:5000`
- [ ] **Speaker, through the service itself.** This plays via the running
      service — its user, its sound device, its volume — so it is the test that
      counts:

```bash
curl -s -X POST http://localhost:5000/api/trigger-deterrent     # {"success":true}, and you hear it
```

**Do not trust "active" without looking.** On the build, an earlier report of
`active` turned out to have come from somewhere other than the Pi — the unit
file was not even installed. `systemctl status heron-deterrent` on the Pi
itself is the check.

---

## 9. Prove it survives (sudo)

```bash
# Restart on crash
ssh -t -i ~/.ssh/heron_pi pi@<pi-ip> 'OLD=$(systemctl show -p MainPID --value heron-deterrent); sudo kill -9 $OLD; sleep 15; echo "old pid $OLD -> new pid $(systemctl show -p MainPID --value heron-deterrent)"; systemctl is-active heron-deterrent'

# Restart on reboot
ssh -t -i ~/.ssh/heron_pi pi@<pi-ip> 'sudo reboot'
# wait, then:
ssh -i ~/.ssh/heron_pi pi@<pi-ip> 'uptime -s; systemctl is-active heron-deterrent'
```

**Check `uptime -s` moved** before believing a reboot test — a quick reconnect
can reach the board before it has actually gone down. On the build, the
service was `active` 3–6 s after boot.

The journal **does not survive a reboot** on the Pi (`Storage=volatile`), so
`journalctl -b -1` finds nothing. To confirm the `kill -9` restart after the
fact, read the app's own log, which does survive:

```bash
grep -E "Initializing Heron|System started successfully" ~/heron_deterrent/heron_deterrent.log | tail
```

- [ ] Survives `kill -9` (systemd waits 10 s, then restarts)
- [ ] Survives a reboot
- [ ] Dashboard reachable after reboot
- [ ] `/etc/asound.conf` still there, and the speaker still plays (step 8's
      `curl`). The per-user `~/.asoundrc` did **not** survive the build's first
      reboot — that is why setup writes the system-wide file

If it fails 5 times in 2 minutes systemd gives up deliberately, so the fault is
visible in `systemctl status` instead of scrolling past in the journal.

---

## 10. Leave it running — tuning notes

Come back after a day and check:

```bash
journalctl -u heron-deterrent --since today | grep -c 'HERON DETECTED'
journalctl -u heron-deterrent --since today | grep -c 'Motion detected'
df -h /
du -sh media/*
vcgencmd measure_temp; vcgencmd get_throttled     # in the box now — this is the reading that matters
tail -20 ~/heron_temps.csv                        # the history, if the temperature logger is installed
```

The temperature logger is a five-minute cron job for `pi` — setup in README,
"The filter-house enclosure". Install it **before** sealing the box, so the
first hours in there are on record.

| Symptom | Fix |
|---|---|
| Deterrent firing at nothing | Check `media/heron/` first — at `modes.day.deter_threshold=0.90` no bench frame clears the bar, so if it is firing something is genuinely triggering. If it truly is nothing, raise it further. |
| Lots of motion, no detections | Lower `motion.sensitivity` to `2000` |
| Hundreds of "unknown" alerts | Raise `motion.sensitivity`; wind and shadow trigger it |
| Herons ignored | Check the mode pill first — at night no model runs by design. By day, lower `modes.day.deter_threshold` toward `0.75` (raises recall on stock positives 0.41 → 0.84, but lets 18% of bench frames through); move the camera closer |
| Stuck in establishing / switching | Frames are ambiguous. Run `venv/bin/python preflight.py` and compare its chroma/luma against `frame_type:` thresholds |
| Hundreds of night "motion" clips | Moths at the IR LEDs or rain on the housing. Raise `modes.ir.cooldown_seconds`, or `motion.sensitivity` |
| Night clips milky / washed out | The dome gasket — README, "The dome: IR washout" |
| Camera reconnects, only at night | USB power: put the camera on a powered hub |
| CPU temp WARN in the box | More heatsink, more air; README, "The filter-house enclosure" |
| Card filling | Lower `storage.max_media_gb` / `storage.max_motion_gb` |

Apply changes with `nano config.yaml && sudo systemctl restart heron-deterrent`.

### Collect mode fills the card, and the safety net eats your data

`storage.max_media_gb` deletes **oldest first**. On a collect run the oldest
clips are exactly the footage you went out to gather, so the mechanism that
protects the card is also the one that quietly discards your dataset. It runs
every 5 minutes (`CLEANUP_INTERVAL_SECONDS`) and logs `Size cap: deleted N
oldest file(s)` when it bites — that line is the only warning you get.

Measured on the Coral bench (640x480, 5 fps, 5 s clips, `record_threshold` 0.30;
not re-measured on the Pi, whose bench clips averaged 170 KB):

| | clips/hour | growth | time to the 1.0 GB cap |
|---|---|---|---|
| Observed, ordinary room activity | ~80 | ~12 MB/h | **~3.8 days** |
| Saturated, continuous motion (10 s cooldown) | ~225 | ~33 MB/h | **~32 hours** |

Mean clip is ~147 KB. **Offload before that window closes.** (Those times are
against the Coral's 1 GB cap; at the Pi's 15 GB they scale up fifteen-fold, but
the principle — the cap eats the oldest data first — does not change.)

#### Round the clock

Running 24 hours adds the night, and the night is where volume becomes
unpredictable: with no model to filter it, every moth in front of the IR LEDs
and every raindrop on the housing is a clip. Daylight clips averaged ~147 KB on
the Coral and ~170 KB on the Pi bench; indoor IR clips on the Pi ran 18–278 KB
(278 KB with a person walking through) — assume ~200 KB until there is pond
footage.
Worst cases, from the cooldowns:

| Pool (cap) | Worst case | Fills its cap in |
|---|---|---|
| motion (5 GB), 45 s night cooldown, 15 h November night | 1,200 clips ≈ 240 MB/night | ~3 weeks |
| classified (15 GB), 10 s day cooldown, 9 h saturated | 3,240 clips ≈ 650 MB/day | ~3 weeks |
| classified, ordinary activity (~80 clips/h, 9 h) | ≈ 145 MB/day | ~3 months |

The caps never let the card fill; what they cost you is the **oldest footage in
that pool**. The two pools are separate, so a moth night only ever evicts older
night clips — never daylight. Offload **weekly** and nothing you want can reach
a cap even at the worst case; monthly is fine on an ordinary pond. The
`Size cap: deleted N oldest file(s) ... from the motion pool` log line is the
warning that you left it too long. From your laptop:

```bash
mkdir -p ~/heron_pull_pi
rsync -a --stats \
  -e "ssh -i ~/.ssh/heron_pi -o IdentitiesOnly=yes" \
  pi@<pi-ip>:/home/pi/heron_deterrent/media/ \
  ~/heron_pull_pi/
```

(Run on the build: 16 clips, 2.8 MB.)

To pull *and* free the card in one step, so the cap can never reach your
footage, add `--remove-source-files` (then prune the emptied directories). This
variant has **not** been run against the Pi — it deletes from the card, so try
the plain pull first and check what arrived:

```bash
rsync -a --stats --remove-source-files \
  -e "ssh -i ~/.ssh/heron_pi -o IdentitiesOnly=yes" \
  pi@<pi-ip>:/home/pi/heron_deterrent/media/ \
  ~/heron_pull_pi/
ssh -i ~/.ssh/heron_pi pi@<pi-ip> 'find ~/heron_deterrent/media -type d -empty -delete'
```

Three warnings:

- **Never pull into this repo's `media/`.** That directory is the curated
  reference set behind every threshold in both repos, and
  `tools/measure_thresholds.py` reads it as verified negatives. Mixing board
  clips into it silently corrupts the ground truth. Use a separate directory.
- **One pull directory per board.** The Coral's clips are a different board,
  scene and possibly camera — treat them as a different image domain. Pulling
  the Pi into the Coral's old
  `~/heron_board_pull` mixes them in the same `motion/`, `unknown/` folders,
  and only the date in each filename tells them apart afterwards. (It happened
  on the build; they were separated by date.)
- macOS ships **openrsync**, not GNU rsync. `--remove-source-files`,
  `--stats` and `--prune-empty-dirs` all work; `--info=stats2` does not.

`media/labeled/` — every directory under it, including `day/`, `ir/` and
`dark/` — is exempt from the age sweep and both size caps by design: curated
files are found recursively under `labeled/`, and no clip pool may live there
(an import-time assertion and `tests/test_media_store.py` both enforce it).
Hand-labelled training data is never deleted; it only ever triggers the
`max_labeled_gb` warning. Train from **one** of `labeled/day`, `labeled/ir`,
`labeled/dark` — never a mixture.

The most valuable thing you can collect is **negative examples** from your own
pond — the model was trained on ~180 close-up herons with no negatives, which is
why non-heron frames score as high as **0.89**: measured on real camera footage,
18% of frames containing no heron at all still clear a 0.75 threshold. `deter=0.90`
holds them out today, but two false-positive tails a hair higher would put the
sound firing at nothing. Label pond footage in the dashboard's To Triage
panel, or turn on `collect_mode` to have the device gather borderline clips.

---

## Rollback

None of this section has been exercised on the Pi; the commands are standard
systemd and git.

**Stop it now (sudo):**

```bash
sudo systemctl stop heron-deterrent
sudo systemctl disable heron-deterrent      # also stop it starting at boot
```

**Deterrent firing when it should not:** set `deterrent: false` on the mode
in question (it ships `false` everywhere) and restart. Detection, recording and
alerts carry on; only the sound stops. Motion-only night modes cannot fire it at
all — that combination refuses to start.

**Bad code or config — go back to the last known-good commit:**

```bash
cd ~/heron_deterrent
git stash                       # keep local config edits
git log --oneline | head        # pick a commit that worked
git checkout <commit>
sudo systemctl restart heron-deterrent
```

Models and `sounds/*.wav` are gitignored, so `git checkout` will not remove
them. **Careful going back before 2026-11-25** (`4d094d7`): older loaders only
know `tflite_runtime`, which does not exist on the Pi, so the model will not
load.

**Back to a clean slate, keeping footage:**

```bash
sudo systemctl stop heron-deterrent
mv media ~/media_backup && mv data ~/data_backup
sudo systemctl start heron-deterrent        # recreates both empty
```

**Total removal:**

```bash
sudo systemctl disable --now heron-deterrent
sudo rm /etc/systemd/system/heron-deterrent.service /etc/heron-deterrent.env /etc/asound.conf
sudo systemctl daemon-reload
rm -rf ~/heron_deterrent
```

---

## One-page recovery

Something is wrong and you want the shortest path to a diagnosis:

```bash
cd ~/heron_deterrent
venv/bin/python preflight.py                     # names the fault and the fix
systemctl status heron-deterrent                 # running? restart looping?
journalctl -u heron-deterrent -n 50              # last 50 lines (this boot only)
tail -50 heron_deterrent.log                     # survives reboots
df -h /                                          # full card?
ls /dev/video*; v4l2-ctl --list-devices          # camera present?
aplay -l; cat /etc/asound.conf                   # speaker present, sound pointed at the jack?
vcgencmd measure_temp; vcgencmd get_throttled    # too hot, or under-voltage?
```
