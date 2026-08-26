# DEPLOY — pond checklist

Ordered steps from a flashed OS to a running service, on either board.
Tick straight down. Rollback is at the bottom.

**Reference values for this build:** `ai.target: cpu`,
`models/heron_cpu_int8.tflite`, `deter_threshold: 0.90` / `alert_threshold: 0.95`,
service `heron-deterrent`. Coral user is `mendel`, Pi user is `pi`.

---

## 0. Before you leave the house

- [ ] Board, **powered** speaker, USB webcam, SD card, power supply, cables
- [ ] Laptop can reach the board (know its IP, or bring an Ethernet cable)
- [ ] `models/heron_cpu_int8.tflite` on your laptop — **it is gitignored, a clone
      does not include it**
- [ ] A real `.wav` if you want better than the built-in startle burst
- [ ] Mounting: camera facing the water, speaker aimed at the landing spot.
      Close to the water beats a wide view — distant birds are missed.

---

## 1. Get a shell on the board

```bash
ssh mendel@<board-ip>          # Coral
ssh pi@<board-ip>              # Pi
```

**The board is on DHCP, so the address moves.** If it stops answering on the
address you wrote down, find it again (`ping`, or check the router) before
assuming a fault. When it does move, SSH refuses the new address with
`Host key verification failed` — that is expected. Confirm it is the same board
rather than blindly clearing the entry: the host key should be *identical* to
the one already trusted for the old address.

```bash
ssh-keyscan -t ed25519 <new-ip>            # compare against ~/.ssh/known_hosts
```

**No answer on port 22?** If the board replies to `ping` but refuses port 22,
`sshd` is probably disabled rather than the network being broken. You need local
access once: connect the **micro-USB** port (the small one, not either USB-C) and
open a serial console at 115200 (macOS needs the Silicon Labs CP210x driver):

```bash
ls /dev/cu.*
screen /dev/cu.SLAB_USBtoUART 115200      # login: mendel / mendel
sudo systemctl enable --now ssh            # Debian-based: ssh, not sshd
ip addr show                               # note the real IP
```

---

## 2. Install

```bash
git clone https://github.com/mbookham7/heron_deterrent.git
cd heron_deterrent
./setup.sh
```

`setup.sh` installs system packages, creates the venv (with
`--system-site-packages` on Mendel — required), installs Python dependencies,
and writes a default `config.yaml` if none exists.

**On Mendel, never `pip install tflite-runtime`.** The system package
(`python3-tflite-runtime` 2.5.0.post1) is version-matched to `libedgetpu`; pip's
newer build segfaults when the Edge TPU delegate constructs an Interpreter.

---

## 3. Copy the model and sounds across

From your laptop:

```bash
scp models/heron_cpu_int8.tflite <user>@<board-ip>:~/heron_deterrent/models/
scp my_heron_call.wav            <user>@<board-ip>:~/heron_deterrent/sounds/
```

The repo already ships `sounds/startle_default.wav`, so the sounds copy is
optional — but the model copy is not. Without it, startup fails.

---

## 4. Check `config.yaml`

```bash
nano config.yaml
```

- [ ] `ai.target: cpu` — **use `cpu` on the Coral too.** The `edgetpu` model
      detects nothing (see README, "Edge TPU status")
- [ ] `system.camera_device_id: 0` — check with `ls /dev/video*`
- [ ] `system.active_hours` covers when herons actually visit
- [ ] `storage.max_media_gb: 1.0` — lower it if the card is small.
      **On Coral, check `df -h /home`: it is a separate 2 GB partition**
- [ ] `alert.sms_enabled: false` unless you have done step 5
- [ ] `collect_mode.enabled` — `false` for a working deterrent. Set `true` only
      for a data-collection run, and read the storage warning in step 10 first:
      it records everything above `record_threshold` (0.30), which on a bench is
      near-continuous. `collect_mode.deterrent_enabled: false` silences the
      sound independently, so you can gather footage without the bang.

---

## 5. SMS — skip if you have no Twilio account

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

**Always run it through the venv.** `./preflight.py` uses the `#!/usr/bin/env
python3` shebang, which on Mendel finds the *system* Python — and that has no
PyYAML and no Flask. It fails at the first check with:

```
[FAIL] config       cannot load config.yaml: No module named 'yaml'
       -> Run preflight from the repo root: cd ~/heron_deterrent && ./preflight.py
```

That hint is misleading: the repo root is not the problem, the interpreter is.
`venv/bin/python preflight.py` (or `source venv/bin/activate` first) fixes it.

**Listen for the speaker.** The audio check plays a real WAV; if you hear
nothing, the deterrent will be silent no matter what the check says about the
device being present. Running remotely, where nobody is in the room to hear it?
Use `--skip-audio` and treat the speaker as unverified — the check can only
prove the device exists, never that sound came out of it.

Every failure prints the command that fixes it. Do not continue until you see:

```
All 8 checks passed.
```

Two things preflight cannot see when run by hand:

- **The camera, if the service is already running** — only one process can hold
  `/dev/video0`, so the check reports `[skip] camera ... held by the running
  service`. That is not a fault. For a full camera test:
  `sudo systemctl stop heron-deterrent && ./preflight.py`
- **`/etc/heron-deterrent.env`** — only the service reads it. `export` the
  variables in your shell if you want to verify SMS manually.

---

## 7. Run it by hand once

```bash
START_WEB_UI=true venv/bin/python main.py
```

**Not `python main.py`.** On Mendel `/usr/bin/python` is Python **2.7.16**, so
that command dies immediately on the first type annotation:

```
def __init__(self, config_path: str = "config.yaml"):
                              ^
SyntaxError: invalid syntax
```

Expect, within a couple of seconds:

```
Model loaded: heron_cpu_int8.tflite | target=cpu | backend=CPU (4 threads) | input=1x640x640x3 float32 | output=1x5x8400
Detector ready: input 640x640 float32 | classes=1 | confidence>=0.9 | iou=0.45
Thresholds: deter=0.9  alert=0.95  vote=3/5  anchor_floor=0.9
Loaded 1 deterrent sound(s): startle_default.wav
Camera 0 open at 640x480
System started successfully
```

With `collect_mode.enabled: true` the last two of those change, and one more
line appears — this is correct, not a fault:

```
Detector ready: ... | confidence>=0.3 | iou=0.45          <- anchor floor drops to record_threshold
Thresholds: deter=0.9  alert=0.95  vote=3/5  anchor_floor=0.3
COLLECT MODE ON: recording everything above 0.3, deterrent DISABLED
```

- [ ] Open `http://<board-ip>:5000` — dashboard loads, live feed moves
- [ ] Walk in front of the camera → journal shows
      `Motion detected - inferring on trigger frame`
- [ ] Press **Trigger Deterrent** on the dashboard → you hear the sound.
      **Silent on a collect run:** with `collect_mode.deterrent_enabled: false`
      this logs `Deterrent SILENCED` and plays nothing. That is the flag
      working, not a broken speaker.
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

`Ctrl-C` when satisfied.

---

## 8. Install as a service

Coral users skip the edit; the unit already targets `mendel`.

```bash
# Raspberry Pi only: change mendel -> pi in User, Group and both paths
nano heron-deterrent.service

sudo cp heron-deterrent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable heron-deterrent
sudo systemctl start heron-deterrent
sleep 10
systemctl status heron-deterrent
```

- [ ] `Active: active (running)`
- [ ] `sudo journalctl -u heron-deterrent -f` shows the startup lines from step 7
- [ ] Dashboard still responds on `:5000`

---

## 9. Prove it survives

```bash
# Restart on crash
sudo kill -9 $(systemctl show -p MainPID --value heron-deterrent)
sleep 15
systemctl is-active heron-deterrent          # expect: active

# Restart on reboot
sudo reboot
# wait, then from your laptop:
ssh <user>@<board-ip> 'systemctl is-active heron-deterrent'
```

The Coral is back on the network in about **25 seconds**, not 90 — quick enough
that an immediate reconnect can succeed *before* the board has actually gone
down, making a failed reboot look like a passed one. Confirm with `uptime -s`
that the boot time is what you expect before trusting the result.

- [ ] Survives `kill -9` (systemd waits 10 s, then restarts)
- [ ] Survives a reboot
- [ ] Dashboard reachable after reboot

If it fails 5 times in 2 minutes systemd gives up deliberately, so the fault is
visible in `systemctl status` instead of scrolling past in the journal.

---

## 10. Leave it running — tuning notes

Come back after a day and check:

```bash
sudo journalctl -u heron-deterrent --since today | grep -c 'HERON DETECTED'
sudo journalctl -u heron-deterrent --since today | grep -c 'Motion detected'
df -h /home
du -sh media/*
```

| Symptom | Fix |
|---|---|
| Deterrent firing at nothing | Check `media/heron/` first — at `deter_threshold=0.90` no bench frame clears the bar, so if it is firing something is genuinely triggering. If it truly is nothing, raise `deter_threshold` further. |
| Lots of motion, no detections | Lower `motion.sensitivity` to `2000` |
| Hundreds of "unknown" alerts | Raise `motion.sensitivity`; wind and shadow trigger it |
| Herons ignored | Lower `deter_threshold` toward `0.75` (raises recall on stock positives 0.41 → 0.84, but lets 18% of bench frames through); move the camera closer |
| Card filling | Lower `storage.max_media_gb` |

Apply changes with `nano config.yaml && sudo systemctl restart heron-deterrent`.

### Collect mode fills the card, and the safety net eats your data

`storage.max_media_gb` deletes **oldest first**. On a collect run the oldest
clips are exactly the footage you went out to gather, so the mechanism that
protects the card is also the one that quietly discards your dataset. It runs
every 5 minutes (`CLEANUP_INTERVAL_SECONDS`) and logs `Size cap: deleted N
oldest file(s)` when it bites — that line is the only warning you get.

Measured on the Coral bench (640x480, 5 fps, 5 s clips, `record_threshold` 0.30):

| | clips/hour | growth | time to the 1.0 GB cap |
|---|---|---|---|
| Observed, ordinary room activity | ~80 | ~12 MB/h | **~3.8 days** |
| Saturated, continuous motion (10 s cooldown) | ~225 | ~33 MB/h | **~32 hours** |

Mean clip is ~147 KB. **Offload before that window closes.** From your laptop:

```bash
mkdir -p ~/heron_board_pull
rsync -a --stats \
  -e "ssh -i ~/.ssh/heron_coral -o IdentitiesOnly=yes" \
  mendel@<board-ip>:/home/mendel/heron_deterrent/media/ \
  ~/heron_board_pull/
```

To pull *and* free the card in one step, so the cap can never reach your
footage, add `--remove-source-files` (then prune the emptied directories):

```bash
rsync -a --stats --remove-source-files \
  -e "ssh -i ~/.ssh/heron_coral -o IdentitiesOnly=yes" \
  mendel@<board-ip>:/home/mendel/heron_deterrent/media/ \
  ~/heron_board_pull/
ssh mendel@<board-ip> 'find ~/heron_deterrent/media -type d -empty -delete'
```

Two warnings:

- **Never pull into this repo's `media/`.** That directory is the curated
  reference set behind every threshold in both repos, and
  `tools/measure_thresholds.py` reads it as verified negatives. Mixing board
  clips into it silently corrupts the ground truth. Use a separate directory.
- macOS ships **openrsync**, not GNU rsync. `--remove-source-files`,
  `--stats` and `--prune-empty-dirs` all work; `--info=stats2` does not.

`media/labeled/` is exempt from both the age sweep and the size cap by design
(`CURATED_MEDIA_SUBDIRS`), so hand-labelled training data is never deleted —
it only ever triggers the `max_labeled_gb` warning.

The most valuable thing you can collect is **negative examples** from your own
pond — the model was trained on ~180 close-up herons with no negatives, which is
why non-heron frames score as high as **0.89**: measured on real camera footage,
18% of frames containing no heron at all still clear a 0.75 threshold. `deter=0.90`
holds them out today, but two false-positive tails a hair higher would put the
sound firing at nothing. Label pond footage in the dashboard's Unclassified
panel, or turn on `collect_mode` to have the device gather borderline clips.

---

## Rollback

**Stop it now:**

```bash
sudo systemctl stop heron-deterrent
sudo systemctl disable heron-deterrent      # also stop it starting at boot
```

**Deterrent firing all night:** stop the service as above, or narrow
`system.active_hours` and restart. To keep detection but silence it, move the
WAVs aside — but note the system then **refuses to start**, deliberately:

```bash
mkdir -p ~/sounds_disabled && mv sounds/*.wav ~/sounds_disabled/
```

**Bad code or config — go back to the last known-good commit:**

```bash
cd ~/heron_deterrent
git stash                       # keep local config edits
git log --oneline | head        # pick a commit that worked
git checkout <commit>
sudo systemctl restart heron-deterrent
```

Models and `sounds/*.wav` are gitignored, so `git checkout` will not remove
them.

**Back to a clean slate, keeping footage:**

```bash
sudo systemctl stop heron-deterrent
mv media ~/media_backup && mv data ~/data_backup
sudo systemctl start heron-deterrent        # recreates both empty
```

**Total removal:**

```bash
sudo systemctl disable --now heron-deterrent
sudo rm /etc/systemd/system/heron-deterrent.service /etc/heron-deterrent.env
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
sudo journalctl -u heron-deterrent -n 50         # last 50 lines
df -h /home                                      # full card?
ls /dev/video*                                   # camera present?
aplay -l                                         # speaker present?
```
