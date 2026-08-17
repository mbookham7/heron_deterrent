# DEPLOY — pond checklist

Ordered steps from a flashed OS to a running service, on either board.
Tick straight down. Rollback is at the bottom.

**Reference values for this build:** `ai.target: cpu`,
`models/heron_cpu_int8.tflite`, threshold `0.75`, service `heron-deterrent`.
Coral user is `mendel`, Pi user is `pi`.

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
./preflight.py
```

**Listen for the speaker.** The audio check plays a real WAV; if you hear
nothing, the deterrent will be silent no matter what the check says about the
device being present.

Every failure prints the command that fixes it. Do not continue until you see:

```
All 7 checks passed.
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
START_WEB_UI=true python main.py
```

Expect, within a couple of seconds:

```
Model loaded: heron_cpu_int8.tflite | target=cpu | backend=CPU (4 threads) | input=1x640x640x3
Detector ready: input 640x640 float32 | classes=1 | confidence>=0.75
Loaded 1 deterrent sound(s): startle_default.wav
Camera 0 open at 640x480
System started successfully
```

- [ ] Open `http://<board-ip>:5000` — dashboard loads, live feed moves
- [ ] Walk in front of the camera → journal shows
      `Motion detected - inferring on trigger frame`
- [ ] Press **Trigger Deterrent** on the dashboard → you hear the sound
- [ ] Optional, strongest test: upload a heron clip via **Test Upload** →
      expect `herons_detected: 1` or more, and the speaker fires

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
# wait ~90s, then from your laptop:
ssh <user>@<board-ip> 'systemctl is-active heron-deterrent'
```

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
| Deterrent firing at nothing | Raise `ai.confidence_threshold` toward `0.85` |
| Lots of motion, no detections | Lower `motion.sensitivity` to `2000` |
| Hundreds of "unknown" alerts | Raise `motion.sensitivity`; wind and shadow trigger it |
| Herons ignored | Lower `confidence_threshold` to `0.65`; move the camera closer |
| Card filling | Lower `storage.max_media_gb` |

Apply changes with `nano config.yaml && sudo systemctl restart heron-deterrent`.

The most valuable thing you can collect is **negative examples** from your own
pond — the model was trained on ~180 close-up herons with no negatives, which is
why non-heron frames score as high as **0.89**: measured on real camera footage,
18% of frames containing no heron at all still clear the 0.75 threshold. Label
them in the dashboard's Unclassified panel.

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
./preflight.py                                   # names the fault and the fix
systemctl status heron-deterrent                 # running? restart looping?
sudo journalctl -u heron-deterrent -n 50         # last 50 lines
df -h /home                                      # full card?
ls /dev/video*                                   # camera present?
aplay -l                                         # speaker present?
```
