# Coral Dev Board and Edge TPU — legacy

> **Read this before buying Coral hardware for this project.** The project now
> targets a Raspberry Pi 4 running CPU inference; start at
> [DEPLOY.md](../../../DEPLOY.md). Everything in this directory is kept for
> reference, not as a recommended path.

## Why this is legacy

- **The Coral software is archived.** As of 2026-11-25 the GitHub API reports
  `google-coral/edgetpu`, `pycoral`, `libedgetpu` and `libcoral` all as
  `archived: true`. (The API does not say *when*; the project owner reports the
  edgetpu repository was archived in April 2026.) No more runtime releases are
  coming, so Mendel — Debian 10, Python 3.7 — is end-of-life.
- **The Edge TPU model detects nothing** (below), so this project has never
  actually used the TPU in the field. The Coral ran CPU inference.
- **A Pi 4 runs the same CPU model about 3× faster** — median 314 ms per
  inference against 920 ms on the Coral — which removes most of the reason to
  want the accelerator in the first place.

The code still supports `ai.target: edgetpu`, and nothing here has been deleted:
`heron-detector`'s `reports/task1_edge_tpu_recoverability.md` refers to this
work, and the board itself still runs.

| File | What it is |
|---|---|
| [coral_dev_board_flash_guide.md](coral_dev_board_flash_guide.md) | Flashing Mendel onto the board. Not re-checked since the Pi became the target |
| this file | The Edge TPU diagnosis, measured Coral figures, and the Mendel traps |

---

## Edge TPU status

**The Edge TPU is not currently usable, and it is the model's fault, not the
board's.** `ai.target: edgetpu` loads and runs at 87 ms/frame, but detects
nothing: full-integer quantisation destroyed the model's classification head, so
every one of the 8,400 anchors scores about `1e-5`. The bounding-box outputs also
come back in pixel units rather than normalised, unlike the CPU model.

The hardware itself is healthy — an older compiled model runs on the same TPU at
3 ms/frame.

### Why quantisation destroyed it

YOLOv8's `[1, 5, 8400]` output puts pixel-scale box coordinates (0–644) and
probability-scale class scores (0–1) in **one tensor**. Full-integer
quantisation gives that tensor one scale; it lands around 2.53 per step, which
rounds every score to zero.

An earlier README said the fix was "re-quantising with better calibration".
**That is wrong.** The calibration images *were* stretched rather than
letterboxed, and that bug is fixed, but `heron-detector`'s
`reports/task1_edge_tpu_recoverability.md` found that correcting calibration
alone produced an *even worse* model. Calibration was not the root cause.

### Two workarounds — untried, not unavailable

Neither of these has been attempted. That is not the same as them being
unavailable, and nothing in the diagnosis suggests they would fail:

1. **Split the output head into two tensors** before quantisation, so boxes and
   scores each get their own scale. A well-trodden path in the YOLO + Coral
   community; look for prior art rather than inventing it.
2. **Ultralytics' native `format='edgetpu'` export**, which may already do that
   split internally. It needs `edgetpu_compiler`, which only runs on x86-64
   Linux — not macOS, not the board.

They were not tried because inference speed was not the bottleneck: the training
data was (≈180 stock images, no real negatives). The report's order still
stands — **retrain on real footage with negatives first, then try (2), then
(1).** Validate detection *scores* on real images before deploying any compiled
model, not just that the file loads.

### Two more things a working TPU model would still need

- **The box units.** The detector's letterbox undo assumes normalised boxes,
  which is right for the CPU model. A TPU model emitting pixels would need
  scaling by `1/input_w` before the undo in `heron_deterrent` — not done.
- **A runtime on the Pi.** `ai/model_loader.py` builds the Edge TPU interpreter
  from `tflite_runtime` only. On Raspberry Pi OS 13 (Python 3.13),
  `pip install tflite-runtime` finds no versions at all — so **`ai.target:
  edgetpu` cannot load on the current Pi** with or without a Coral USB
  Accelerator. Using one would need the loader to take `load_delegate` from
  `ai_edge_litert` as well, plus a `libedgetpu` that installs on Trixie. Neither
  has been tried.

### Exporting for the Edge TPU

Full-integer quantisation (uint8 in and out) is required by `edgetpu_compiler`,
and it is that step which destroyed the classification head in the current
model:

```bash
edgetpu_compiler -s best_full_integer_quant.tflite
cp best_full_integer_quant_edgetpu.tflite models/heron_edgetpu.tflite
```

### Coral USB Accelerator on a Pi

The old Pi guide described adding a Coral USB Accelerator via Google's apt
repository (`packages.cloud.google.com/apt coral-edgetpu-stable`) and
`apt-key add`. **Not tried on this Pi**, and unlikely to work as written:
`apt-key` is deprecated on current Debian, the runtime repositories are
archived, and the loader problem above applies regardless.

---

## Measured on the Coral

All figures from a Coral Dev Board (Mendel 5, quad Cortex-A53), CPU inference,
640×640 model:

| Metric | Value |
|---|---|
| Inference | 920 ms/frame |
| Motion → deterrent sound | **1.02 s** |
| Saved clip size | 117 KB (5 s, 640×480) |
| Resident memory | ~110 MB |
| Edge TPU, current (broken) model | 87 ms/frame, detects nothing |
| Edge TPU, an older compiled model | 3 ms/frame |

**A note on 320 vs 640.** A 320×320 export ran 4× faster on the Coral (220 ms
vs 920 ms) but measured **10 points lower recall** — 86% vs 96% at threshold
0.75, with distant birds worst affected. Those recall figures predate the
camera simulation in `tools/measure_thresholds.py` (today's harness gives 0.84
at 0.75 for the 640 model), so read them as a relative comparison, not current
numbers. The conclusion — 640 is fast enough, resolution is not worth trading
— is even more true on the Pi.

---

## Mendel traps

Collected from real deploys on the bench board. Each of these cost time once.

**Interpreters.** `/usr/bin/python` is Python **2.7.16**, and the system
`python3` has no PyYAML and no Flask. So `python main.py` dies on the first type
annotation with `SyntaxError: invalid syntax`, and `./preflight.py` (whose
shebang finds the system `python3`) fails with `No module named 'yaml'` and a
misleading hint about the repo root. Always:

```bash
venv/bin/python preflight.py
START_WEB_UI=true venv/bin/python main.py
```

**The venv must see system packages.** Mendel's `python3-tflite-runtime`
(2.5.0.post1) is version-matched to `libedgetpu 16.0`:

```bash
python3 -m venv --system-site-packages venv
```

**Never `pip install tflite-runtime` on Mendel** — the pip build targets a newer
TFLite and segfaults when the Edge TPU delegate constructs an Interpreter. If
the model load segfaults, that is usually why: `pip uninstall tflite-runtime`
and check the venv has `include-system-site-packages = true`. `setup.sh`'s
non-Pi branch handles this.

**`/home` is a separate 2 GB partition**, and the repo lives there. `df -h /`
showing plenty of space is misleading — check `df -h /home`. The clip caps in
`config.yaml` are sized for a Pi's card; on a Coral bring them down to roughly
`max_media_gb: 0.7` and `max_motion_gb: 0.3`. Preflight fails if they cannot
fit.

**The unit file targets the Pi.** `heron-deterrent.service` runs as `pi` from
`/home/pi`; on a Coral change `User`, `Group` and both paths to `mendel`.

**Networking.** The bench board was on Wi-Fi (`wlan0`; there is no `eth0` on
it), on DHCP, and its address moved between sessions. When it moves, SSH says
`Host key verification failed`; confirm it is the same board with
`ssh-keyscan -t ed25519 <new-ip>` against the key already trusted, rather than
clearing the entry blindly.

**No answer on port 22** while it answers `ping` usually means `sshd` is off.
Connect the **micro-USB** port (the small one, not either USB-C) and open a
serial console at 115200 (macOS needs the Silicon Labs CP210x driver):

```bash
ls /dev/cu.*
screen /dev/cu.SLAB_USBtoUART 115200      # login: mendel / mendel
sudo systemctl enable --now ssh            # Debian-based: ssh, not sshd
ip addr show
```

**Reboots are fast** — back on the network in about **25 seconds**. Quick enough
that an immediate reconnect can succeed *before* the board has gone down,
making a failed reboot look like a passed one. Check `uptime -s`.

**The clock is UTC**, not local time.

### Edge TPU not detected

```bash
lspci -nn | grep 089a
dmesg | grep apex
python3 -c "from pycoral.utils.edgetpu import list_edge_tpus; print(list_edge_tpus())"
```
