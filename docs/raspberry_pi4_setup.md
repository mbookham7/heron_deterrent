# Raspberry Pi 4 — from a blank card to a shell

This gets a Raspberry Pi 4 from a blank microSD card to an SSH session you can
deploy into. The deploy itself — install, model, preflight, service — is
[DEPLOY.md](../DEPLOY.md). Why things are the way they are is
[README.md](../README.md).

The Raspberry Pi 4 is the only first-class target. (The Coral Dev Board is
legacy: [docs/legacy/coral/](legacy/coral/README.md).)

**What the build used, on 2026-11-25:** Raspberry Pi 4 Model B Rev 1.5 (8 GB),
128 GB card, and the image `/etc/rpi-issue` calls "Raspberry Pi reference
2026-10-06" — Raspberry Pi OS 13 "Trixie" (Debian 13), 64-bit, Python 3.13.5.
Every command below ran on that Pi unless it says otherwise.

---

## Step 1 — Flash the card

Use **Raspberry Pi Imager**. Choose **Raspberry Pi OS (64-bit)** and, in the
customisation settings, set:

| Setting | The build used | Notes |
|---|---|---|
| Hostname | `heron-pi` | |
| Username | `pi` | `heron-deterrent.service` assumes `pi`. Choose another and you edit the unit |
| Password | set one | `sudo` will ask for it — Imager does not set up password-less sudo |
| Wi-Fi | your network and country | |
| SSH | enabled, password authentication | Imager's "public-key only" option, pasting the key from Step 3, should save the `ssh-copy-id` step. **Not tried** |
| Time zone | Europe/London | The build's clock came up as `Europe/London` with NTP active (`timedatectl`) — unlike the old Coral, which ran on UTC |

**Lite or desktop?** Nothing here needs a desktop, and in a sealed box it is
only extra heat — so **Raspberry Pi OS Lite** is the natural choice. But be
honest about what was tested: **the build used the desktop image** and then
switched it to boot to the console (Step 5). Lite has not been tried. `setup.sh`
installs everything the app needs from apt (`ffmpeg`, `alsa-utils`,
`v4l-utils`, …), so it *should* work on Lite; on the desktop image most of
those were already there.

Imager applies these settings on first boot, from `user-data` and
`network-config` files it writes to the card's boot partition.

---

## Step 2 — First boot, and find it

Put the card in, power on, give it a minute or two. Then find its address —
the router's list of connected devices is the most reliable way, or try
`ping heron-pi.local` from a Mac.

If Ethernet is connected as well as Wi-Fi, the Pi gets an address on each.
Either works; the Wi-Fi one is what it will have in the filter house.

---

## Step 3 — Key-based SSH

Everything after this is easier with a key — and a tool that drives the Pi
remotely cannot type a password at all.

On your laptop, once:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/heron_pi -N "" -C "heron-pi-deploy"
```

Then copy it to the Pi. **Do this in a normal terminal window** — it asks
"are you sure you want to continue connecting?" and then for the Pi's
password, and both need a real terminal:

```bash
ssh-copy-id -i ~/.ssh/heron_pi.pub pi@<pi-ip>
```

It ends with `Number of key(s) added: 1`. From then on:

```bash
ssh -i ~/.ssh/heron_pi -o IdentitiesOnly=yes pi@<pi-ip>
```

> On the build, `ssh-copy-id` first failed with `Host key verification failed`.
> That was *not* a changed key — it was a host SSH had never seen, run from a
> place with no terminal to answer the first-contact question. Trusting the
> key explicitly fixed it:
>
> ```bash
> ssh-keyscan -t ed25519 <pi-ip> | ssh-keygen -lf -       # note the fingerprint
> ssh-keyscan -t ed25519 <pi-ip> >> ~/.ssh/known_hosts
> ```
>
> To check the fingerprint is really the Pi's, compare it with
> `ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub` run on the Pi's own
> console. Re-flashing makes a new host key, so expect the warning again after
> a re-flash.

---

## Step 4 — Check what you have

```bash
cat /proc/device-tree/model; echo
grep PRETTY /etc/os-release
python3 --version
free -h | sed -n 2p
df -h /
lsusb                                    # the camera: 32e4:9230 HD USB Camera
vcgencmd measure_temp; vcgencmd get_throttled
sudo -n true && echo "sudo: no password" || echo "sudo: needs a password"
```

On the build:

```
Raspberry Pi 4 Model B Rev 1.5
PRETTY_NAME="Debian GNU/Linux 13 (trixie)"
Python 3.13.5
Mem:           7.6Gi       426Mi       6.6Gi ...
/dev/mmcblk0p2  117G  5.2G  107G   5% /
Bus 001 Device 003: ID 32e4:9230 HD USB Camera HD USB Camera
temp=55.5'C
throttled=0x0
sudo: needs a password
```

`lsusb` also lists `2109:3431 VIA Labs, Inc. Hub` — that is the Pi's own
built-in USB hub, not a powered hub you added. A powered hub for the camera
shows up as a second hub.

---

## Step 5 — Boot to the console, not the desktop

If you flashed the desktop image:

```bash
systemctl get-default                    # graphical.target = desktop
sudo systemctl set-default multi-user.target && sudo reboot
```

Three reasons, all found on the build:

- **It deletes the sound fix.** The desktop's volume plugin removes
  `~/.asoundrc` at login. (`setup.sh` uses `/etc/asound.conf` for that reason,
  so this no longer breaks sound — but there is no reason to keep the plugin.)
- **Heat.** It runs a window manager and a login manager in a sealed box.
- **Nothing needs it.** The dashboard is a web page.

After this, `pipewire` and `wireplumber` still start for `pi` (most likely on
SSH login). The deterrent plays through ALSA directly, so they do not affect
it — but that is an observation, not something tested exhaustively.

Now go to **[DEPLOY.md](../DEPLOY.md)**, step 2.

---

## In the box

The Pi goes into a sealed IP-rated box inside the filter house: bare board on
standoffs, heatsinks on the SoC, desiccant, camera on a powered hub. Why each
of those, and the temperatures measured on the bench, are in the README under
"The filter-house enclosure" and "Why the camera goes through a powered hub".

**Nothing has been measured inside the box yet.** On the bench, in open air, it
reached 68.6 °C after a few seconds of back-to-back inference — near
`preflight`'s 70 °C warning before any enclosure. Check once it is in:

```bash
watch -n 5 'vcgencmd measure_temp; vcgencmd get_throttled'
```

`throttled=0x0` is what you want. Any other value: the bits are decoded by
`preflight`'s `cpu temp` check.

---

## Pi-specific troubleshooting

### `pip install tflite-runtime` — "No matching distribution found"

Expected on Python 3.13: there is no build. Use LiteRT, which `setup.sh`
installs:

```bash
venv/bin/pip install ai-edge-litert
venv/bin/python -c "from ai_edge_litert.interpreter import Interpreter; print('OK')"
```

### `libatlas-base-dev` will not install

Expected on Trixie: Debian 13 removed it. `apt-cache policy libatlas-base-dev`
shows `Candidate: (none)`. Nothing needs it — numpy's wheel
brings its own BLAS. Do not hand-install the old Bookworm `.deb`.

### Lots of `/dev/video*` devices

Normal. `/dev/video10`–`31` are the Pi's own video codec and ISP. The USB
camera was `/dev/video0` (and `video1`, its metadata node) on the build:

```bash
v4l2-ctl --list-devices
```

(The old `vcgencmd get_camera` and `raspi-config` camera option are for the
ribbon-cable Pi Camera Module. A USB camera needs neither.)

### Silent speaker

See README, "Sound output", and DEPLOY step 6. In short: ALSA defaults to HDMI;
`/etc/asound.conf` must name the `Headphones` card; `preflight` cannot tell;
headphones in the jack tell you in seconds.

### "What happened before it rebooted?"

The system journal is in RAM only on Raspberry Pi OS
(`/usr/lib/systemd/journald.conf.d/40-rpi-volatile-storage.conf`), so it is
empty after a reboot. The app's `~/heron_deterrent/heron_deterrent.log`
survives. Making the journal persistent would mean more SD card writes; it has
not been changed.
