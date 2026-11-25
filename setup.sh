#!/bin/bash

echo "=== Heron Deterrent System Setup ==="
echo ""

# Detect platform. The Raspberry Pi 4 is the target; anything else is assumed
# to be the legacy Coral Dev Board (Mendel) — see docs/legacy/coral/README.md.
if grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "Detected: Raspberry Pi"
    IS_RASPI=true
else
    echo "Detected: not a Raspberry Pi — assuming legacy Coral/Mendel"
    IS_RASPI=false
fi

# Update system
echo ""
echo "Updating system packages..."
sudo apt-get update

# Install system dependencies
#
# Deliberately short. numpy and OpenCV come from pip as wheels that bundle their
# own libraries (numpy carries its own OpenBLAS), so there is no BLAS package
# here. libatlas-base-dev in particular no longer exists: Debian 13 (Trixie),
# which Raspberry Pi OS 13 is built on, removed it, and apt-get fails outright.
#
#   ffmpeg      writes clips as H.264 (needs libx264) so the dashboard can play them
#   alsa-utils  aplay/amixer — the deterrent
#   v4l-utils   v4l2-ctl, for checking the camera
#   sqlite3     the CLI, for inspecting data/heron.db by hand
echo ""
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    alsa-utils \
    v4l-utils \
    sqlite3

# Raspberry Pi: send sound to the 3.5 mm jack.
#
# Out of the box ALSA's default is HDMI (card 0), so the deterrent plays to
# nothing while every check passes. This is system-wide on purpose: the Pi
# desktop's volume plugin removes ~/.asoundrc at login, so a per-user file
# silently stops working after a reboot. And by card name, not number, because
# numbers can shift between boots. "defaults.pcm.card Headphones" looks simpler
# but alsa-lib rejects it ("card is not a string"), breaking ALSA entirely.
if [ "$IS_RASPI" = true ]; then
    echo ""
    if [ -e /etc/asound.conf ]; then
        echo "/etc/asound.conf already exists — leaving it alone. Check it sends sound to the 3.5 mm jack."
    elif grep -q "Headphones" /proc/asound/cards 2>/dev/null; then
        echo "Sending sound to the 3.5 mm jack (/etc/asound.conf)..."
        sudo tee /etc/asound.conf >/dev/null <<'EOF'
# heron_deterrent: the speaker is on the 3.5 mm jack. Without this, ALSA
# defaults to HDMI (card 0) and the deterrent plays to nothing.
# System-wide, not ~/.asoundrc: the desktop volume plugin removes that at login.
# By name, not number: card numbers can shift between boots.
pcm.!default {
    type plug
    slave.pcm "hw:Headphones"
}
ctl.!default {
    type hw
    card "Headphones"
}
EOF
    else
        echo "WARNING: no 'Headphones' sound card found — /etc/asound.conf not written."
        echo "         Sound will go to ALSA's default device. Check with: aplay -l"
    fi
fi

# Create virtual environment
#
# On Mendel (Coral) the venv MUST see system site-packages: the TFLite runtime
# and pycoral come from apt and are version-matched to libedgetpu. Installing
# tflite-runtime from pip there produces a segfault when the Edge TPU delegate
# builds an Interpreter, because the pip build targets a newer TFLite than
# libedgetpu was compiled against.
echo ""
if [ "$IS_RASPI" = true ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
else
    echo "Creating Python virtual environment (--system-site-packages for Mendel)..."
    python3 -m venv --system-site-packages venv
fi
source venv/bin/activate

# Install Python packages
echo ""
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# TFLite runtime (Pi)
#
# tflite-runtime has no wheel for Python 3.13, which Raspberry Pi OS 13 ships:
# pip finds no versions at all. ai-edge-litert is its successor (TensorFlow Lite
# was renamed LiteRT) and installs cleanly; ai/model_loader.py tries it after
# tflite_runtime.
if [ "$IS_RASPI" = true ]; then
    echo ""
    echo "Installing ai-edge-litert (the TFLite runtime) into venv..."
    pip install ai-edge-litert || echo "WARNING: ai-edge-litert install failed — the model will not load."
fi

# Create directory structure. models/ must exist before you scp a model into
# it: it is gitignored, so a fresh clone does not have it.
echo ""
echo "Creating directory structure..."
mkdir -p models sounds media data

# config.yaml is tracked in git and is the only copy of the schema. There used
# to be a second copy here as a heredoc; it drifted (it still carried a 0.75
# threshold and active_hours long after both were gone), so restore from git
# instead of generating.
if [ ! -f config.yaml ]; then
    echo ""
    echo "config.yaml is missing — restoring the tracked copy from git..."
    git checkout -- config.yaml || {
        echo "ERROR: could not restore config.yaml. Re-clone the repo, or copy it from another checkout."
        exit 1
    }
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Copy the model into models/ from your laptop (models are not in git):"
echo "     scp models/heron_cpu_int8.tflite $USER@<board-ip>:~/heron_deterrent/models/"
echo "   config.yaml selects the backend with ai.target (currently: $(grep -m1 "^  target:" config.yaml | awk "{print \$2}"))"
echo "2. sounds/ already has startle_default.wav; add more .wav files if you like"
echo "3. For SMS, put credentials in /etc/heron-deterrent.env (root, mode 600) —"
echo "   never in config.yaml, which is tracked in git:"
echo "     TWILIO_ACCOUNT_SID TWILIO_AUTH_TOKEN TWILIO_FROM_NUMBER TWILIO_TO_NUMBER"
echo "   Then set alert.sms_enabled: true in config.yaml"
echo "4. Run: venv/bin/python preflight.py     # checks every dependency; plays the deterrent"
echo "5. Run: START_WEB_UI=true venv/bin/python main.py"
echo ""
if [ "$IS_RASPI" = true ]; then
    if [ "$(systemctl get-default 2>/dev/null)" = "graphical.target" ]; then
        echo "NOTE: this Pi boots to the desktop. Nothing here needs it, and in a sealed"
        echo "      box it is only extra heat. To boot to the console instead:"
        echo "        sudo systemctl set-default multi-user.target && sudo reboot"
        echo ""
    fi
    echo "Pi tips:"
    echo "  - Camera: v4l2-ctl --list-devices"
    echo "  - Sound:  preflight plays the deterrent — listen for it. It passes on any"
    echo "            device that exists, so it cannot tell you the speaker is silent"
    echo "  - Heat:   vcgencmd measure_temp; vcgencmd get_throttled   # want throttled=0x0"
    echo ""
fi
