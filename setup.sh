#!/bin/bash

echo "=== Heron Deterrent System Setup ==="
echo ""

# Detect platform
if grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "Detected: Raspberry Pi"
    IS_RASPI=true
else
    echo "Detected: Non-Raspberry Pi Linux"
    IS_RASPI=false
fi

# Update system
echo ""
echo "Updating system packages..."
sudo apt-get update

# Install system dependencies
echo ""
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    libopencv-dev \
    python3-opencv \
    alsa-utils \
    v4l-utils \
    libportaudio2 \
    sqlite3 \
    git \
    libatlas-base-dev \
    libjpeg-dev \
    libopenjp2-7

# Raspberry Pi 4 specific setup
if [ "$IS_RASPI" = true ]; then
    echo ""
    echo "=== Raspberry Pi 4 specific setup ==="

    # Enable camera interface via raspi-config (non-interactive)
    echo "Enabling camera interface..."
    sudo raspi-config nonint do_camera 0

    # Ensure audio output goes to 3.5mm jack (0=auto, 1=headphones, 2=HDMI)
    echo "Setting audio output to 3.5mm jack..."
    sudo raspi-config nonint do_audio 1

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

# Install tflite-runtime inside the venv (Pi)
# The correct install method depends on the Python version:
#   Python 3.9  → Google Coral wheel (only version with a pre-built cp39 aarch64 wheel)
#   Python 3.10+ → tflite-runtime is on PyPI with proper aarch64 wheels
if [ "$IS_RASPI" = true ]; then
    echo ""
    echo "Installing tflite-runtime into venv..."
    PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
    if [ "$PY_MINOR" = "9" ]; then
        TFLITE_WHEEL="https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl"
        pip install "$TFLITE_WHEEL" || echo "WARNING: tflite-runtime wheel install failed."
    else
        # Python 3.10+ — PyPI has aarch64 wheels
        pip install tflite-runtime || echo "WARNING: tflite-runtime install failed. Try: pip install tflite-runtime --extra-index-url https://google-coral.github.io/py-repo/"
    fi
fi

# Create directory structure
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
echo "1. Copy the model for your target into models/ (models are not in git):"
echo "     target: cpu      ->  models/heron_cpu_int8.tflite   (either board)"
echo "     target: edgetpu  ->  models/heron_edgetpu.tflite    (Coral, currently broken)"
echo "   config.yaml selects the backend with ai.target (currently: $(grep -m1 "^  target:" config.yaml | awk "{print \$2}"))"
echo "2. Add WAV sound files to sounds/ (startup fails if it is empty)"
echo "3. For SMS, put credentials in /etc/heron-deterrent.env (root, mode 600) —"
echo "   never in config.yaml, which is tracked in git:"
echo "     TWILIO_ACCOUNT_SID TWILIO_AUTH_TOKEN TWILIO_FROM_NUMBER TWILIO_TO_NUMBER"
echo "   Then set alert.sms_enabled: true in config.yaml"
echo "4. Run: source venv/bin/activate"
echo "5. Run: ./preflight.py     # checks every dependency before you start"
echo "6. Run: python main.py"
echo ""
echo "To start with web UI: START_WEB_UI=true python main.py"
echo ""
if [ "$IS_RASPI" = true ]; then
    echo "Pi 4 tips:"
    echo "  - If camera is not detected: run 'vcgencmd get_camera' to verify it's enabled"
    echo "  - If audio is silent: run 'aplay -l' to list devices, 'speaker-test -t wav' to test"
    echo "  - To check tflite: python3 -c 'import tflite_runtime.interpreter; print(\"OK\")'"
    echo ""
fi
