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
echo ""
echo "Creating Python virtual environment..."
python3 -m venv venv
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

# Create default config if it doesn't exist
if [ ! -f config.yaml ]; then
    echo ""
    echo "Creating default config.yaml..."
    cat > config.yaml << EOF
motion:
  sensitivity: 5000
  cooldown_seconds: 10

ai:
  confidence_threshold: 0.6
  model_path: "./models/best_int8.tflite"
  use_edge_tpu: false
  num_threads: 4

alert:
  sms_enabled: false
  sms_throttle_minutes: 15
  twilio_account_sid: ""
  twilio_auth_token: ""
  twilio_from_number: ""
  twilio_to_number: ""

audio:
  volume: 80
  sound_path: "./sounds/"

storage:
  retention_days: 365
  media_path: "./media/"
  db_path: "./data/heron.db"

mqtt:
  enabled: false
  broker: "localhost"
  port: 1883
  topics:
    detection: "heron/detection"
    alert: "heron/alert"
    unclassified: "heron/unclassified"
    status: "heron/system/status"

system:
  active_hours:
    start: "06:00"
    end: "20:00"
  camera_device_id: 0
  frame_rate: 5
  resolution:
    width: 640
    height: 480
  running_on_raspberry_pi: ${IS_RASPI}
EOF
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Copy your best_int8.tflite model to the models/ directory"
echo "   NOTE: Use the standard int8 model, NOT the _edgetpu variant"
echo "2. Add WAV sound files to the sounds/ directory"
echo "3. Edit config.yaml with your Twilio credentials (or use env vars)"
echo "4. Run: source venv/bin/activate"
echo "5. Run: python main.py"
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
