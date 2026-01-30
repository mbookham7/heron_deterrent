#!/bin/bash

echo "=== Heron Deterrent System Setup ==="
echo ""

# Check if running on Raspberry Pi
if grep -q "Raspberry Pi" /proc/cpuinfo; then
    echo "✓ Detected Raspberry Pi"
    IS_RASPI=true
else
    echo "ℹ Running on other Linux system"
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
    python3-opencv \
    alsa-utils \
    v4l-utils \
    libportaudio2 \
    sqlite3 \
    git

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

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p models sounds media data

# Create default config if it doesn't exist
if [ ! -f config.yaml ]; then
    echo ""
    echo "Creating default config.yaml..."
    cat > config.yaml << 'EOF'
motion:
  sensitivity: 5000
  cooldown_seconds: 10

ai:
  confidence_threshold: 0.6
  model_path: "./models/heron_int8_edgetpu.tflite"
  use_edge_tpu: false

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
  frame_rate: 10
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
echo "1. Add your YOLO model to the models/ directory"
echo "2. Add WAV sound files to the sounds/ directory"
echo "3. Edit config.yaml to configure the system"
echo "4. Run: source venv/bin/activate"
echo "5. Run: python main.py"
echo ""
echo "To start with web UI: START_WEB_UI=true python main.py"
echo ""