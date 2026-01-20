# ============================================================
# FILE: README.md
# ============================================================

README = '''
# Heron Deterrent Solution

An edge AI-powered system to detect and deter herons from fish ponds using computer vision and audio deterrents.

## Features

- **Real-time Motion Detection**: Monitors camera feed for movement
- **AI Object Detection**: Uses YOLO model optimized for heron detection
- **Audio Deterrent**: Automatically plays random sounds to scare away herons
- **Web Dashboard**: Monitor live stream, view alerts, and manage detections
- **Manual Labeling**: Review and label unclassified objects to improve the model
- **MQTT Integration**: Optional message queue for IoT integrations
- **SMS Alerts**: Get notified when herons are detected (via Twilio)
- **Data Retention**: Automatic cleanup of old videos and database records

## Hardware Requirements

- Coral Dev Board or Raspberry Pi 4
- USB Webcam (720p or 1080p)
- External Speaker (3.5mm or USB)
- Optional: Coral USB Accelerator for Raspberry Pi

## Software Requirements

- Python 3.8+
- OpenCV
- TensorFlow Lite
- Flask (for web UI)
- See requirements.txt for full list

## Installation

### 1. Clone and Setup

```bash
git clone <repository-url>
cd heron_deterrent
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2. Install System Dependencies (Linux/Raspberry Pi)

```bash
sudo apt-get update
sudo apt-get install -y \\
    python3-opencv \\
    alsa-utils \\
    v4l-utils \\
    libportaudio2
```

### 3. Configure the System

Edit `config.yaml` to match your setup:
- Set camera device ID
- Configure active hours
- Add Twilio credentials for SMS alerts
- Set storage paths
- Adjust AI confidence threshold

### 4. Add Your YOLO Model

Place your trained YOLO model (.tflite format) in the models directory:
```bash
mkdir -p models
# Copy your model
cp /path/to/heron_yolo.tflite models/
```

### 5. Add Deterrent Sounds

Add WAV files to the sounds directory:
```bash
mkdir -p sounds
# Add your .wav files
cp /path/to/deterrent_sounds/*.wav sounds/
```

## Running the System

### Start Main System Only

```bash
python main.py
```

### Start with Web UI

```bash
START_WEB_UI=true python main.py
```

Then access the web interface at: http://localhost:5000

## Project Structure

```
heron_deterrent/
├── main.py                 # Main application entry point
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── capture/                # Camera and motion detection
│   ├── camera.py
│   └── motion_detector.py
├── ai/                     # AI model and detection
│   ├── model_loader.py
│   └── detector.py
├── storage/                # Data storage
│   ├── media_store.py
│   └── database.py
├── alerts/                 # Alert system
│   ├── alert_manager.py
│   └── mqtt_client.py
├── deterrent/              # Audio playback
│   └── audio_player.py
├── ui/                     # Web interface
│   ├── app.py
│   └── templates/
│       └── index.html
└── utils/                  # Utilities
    └── config_loader.py
```

## Configuration

Key configuration options in `config.yaml`:

- **motion.sensitivity**: Motion detection threshold (default: 5000)
- **motion.cooldown_seconds**: Delay between detections (default: 10)
- **ai.confidence_threshold**: Minimum confidence for detections (default: 0.6)
- **ai.use_edge_tpu**: Enable Edge TPU acceleration
- **alert.sms_enabled**: Enable SMS notifications
- **storage.retention_days**: Days to keep videos (default: 365)
- **system.active_hours**: Time window for monitoring

## API Endpoints

- `GET /api/live-stream` - Live camera feed
- `GET /api/alerts` - Recent alerts
- `GET /api/detections` - Detection history
- `GET /api/unclassified` - Objects needing manual classification
- `POST /api/label` - Submit manual label
- `POST /api/trigger-deterrent` - Manually trigger sound
- `POST /api/upload-test-video` - Test detection with video file
- `GET /api/system-status` - System health status

## Training Your Model

1. Use the manual labeling feature to create a dataset
2. Export labeled images in YOLO format from `media/labeled/`
3. Train your model using YOLOv8:

```bash
# Install ultralytics
pip install ultralytics

# Train
yolo detect train data=dataset.yaml model=yolov8n.pt epochs=100

# Export to TFLite
yolo export model=runs/detect/train/weights/best.pt format=tflite
```

## Troubleshooting

### Camera not detected
```bash
# List available cameras
v4l2-ctl --list-devices

# Test camera
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### Audio not playing
```bash
# Test speaker
speaker-test -t wav -c 2

# Check volume
amixer get Master
```

### Edge TPU not working
```bash
# Install Edge TPU runtime
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install libedgetpu1-std
```

## License

Open source - free for commercial and personal use.

## Contributing

Contributions welcome! Please submit pull requests or open issues.

## Support

For issues and questions, please open a GitHub issue.
'''

with open('README.md', 'w') as f:
    f.write(README)