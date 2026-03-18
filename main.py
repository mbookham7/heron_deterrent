# ============================================================
# FILE: main.py
# ============================================================

import logging
import logging.handlers
import os
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
import signal
import sys

from utils.config_loader import Config
from storage.database import Database
from storage.media_store import MediaStore
from capture.camera import Camera
from capture.motion_detector import MotionDetector
from ai.model_loader import ModelLoader
from ai.detector import ObjectDetector
from deterrent.audio_player import AudioPlayer
from alerts.mqtt_client import MQTTClient
from alerts.alert_manager import AlertManager

# Setup logging with rotation so the log file never fills the SD card
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            'heron_deterrent.log',
            maxBytes=10 * 1024 * 1024,  # 10 MB per file
            backupCount=3
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HeronDeterrentSystem:
    def __init__(self, config_path: str = "config.yaml"):
        logger.info("Initializing Heron Deterrent System...")

        self.config = Config(config_path)
        self.running = False

        self._initialize_components()

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("System initialization complete")

    def _initialize_components(self):
        # Database
        db_path = self.config.get('storage.db_path', './data/heron.db')
        self.database = Database(db_path)

        # Media storage
        media_path = self.config.get('storage.media_path', './media/')
        self.media_store = MediaStore(media_path)

        # Camera
        device_id = self.config.get('system.camera_device_id', 0)
        resolution = (
            self.config.get('system.resolution.width', 640),
            self.config.get('system.resolution.height', 480)
        )
        self.camera = Camera(device_id, resolution)

        # Motion detector
        sensitivity = self.config.get('motion.sensitivity', 5000)
        self.motion_detector = MotionDetector(sensitivity)

        # AI model
        model_path = self.config.get('ai.model_path')
        use_edge_tpu = self.config.get('ai.use_edge_tpu', False)
        num_threads = self.config.get('ai.num_threads', 1)

        # Raspberry Pi 4 has no built-in Edge TPU
        if self.config.get('system.running_on_raspberry_pi', False):
            logger.info("Running on Raspberry Pi - using CPU inference")
            use_edge_tpu = False

        self.model_loader = ModelLoader(model_path, use_edge_tpu, num_threads)

        # Load YOLO class labels from config so they match the trained model
        labels_config = self.config.get('ai.labels', {})
        labels = {int(k): v for k, v in labels_config.items()} if labels_config else None

        confidence_threshold = self.config.get('ai.confidence_threshold', 0.6)
        self.detector = ObjectDetector(self.model_loader, confidence_threshold, labels=labels)

        # Audio player
        sound_path = self.config.get('audio.sound_path', './sounds/')
        volume = self.config.get('audio.volume', 80)
        self.audio_player = AudioPlayer(sound_path, volume)

        # MQTT
        mqtt_enabled = self.config.get('mqtt.enabled', False)
        mqtt_broker = self.config.get('mqtt.broker', 'localhost')
        mqtt_port = self.config.get('mqtt.port', 1883)
        mqtt_topics = self.config.get('mqtt.topics', {})
        self.mqtt_client = MQTTClient(mqtt_broker, mqtt_port, mqtt_topics, mqtt_enabled)

        # Alert manager
        self.alert_manager = AlertManager(self.config, self.database, self.mqtt_client)

        # State
        self.last_trigger_time = 0
        self.cooldown_seconds = self.config.get('motion.cooldown_seconds', 10)

        # Semaphore: allow at most 2 concurrent detection threads to prevent
        # runaway spawning under sustained motion
        self._detection_semaphore = threading.Semaphore(2)

    def _signal_handler(self, sig, frame):
        logger.info("Shutdown signal received")
        self.stop()
        sys.exit(0)

    def is_within_active_hours(self) -> bool:
        now = datetime.now().time()
        start_time = datetime.strptime(
            self.config.get('system.active_hours.start', '06:00'), '%H:%M'
        ).time()
        end_time = datetime.strptime(
            self.config.get('system.active_hours.end', '20:00'), '%H:%M'
        ).time()

        if start_time <= end_time:
            # Normal daytime window (e.g. 06:00 – 20:00)
            return start_time <= now <= end_time
        else:
            # Overnight window (e.g. 22:00 – 06:00)
            return now >= start_time or now <= end_time

    def cooldown_elapsed(self) -> bool:
        return (time.time() - self.last_trigger_time) > self.cooldown_seconds

    def capture_video_clip(self, duration_seconds: int = 5) -> list:
        frames = []
        frame_rate = self.config.get('system.frame_rate', 5)
        num_frames = duration_seconds * frame_rate

        for _ in range(num_frames):
            frame = self.camera.read()
            if frame is not None:
                frames.append(frame)
            time.sleep(1.0 / frame_rate)

        return frames

    def extract_key_frames(self, video_clip: list, num_frames: int = 5) -> list:
        if len(video_clip) <= num_frames:
            return video_clip

        step = len(video_clip) // num_frames
        return [video_clip[i * step] for i in range(num_frames)]

    def process_detection_event(self, video_clip: list):
        logger.info("Processing detection event...")

        key_frames = self.extract_key_frames(video_clip)

        detections = []
        for frame in key_frames:
            detection = self.detector.infer(frame)
            detections.append(detection)

        final_detection = self.detector.aggregate_detections(detections)

        video_path = self.media_store.save_video_clip(video_clip, final_detection.label)

        self.handle_detection_result(final_detection, video_path)

    def _process_detection_safe(self, video_clip: list):
        """Wrapper that acquires the semaphore so at most 2 threads run at once."""
        with self._detection_semaphore:
            self.process_detection_event(video_clip)

    def handle_detection_result(self, detection, video_path: str):
        if detection.label == "heron":
            logger.warning("HERON DETECTED!")
            self.trigger_deterrent()
            self.alert_manager.create_alert("CRITICAL", detection, video_path)

        elif detection.label == "unknown":
            logger.info("Unknown object detected")
            self.alert_manager.create_alert("WARNING", detection, video_path)

        else:
            logger.info(f"Other object detected: {detection.label}")
            self.database.save_detection(
                label=detection.label,
                confidence=detection.confidence,
                video_path=video_path,
                metadata={'type': 'INFO'}
            )

    def trigger_deterrent(self):
        logger.info("Triggering deterrent sound...")
        self.audio_player.play_random()

        self.mqtt_client.publish('detection', {
            'event': 'deterrent_triggered',
            'timestamp': datetime.now().isoformat()
        })

    def video_pipeline_loop(self):
        logger.info("Starting video pipeline...")
        frame_rate = self.config.get('system.frame_rate', 5)
        frame_delay = 1.0 / frame_rate

        while self.running:
            try:
                if not self.is_within_active_hours():
                    time.sleep(1)
                    continue

                frame = self.camera.read()
                if frame is None:
                    logger.warning("Failed to read frame")
                    time.sleep(1)
                    continue

                motion_detected = self.motion_detector.detect(frame)

                if motion_detected and self.cooldown_elapsed():
                    logger.info("Motion detected - capturing video clip")
                    self.last_trigger_time = time.time()

                    video_clip = self.capture_video_clip(duration_seconds=5)

                    threading.Thread(
                        target=self._process_detection_safe,
                        args=(video_clip,),
                        daemon=True
                    ).start()

                time.sleep(frame_delay)

            except Exception as e:
                logger.error(f"Error in video pipeline: {e}")
                time.sleep(1)

    def monitor_system_health(self):
        logger.info("Starting system health monitor...")

        while self.running:
            try:
                if not self.camera.is_connected():
                    logger.warning("Camera disconnected, attempting restart...")
                    self.camera.restart()

                media_path = Path(self.config.get('storage.media_path', './media/'))
                if media_path.exists():
                    stat = os.statvfs(media_path)
                    free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
                    if free_gb < 1.0:
                        logger.warning(f"Low disk space: {free_gb:.2f} GB remaining")

                self.mqtt_client.publish('status', {
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat()
                })

                time.sleep(30)

            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                time.sleep(30)

    def cleanup_old_data(self):
        logger.info("Starting cleanup job...")
        retention_days = self.config.get('storage.retention_days', 365)

        while self.running:
            try:
                logger.info("Running cleanup...")
                self.database.cleanup_old_records(retention_days)
                self.media_store.cleanup_old_media(retention_days)
                time.sleep(86400)  # daily
            except Exception as e:
                logger.error(f"Error in cleanup: {e}")
                time.sleep(3600)

    def start(self):
        logger.info("Starting Heron Deterrent System...")
        self.running = True

        video_thread  = threading.Thread(target=self.video_pipeline_loop,  daemon=True)
        health_thread = threading.Thread(target=self.monitor_system_health, daemon=True)
        cleanup_thread = threading.Thread(target=self.cleanup_old_data,     daemon=True)

        video_thread.start()
        health_thread.start()
        cleanup_thread.start()

        logger.info("System started successfully")

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        logger.info("Stopping system...")
        self.running = False

        self.camera.release()
        self.mqtt_client.disconnect()
        self.database.close()

        logger.info("System stopped")

def main():
    system = HeronDeterrentSystem()

    if os.getenv('START_WEB_UI', 'false').lower() == 'true':
        from ui.app import start_web_ui
        threading.Thread(target=start_web_ui, args=(system,), daemon=True).start()

    system.start()

if __name__ == "__main__":
    main()
