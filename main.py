# ============================================================
# FILE: main.py
# ============================================================

import logging
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
import cv2
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('heron_deterrent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HeronDeterrentSystem:
    def __init__(self, config_path: str = "config.yaml"):
        logger.info("Initializing Heron Deterrent System...")
        
        # Load configuration
        self.config = Config(config_path)
        self.running = False
        
        # Initialize components
        self._initialize_components()
        
        # Setup signal handlers
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
        
        # Adjust for Raspberry Pi
        if self.config.get('system.running_on_raspberry_pi', False):
            logger.info("Running on Raspberry Pi - optimizing settings")
            use_edge_tpu = False  # Use USB TPU if available
        
        self.model_loader = ModelLoader(model_path, use_edge_tpu)
        
        confidence_threshold = self.config.get('ai.confidence_threshold', 0.6)
        self.detector = ObjectDetector(self.model_loader, confidence_threshold)
        
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
        self.video_buffer = []
    
    def _signal_handler(self, sig, frame):
        logger.info("Shutdown signal received")
        self.stop()
        sys.exit(0)
    
    def is_within_active_hours(self) -> bool:
        now = datetime.now().time()
        start_time = datetime.strptime(
            self.config.get('system.active_hours.start', '06:00'), 
            '%H:%M'
        ).time()
        end_time = datetime.strptime(
            self.config.get('system.active_hours.end', '20:00'), 
            '%H:%M'
        ).time()
        
        return start_time <= now <= end_time
    
    def cooldown_elapsed(self) -> bool:
        return (time.time() - self.last_trigger_time) > self.cooldown_seconds
    
    def capture_video_clip(self, duration_seconds: int = 5) -> list:
        frames = []
        frame_rate = self.config.get('system.frame_rate', 10)
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
        
        # Extract key frames
        key_frames = self.extract_key_frames(video_clip)
        
        # Run detection on each frame
        detections = []
        for frame in key_frames:
            detection = self.detector.infer(frame)
            detections.append(detection)
        
        # Aggregate results
        final_detection = self.detector.aggregate_detections(detections)
        
        # Save video
        video_path = self.media_store.save_video_clip(video_clip, final_detection.label)
        
        # Handle detection result
        self.handle_detection_result(final_detection, video_path)
    
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
            # Just log, don't alert
            self.database.save_detection(
                label=detection.label,
                confidence=detection.confidence,
                video_path=video_path,
                metadata={'type': 'INFO'}
            )
    
    def trigger_deterrent(self):
        logger.info("Triggering deterrent sound...")
        self.audio_player.play_random()
        
        # Publish MQTT event
        self.mqtt_client.publish('detection', {
            'event': 'deterrent_triggered',
            'timestamp': datetime.now().isoformat()
        })
    
    def video_pipeline_loop(self):
        logger.info("Starting video pipeline...")
        frame_rate = self.config.get('system.frame_rate', 10)
        frame_delay = 1.0 / frame_rate
        
        while self.running:
            try:
                # Check active hours
                if not self.is_within_active_hours():
                    time.sleep(1)
                    continue
                
                # Read frame
                frame = self.camera.read()
                if frame is None:
                    logger.warning("Failed to read frame")
                    time.sleep(1)
                    continue
                
                # Detect motion
                motion_detected = self.motion_detector.detect(frame)
                
                if motion_detected and self.cooldown_elapsed():
                    logger.info("Motion detected - capturing video clip")
                    self.last_trigger_time = time.time()
                    
                    # Capture video clip
                    video_clip = self.capture_video_clip(duration_seconds=5)
                    
                    # Process in separate thread to avoid blocking
                    threading.Thread(
                        target=self.process_detection_event,
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
                # Check camera
                if not self.camera.is_connected():
                    logger.warning("Camera disconnected, attempting restart...")
                    self.camera.restart()
                
                # Check disk space
                media_path = Path(self.config.get('storage.media_path', './media/'))
                if media_path.exists():
                    stat = os.statvfs(media_path)
                    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
                    if free_gb < 1.0:
                        logger.warning(f"Low disk space: {free_gb:.2f} GB remaining")
                
                # Publish status
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
                
                # Run daily
                time.sleep(86400)
                
            except Exception as e:
                logger.error(f"Error in cleanup: {e}")
                time.sleep(3600)
    
    def start(self):
        logger.info("Starting Heron Deterrent System...")
        self.running = True
        
        # Start threads
        video_thread = threading.Thread(target=self.video_pipeline_loop, daemon=True)
        health_thread = threading.Thread(target=self.monitor_system_health, daemon=True)
        cleanup_thread = threading.Thread(target=self.cleanup_old_data, daemon=True)
        
        video_thread.start()
        health_thread.start()
        cleanup_thread.start()
        
        logger.info("System started successfully")
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        logger.info("Stopping system...")
        self.running = False
        
        # Cleanup
        self.camera.release()
        self.mqtt_client.disconnect()
        self.database.close()
        
        logger.info("System stopped")

def main():
    import os
    system = HeronDeterrentSystem()
    
    # Start web UI in separate process if requested
    if os.getenv('START_WEB_UI', 'false').lower() == 'true':
        from ui.app import start_web_ui
        threading.Thread(target=start_web_ui, args=(system,), daemon=True).start()
    
    # Start main system
    system.start()

if __name__ == "__main__":
    main()