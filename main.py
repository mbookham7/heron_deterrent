# ============================================================
# FILE: main.py
# ============================================================

import logging
import logging.handlers
import os
import time
import threading
from datetime import datetime
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

# Retention and the media size cap are checked this often. Frequent enough that
# a burst of triggers cannot fill the card between passes, cheap enough to ignore.
CLEANUP_INTERVAL_SECONDS = 300


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

        # Media storage — clips are written at the capture rate so they play
        # back at real speed rather than double time.
        media_path = self.config.get('storage.media_path', './media/')
        self.media_store = MediaStore(media_path, fps=self.config.get('system.frame_rate', 5))

        # Config-driven components first: a bad model path or an empty sounds/
        # should be reported before we start grabbing hardware, so that a camera
        # error cannot mask a configuration error.

        # AI model — ai.target selects both the model file and the backend
        target = self.config.get('ai.target', 'cpu')
        num_threads = self.config.get('ai.num_threads', 4)
        self.model_loader = ModelLoader(target, num_threads)

        # Load YOLO class labels from config so they match the trained model
        labels_config = self.config.get('ai.labels', {})
        labels = {int(k): v for k, v in labels_config.items()} if labels_config else None

        # Two thresholds now. The detector's internal anchor filter uses the
        # LOWEST relevant threshold so every decision path sees the boxes it
        # needs. Config falls back to the old single `confidence_threshold` if
        # the split keys are absent, so an older config file still boots.
        fallback = self.config.get('ai.confidence_threshold', 0.90)
        self.deter_threshold = self.config.get('ai.deter_threshold', fallback)
        self.alert_threshold = self.config.get('ai.alert_threshold', fallback)
        self.alert_vote_k = self.config.get('ai.alert_vote_k', 3)
        self.alert_vote_m = self.config.get('ai.alert_vote_m', 5)

        # Collect mode: record any motion event above record_threshold, and
        # optionally silence the deterrent. Off in production.
        self.collect_mode_enabled = self.config.get('collect_mode.enabled', False)
        self.record_threshold = self.config.get('collect_mode.record_threshold', 0.30)
        self.deterrent_enabled = self.config.get('collect_mode.deterrent_enabled', True)

        # Anchor floor: pull the detector's filter down to whichever number
        # actually needs boxes. Collect mode needs boxes at record_threshold or
        # nothing gets recorded; alert path needs alert_threshold; deter path
        # needs deter_threshold. Pick the minimum.
        anchor_floor = min(self.deter_threshold, self.alert_threshold)
        if self.collect_mode_enabled:
            anchor_floor = min(anchor_floor, self.record_threshold)
        self.detector = ObjectDetector(self.model_loader, anchor_floor, labels=labels)
        logger.info(
            f"Thresholds: deter={self.deter_threshold}  alert={self.alert_threshold}  "
            f"vote={self.alert_vote_k}/{self.alert_vote_m}  "
            f"anchor_floor={anchor_floor}"
        )
        if self.collect_mode_enabled:
            logger.warning(
                f"COLLECT MODE ON: recording everything above {self.record_threshold}, "
                f"deterrent {'enabled' if self.deterrent_enabled else 'DISABLED'}"
            )

        # Audio player — raises if sounds/ is empty
        sound_path = self.config.get('audio.sound_path', './sounds/')
        volume = self.config.get('audio.volume', 80)
        self.audio_player = AudioPlayer(sound_path, volume)

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
        """Aggregate over key frames using N-of-M voting on the ALERT threshold.

        Used by the dashboard's test-upload path and by the fallback pass when
        the trigger frame was inconclusive. The deter path never touches this
        function — it fires on the trigger frame directly, so its latency and
        false-positive profile are unaffected by anything here.
        """
        logger.info("Processing detection event...")

        key_frames = self.extract_key_frames(video_clip, num_frames=self.alert_vote_m)

        detections = []
        for frame in key_frames:
            detection = self.detector.infer(frame)
            detections.append(detection)

        final_detection = self.detector.aggregate_detections(
            detections,
            mode="vote",
            alert_threshold=self.alert_threshold,
            vote_k=self.alert_vote_k,
        )

        video_path = self.media_store.save_video_clip(video_clip, final_detection.label)

        self.handle_detection_result(final_detection, video_path)

    def _handle_motion_event(self, trigger_frame):
        """Decide on the frame we already have, then record the clip.

        The old order captured five seconds of video before running any
        inference, so the deterrent fired ~9.6s after the bird appeared. Judging
        the trigger frame first cuts that to one inference (~0.9s on this board)
        and, because a single frame has fewer chances to trip, it also produces
        fewer false positives than aggregating over five.
        """
        try:
            detection = self.detector.infer(trigger_frame)

            # The detector's anchor filter runs at min(deter, alert), so
            # detection.label == 'heron' is necessary but not sufficient — gate
            # the deter path on deter_threshold explicitly. Anything below deter
            # falls through to the full-clip pass (which uses N-of-M voting on
            # alert_threshold) so a genuinely-there bird still gets its clip
            # recorded and its alert raised.
            if detection.label == "heron" and detection.confidence >= self.deter_threshold:
                logger.warning(
                    f"HERON DETECTED on trigger frame (conf={detection.confidence:.3f} "
                    f">= deter={self.deter_threshold})"
                )
                self.trigger_deterrent()          # first, before anything slow
                video_clip = self.capture_video_clip(duration_seconds=5)
                video_path = self.media_store.save_video_clip(video_clip, detection.label)
                self.alert_manager.create_alert("CRITICAL", detection, video_path)
                return

            # Inconclusive OR under deter threshold: fall back to the thorough
            # multi-frame pass with voting on the alert threshold.
            logger.info(
                f"Trigger frame inconclusive (label={detection.label} "
                f"conf={detection.confidence:.3f}) — capturing clip for voting pass"
            )
            video_clip = self.capture_video_clip(duration_seconds=5)
            self.process_detection_event(video_clip)

        except Exception as e:
            logger.error(f"Error handling motion event: {e}", exc_info=True)
        finally:
            self._detection_semaphore.release()

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
                metadata={'type': 'INFO'},
                bbox=detection.bbox,
            )

    def trigger_deterrent(self):
        """Fire the deterrent sound, unless collect mode has silenced it.

        Silenced runs still log and publish the "would have fired" event so a
        dashboard operator can see the counterfactual — did you disable the
        deterrent and now nothing is being scared off, or is the model just
        not firing?
        """
        if self.collect_mode_enabled and not self.deterrent_enabled:
            logger.info("Deterrent SILENCED (collect_mode.deterrent_enabled=false)")
            self.mqtt_client.publish('detection', {
                'event': 'deterrent_silenced',
                'timestamp': datetime.now().isoformat(),
            })
            return

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
                    # Camera tracks consecutive failures itself and the health
                    # monitor reconnects; just back off so we do not spin.
                    time.sleep(1)
                    continue

                motion_detected = self.motion_detector.detect(frame)

                if motion_detected and self.cooldown_elapsed():
                    # Drop rather than queue. Each queued event would hold ~22 MB
                    # of frames, and this board has under 700 MB free, so piling
                    # them up behind a slow alert is how it runs out of memory.
                    if not self._detection_semaphore.acquire(blocking=False):
                        logger.warning("Detection slots busy — dropping this motion event")
                        time.sleep(frame_delay)
                        continue

                    logger.info("Motion detected - inferring on trigger frame")
                    self.last_trigger_time = time.time()

                    threading.Thread(
                        target=self._handle_motion_event,
                        args=(frame,),
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
                    # restart() applies its own exponential backoff, so a camera
                    # that is genuinely gone does not spin the log.
                    self.camera.restart()

                media_path = Path(self.config.get('storage.media_path', './media/'))
                if media_path.exists():
                    stat = os.statvfs(media_path)
                    free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
                    if free_gb < 1.0:
                        logger.warning(
                            f"Low disk space: {free_gb:.2f} GB free, "
                            f"media using {self.media_store.total_bytes() / 1024 ** 3:.2f} GB"
                        )

                self.mqtt_client.publish('status', {
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat()
                })

                time.sleep(30)

            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                time.sleep(30)

    def cleanup_old_data(self):
        """Retention plus a hard size cap.

        Runs every few minutes rather than daily. Retention alone cannot protect
        the card: at 100 events/day a 2 GB partition fills in about four months,
        long before a 365-day cutoff deletes anything, and a daily job could
        sleep through the filling.
        """
        logger.info("Starting cleanup job...")
        retention_days = self.config.get('storage.retention_days', 365)
        max_media_gb = self.config.get('storage.max_media_gb', 1.0)
        max_labeled_gb = self.config.get('storage.max_labeled_gb', 5.0)

        while self.running:
            try:
                self.database.cleanup_old_records(retention_days)
                self.media_store.cleanup_old_media(retention_days)
                self.media_store.enforce_size_cap(max_media_gb)
                # Curated training data is never auto-deleted, only reported.
                self.media_store.warn_labeled_size(max_labeled_gb)
                time.sleep(CLEANUP_INTERVAL_SECONDS)
            except Exception as e:
                logger.error(f"Error in cleanup: {e}")
                time.sleep(CLEANUP_INTERVAL_SECONDS)

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
    try:
        system = HeronDeterrentSystem()
    except RuntimeError as e:
        # Missing model, missing sounds, label/model mismatch, no camera. These are
        # operator-fixable, so print the instruction rather than a traceback.
        print(f"\nSTARTUP FAILED\n\n{e}\n", file=sys.stderr)
        print("Run ./preflight.py to check every dependency at once.\n", file=sys.stderr)
        sys.exit(2)

    if os.getenv('START_WEB_UI', 'false').lower() == 'true':
        from ui.app import start_web_ui
        threading.Thread(target=start_web_ui, args=(system,), daemon=True).start()

    system.start()

if __name__ == "__main__":
    main()
