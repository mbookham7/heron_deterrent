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
from storage.video_encoder import require_encoder
from capture.camera import Camera
from capture.motion_detector import MotionDetector
from capture.frame_classifier import FrameTypeTracker, classify_frame, clip_frame_types
from ai.model_loader import ModelLoader
from ai.detector import ObjectDetector
from utils.modes import load_modes, load_frame_type_config
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

# Shadow inference logs the model's best score whatever it is, so its anchor
# filter sits near zero. The number is only ever logged, never compared.
SHADOW_ANCHOR_FLOOR = 0.01


def anchor_floor(mode, collect_enabled: bool) -> float:
    """The detector's anchor filter for `mode`: the LOWEST threshold any
    decision path needs boxes at. Collect mode needs boxes at record_threshold
    or nothing borderline gets recorded."""
    floor = min(mode.deter_threshold, mode.alert_threshold)
    if collect_enabled and mode.record_threshold is not None:
        floor = min(floor, mode.record_threshold)
    return floor


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
        #
        # Check the video encoder BEFORE anything else touches the disk. Clips
        # were previously written as MPEG-4 Part 2, which every browser parses
        # and no browser can decode — the dashboard showed playback controls
        # over a black frame and reported nothing wrong. Refusing to start is
        # deliberate: a recorder that silently produces unplayable footage is
        # worse than one that does not run, because the loss is only discovered
        # when you go looking for the footage.
        encoder = require_encoder()
        logger.info(f"Video encoder: {encoder}")

        media_path = self.config.get('storage.media_path', './media/')
        self.media_store = MediaStore(media_path, fps=self.config.get('system.frame_rate', 5))

        # Config-driven components first: a bad model path or an empty sounds/
        # should be reported before we start grabbing hardware, so that a camera
        # error cannot mask a configuration error.

        # Modes: one per frame type (day / ir / dark). Validated as a whole
        # before anything is loaded, so every config problem is reported at
        # once and before any hardware is touched.
        self.collect_mode_enabled = self.config.get('collect_mode.enabled', False)
        self.modes = load_modes(self.config)
        self.frame_tracker = FrameTypeTracker(load_frame_type_config(self.config))
        for mode in self.modes.values():
            logger.info(f"Mode {mode.describe()}")

        self.alert_vote_k = self.config.get('ai.alert_vote_k', 3)
        self.alert_vote_m = self.config.get('ai.alert_vote_m', 5)

        # AI models — ai.target selects the backend for every model. Each
        # registered model is loaded once and shared by any mode naming it; a
        # mode with model: null loads nothing.
        target = self.config.get('ai.target', 'cpu')
        num_threads = self.config.get('ai.num_threads', 4)
        labels_config = self.config.get('ai.labels', {})
        labels = {int(k): v for k, v in labels_config.items()} if labels_config else None

        self._loaders = {}

        def loader_for(name):
            if name not in self._loaders:
                self._loaders[name] = ModelLoader(target, num_threads, model=name)
            return self._loaders[name]

        self.detectors = {}          # mode name -> detector it ACTS on
        self.shadow_detectors = {}   # mode name -> detector it only LOGS
        for mode in self.modes.values():
            if mode.model:
                floor = anchor_floor(mode, self.collect_mode_enabled)
                self.detectors[mode.name] = ObjectDetector(
                    loader_for(mode.model), floor, labels=labels)
                logger.info(f"{mode.name}: vote={self.alert_vote_k}/{self.alert_vote_m} "
                            f"anchor_floor={floor}")
            if mode.shadow_model:
                self.shadow_detectors[mode.name] = ObjectDetector(
                    loader_for(mode.shadow_model), SHADOW_ANCHOR_FLOOR, labels=labels)
                logger.warning(f"{mode.name}: SHADOW inference on — {mode.shadow_model} "
                               f"confidence is logged per trigger and never acted on")

        if self.collect_mode_enabled:
            day = self.modes['day']
            logger.warning(f"COLLECT MODE ON: recording day events above {day.record_threshold}")

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

        # State. Each mode has its own cooldown; motion.cooldown_seconds applies
        # only while the frame type is establishing or switching.
        self.last_trigger_time = 0
        self.unsettled_cooldown_seconds = self.config.get('motion.cooldown_seconds', 10)

        # Semaphore: allow at most 2 concurrent detection threads to prevent
        # runaway spawning under sustained motion
        self._detection_semaphore = threading.Semaphore(2)

    def _signal_handler(self, sig, frame):
        logger.info("Shutdown signal received")
        self.stop()
        sys.exit(0)

    def frame_state(self) -> str:
        """day / ir / dark, or establishing / switching."""
        return self.frame_tracker.state

    def cooldown_for(self, state: str) -> float:
        mode = self.modes.get(state)
        return mode.cooldown_seconds if mode else self.unsettled_cooldown_seconds

    def cooldown_elapsed(self, state: str = None) -> bool:
        state = state or self.frame_state()
        return (time.time() - self.last_trigger_time) > self.cooldown_for(state)

    def _capture_clip_with_types(self, metadata: dict) -> list:
        """Capture the 5 s clip and record every frame's type, classified from
        the live frames, into `metadata['clip_frame_types']`. Labelling relies
        on this record; decoded clip frames cannot be re-classified reliably."""
        clip = self.capture_video_clip(duration_seconds=5)
        metadata['clip_frame_types'] = clip_frame_types(clip, self.frame_tracker.cfg)
        return clip

    def classify_clip(self, frames: list):
        """Frame type of a standalone clip (e.g. an upload): the type every key
        frame agrees on, or None if they disagree or any is ambiguous. No
        hysteresis — there is no stream to smooth over, so demand unanimity."""
        cfg = self.frame_tracker.cfg
        types = {classify_frame(f, cfg) for f in self.extract_key_frames(frames, 5)}
        return types.pop() if len(types) == 1 else None

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

    def process_detection_event(self, video_clip: list, mode, metadata: dict = None):
        """Aggregate over key frames using N-of-M voting on the ALERT threshold.

        Used by the dashboard's test-upload path and by the fallback pass when
        the trigger frame was inconclusive. The deter path never touches this
        function — it fires on the trigger frame directly, so its latency and
        false-positive profile are unaffected by anything here.

        `mode` must be one with a model; motion-only modes never get here.
        """
        logger.info("Processing detection event...")
        detector = self.detectors[mode.name]

        key_frames = self.extract_key_frames(video_clip, num_frames=self.alert_vote_m)

        detections = []
        for frame in key_frames:
            detection = detector.infer(frame)
            detections.append(detection)

        final_detection = detector.aggregate_detections(
            detections,
            mode="vote",
            alert_threshold=mode.alert_threshold,
            vote_k=self.alert_vote_k,
        )

        video_path = self.media_store.save_video_clip(
            video_clip, final_detection.label, frame_type=mode.name)

        self.handle_detection_result(final_detection, video_path, mode, metadata)

    def _handle_motion_event(self, trigger_frame, snapshot):
        """Branch on the frame type at the moment of the trigger.

        `snapshot` is the FrameTypeTracker's view of the trigger frame. If its
        state is a mode with a model, run that model. Otherwise — a
        motion-only mode, or establishing/switching where nobody knows which
        model applies — record the clip unclassified. The clip is ALWAYS kept:
        dawn is both peak heron time and when the camera cuts to and from IR,
        so dropping unsettled triggers would lose exactly the footage that
        matters most.
        """
        try:
            mode = self.modes.get(snapshot.state)
            if mode is None or mode.motion_only:
                self._record_motion_only(trigger_frame, snapshot, mode)
                return
            self._classify_and_act(trigger_frame, snapshot, mode)

        except Exception as e:
            logger.error(f"Error handling motion event: {e}", exc_info=True)
        finally:
            self._detection_semaphore.release()

    def _classify_and_act(self, trigger_frame, snapshot, mode):
        """Decide on the frame we already have, then record the clip.

        The old order captured five seconds of video before running any
        inference, so the deterrent fired ~9.6s after the bird appeared. Judging
        the trigger frame first cuts that to one inference (~0.9s on this board)
        and, because a single frame has fewer chances to trip, it also produces
        fewer false positives than aggregating over five.
        """
        metadata = snapshot.as_metadata()
        detection = self.detectors[mode.name].infer(trigger_frame)

        # The detector's anchor filter runs at min(deter, alert), so
        # detection.label == 'heron' is necessary but not sufficient — gate
        # the deter path on deter_threshold explicitly. Anything below deter
        # falls through to the full-clip pass (which uses N-of-M voting on
        # alert_threshold) so a genuinely-there bird still gets its clip
        # recorded and its alert raised.
        if detection.label == "heron" and detection.confidence >= mode.deter_threshold:
            logger.warning(
                f"HERON DETECTED on trigger frame (conf={detection.confidence:.3f} "
                f">= deter={mode.deter_threshold}, mode={mode.name})"
            )
            self.trigger_deterrent(mode)          # first, before anything slow
            video_clip = self._capture_clip_with_types(metadata)
            video_path = self.media_store.save_video_clip(
                video_clip, detection.label, frame_type=mode.name)
            self.alert_manager.create_alert("CRITICAL", detection, video_path,
                                            frame_type=mode.name, metadata=metadata)
            return

        # Inconclusive OR under deter threshold: fall back to the thorough
        # multi-frame pass with voting on the alert threshold.
        logger.info(
            f"Trigger frame inconclusive (label={detection.label} "
            f"conf={detection.confidence:.3f}) — capturing clip for voting pass"
        )
        video_clip = self._capture_clip_with_types(metadata)
        self.process_detection_event(video_clip, mode, metadata)

    def _record_motion_only(self, trigger_frame, snapshot, mode):
        """Record a clip and raise an unclassified alert. No model decides
        anything here and the deterrent is never touched.

        Optional shadow inference runs AFTER the clip is captured, so it cannot
        delay the recording, and its score goes only into metadata.
        """
        reason = 'motion-only mode' if mode else f'frame type {snapshot.state}'
        logger.info(f"Motion ({reason}, raw={snapshot.raw}) — recording unclassified")
        metadata = {'trigger': 'motion_only', **snapshot.as_metadata()}

        video_clip = self._capture_clip_with_types(metadata)

        shadow = self.shadow_detectors.get(mode.name) if mode else None
        if shadow is not None:
            d = shadow.infer(trigger_frame)
            metadata['shadow_model'] = mode.shadow_model
            metadata['shadow_label'] = d.label
            metadata['shadow_confidence'] = round(float(d.confidence), 4)
            logger.info(f"SHADOW {mode.shadow_model} on {mode.name} frame: "
                        f"{d.label} {d.confidence:.3f} (logged only, not acted on)")

        video_path = self.media_store.save_video_clip(
            video_clip, 'motion', frame_type=snapshot.state)
        self.alert_manager.create_motion_alert(snapshot.state, video_path, metadata)

    def handle_detection_result(self, detection, video_path: str, mode, metadata: dict = None):
        if detection.label == "heron":
            logger.warning("HERON DETECTED!")
            self.trigger_deterrent(mode)
            self.alert_manager.create_alert("CRITICAL", detection, video_path,
                                            frame_type=mode.name, metadata=metadata)

        elif detection.label == "unknown":
            logger.info("Unknown object detected")
            self.alert_manager.create_alert("WARNING", detection, video_path,
                                            frame_type=mode.name, metadata=metadata)

        else:
            logger.info(f"Other object detected: {detection.label}")
            self.database.save_detection(
                label=detection.label,
                confidence=detection.confidence,
                video_path=video_path,
                metadata={'type': 'INFO', **(metadata or {})},
                bbox=detection.bbox,
                frame_type=mode.name,
            )

    def trigger_deterrent(self, mode):
        """Fire the deterrent sound if `mode` allows it.

        modes.<mode>.deterrent is the ONE switch for the sound — collect mode
        no longer has a say. Silenced runs still log and publish the "would
        have fired" event so a dashboard operator can see the counterfactual —
        did you disable the deterrent and now nothing is being scared off, or
        is the model just not firing? Recording and alerting happen in the
        caller either way.
        """
        if not mode.deterrent:
            logger.info(f"Deterrent SILENCED (modes.{mode.name}.deterrent=false)")
            self.mqtt_client.publish('detection', {
                'event': 'deterrent_silenced',
                'mode': mode.name,
                'timestamp': datetime.now().isoformat(),
            })
            return

        logger.info("Triggering deterrent sound...")
        self.audio_player.play_random()

        self.mqtt_client.publish('detection', {
            'event': 'deterrent_triggered',
            'mode': mode.name,
            'timestamp': datetime.now().isoformat()
        })

    def play_deterrent_manually(self):
        """The dashboard's "test speaker" button. A deliberate operator action,
        so it plays regardless of mode — exactly as preflight's audio check
        does. Nothing automatic calls this."""
        logger.info("Deterrent played manually from the dashboard")
        self.audio_player.play_random()

    def video_pipeline_loop(self):
        logger.info("Starting video pipeline...")
        frame_rate = self.config.get('system.frame_rate', 5)
        frame_delay = 1.0 / frame_rate

        # Round the clock. There is no time-of-day window: the camera decides
        # when it is night by cutting to IR, and the frame tracker reads that
        # off the pixels.
        while self.running:
            try:
                frame = self.camera.read()
                if frame is None:
                    # Camera tracks consecutive failures itself and the health
                    # monitor reconnects; just back off so we do not spin.
                    time.sleep(1)
                    continue

                snapshot, switched = self.frame_tracker.update(frame)
                if switched:
                    logger.warning(f"Mode is now {self.modes[snapshot.state].describe()}")
                    # Every pixel changed at the cut; the old background would
                    # read the new scene as motion for ~100 s.
                    self.motion_detector.reset()

                motion_detected = self.motion_detector.detect(frame)

                if motion_detected and self.cooldown_elapsed(snapshot.state):
                    # Drop rather than queue. Each queued event would hold ~22 MB
                    # of frames, and this board has under 700 MB free, so piling
                    # them up behind a slow alert is how it runs out of memory.
                    if not self._detection_semaphore.acquire(blocking=False):
                        logger.warning("Detection slots busy — dropping this motion event")
                        time.sleep(frame_delay)
                        continue

                    logger.info(f"Motion detected ({snapshot.state})")
                    self.last_trigger_time = time.time()

                    threading.Thread(
                        target=self._handle_motion_event,
                        args=(frame, snapshot),
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
        max_motion_gb = self.config.get('storage.max_motion_gb', 1.0)
        max_labeled_gb = self.config.get('storage.max_labeled_gb', 5.0)

        while self.running:
            try:
                self.database.cleanup_old_records(retention_days)
                self.media_store.cleanup_old_media(retention_days)
                # Separate caps: a noisy night can only evict night clips.
                self.media_store.enforce_size_cap(max_media_gb, pool='classified')
                self.media_store.enforce_size_cap(max_motion_gb, pool='motion')
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
