# ============================================================
# FILE: alerts/alert_manager.py
# ============================================================

import logging
import os
from datetime import datetime, timedelta
from twilio.rest import Client as TwilioClient
from twilio.base.exceptions import TwilioRestException
from twilio.http.http_client import TwilioHttpClient

logger = logging.getLogger(__name__)

# Twilio's helper uses requests with no default timeout, so a flaky network can
# block the caller indefinitely. The deterrent already fired by the time we get
# here, but an unbounded wait would still tie up a detection slot.
SMS_TIMEOUT_SECONDS = 10

class AlertManager:
    def __init__(self, config, database, mqtt_client):
        self.config = config
        self.database = database
        self.mqtt_client = mqtt_client
        self.twilio_client = None

        self.sms_enabled = config.get('alert.sms_enabled', False)
        self.sms_throttle_minutes = config.get('alert.sms_throttle_minutes', 15)

        if self.sms_enabled:
            # Environment variables are the only credential mechanism. There is
            # deliberately no config.yaml fallback, so credentials cannot be
            # committed by accident.
            env = {name: os.getenv(name) for name in (
                'TWILIO_ACCOUNT_SID', 'TWILIO_AUTH_TOKEN',
                'TWILIO_FROM_NUMBER', 'TWILIO_TO_NUMBER',
            )}
            missing = [name for name, value in env.items() if not value]

            if missing:
                # Warn rather than exit: losing SMS is bad, losing the deterrent
                # because SMS is misconfigured is worse. preflight.py fails hard on this.
                logger.warning(
                    f"alert.sms_enabled is true but these environment variables are unset: "
                    f"{', '.join(missing)}. SMS is disabled for this run; "
                    f"detection and the deterrent are unaffected. "
                    f"Put them in /etc/heron-deterrent.env (mode 600, root-owned), then: "
                    f"sudo systemctl restart heron-deterrent"
                )
                self.sms_enabled = False
            else:
                self.from_number = env['TWILIO_FROM_NUMBER']
                self.to_number = env['TWILIO_TO_NUMBER']
                try:
                    self.twilio_client = TwilioClient(
                        env['TWILIO_ACCOUNT_SID'], env['TWILIO_AUTH_TOKEN'],
                        http_client=TwilioHttpClient(timeout=SMS_TIMEOUT_SECONDS),
                    )
                    logger.info(
                        f"Twilio SMS client initialized (to {self.to_number}, "
                        f"{SMS_TIMEOUT_SECONDS}s timeout)"
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize Twilio: {e}")
                    self.sms_enabled = False

    def create_alert(self, alert_type: str, detection, video_path: str = "", *,
                     frame_type: str, metadata: dict = None):
        message = f"{alert_type}: {detection.label} detected (confidence: {detection.confidence:.2f})"

        detection_id = self.database.save_detection(
            label=detection.label,
            confidence=detection.confidence,
            video_path=video_path,
            metadata={'type': alert_type, **(metadata or {})},
            bbox=detection.bbox,
            frame_type=frame_type,
        )

        alert_id = self.database.save_alert(
            alert_type=alert_type,
            detection_id=detection_id,
            message=message
        )

        logger.info(f"Alert created: {message}")

        if alert_type == "CRITICAL" and self.sms_enabled:
            self._send_sms(alert_id, message)

        self.mqtt_client.publish('alert', {
            'type': alert_type,
            'label': detection.label,
            'confidence': detection.confidence,
            'timestamp': datetime.now().isoformat()
        })

    def create_motion_alert(self, frame_type: str, video_path: str, metadata: dict):
        """A motion trigger no model classified: night, or an unsettled frame type.

        Stored as label 'motion' with NULL confidence — there is no score, and
        writing 0.0 would read as "the model looked and saw nothing". The
        message says unclassified in words so nobody reads it as a heron. Never
        sends SMS: an unclassified trigger is not worth waking anyone for.
        """
        message = f"Motion ({frame_type}) — unclassified, no model ran"

        detection_id = self.database.save_detection(
            label='motion',
            confidence=None,
            video_path=video_path,
            metadata={'type': 'MOTION', **metadata},
            frame_type=frame_type,
        )
        self.database.save_alert(alert_type='MOTION', detection_id=detection_id,
                                 message=message)
        logger.info(f"Alert created: {message}")

        self.mqtt_client.publish('unclassified', {
            'type': 'MOTION',
            'frame_type': frame_type,
            'timestamp': datetime.now().isoformat()
        })
        return detection_id

    def _send_sms(self, alert_id: int, message: str):
        last_sms_time = self.database.get_last_sms_time()
        if last_sms_time:
            time_since_last = datetime.now() - last_sms_time
            if time_since_last < timedelta(minutes=self.sms_throttle_minutes):
                logger.info("SMS throttled — too soon since last message")
                return

        try:
            self.twilio_client.messages.create(
                body=message,
                from_=self.from_number,
                to=self.to_number
            )
            self.database.log_sms(alert_id, success=True)
            logger.info(f"SMS sent successfully to {self.to_number}")
        except TwilioRestException as e:
            logger.error(f"Failed to send SMS: {e}")
            self.database.log_sms(alert_id, success=False)
        except Exception as e:
            logger.error(f"SMS error: {e}")
            self.database.log_sms(alert_id, success=False)
