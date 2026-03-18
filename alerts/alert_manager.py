# ============================================================
# FILE: alerts/alert_manager.py
# ============================================================

import logging
import os
from datetime import datetime, timedelta
from typing import Optional
from twilio.rest import Client as TwilioClient
from twilio.base.exceptions import TwilioRestException

logger = logging.getLogger(__name__)

class AlertManager:
    def __init__(self, config, database, mqtt_client):
        self.config = config
        self.database = database
        self.mqtt_client = mqtt_client
        self.twilio_client = None

        self.sms_enabled = config.get('alert.sms_enabled', False)
        self.sms_throttle_minutes = config.get('alert.sms_throttle_minutes', 15)

        if self.sms_enabled:
            # Prefer environment variables; fall back to config file values
            account_sid = os.getenv('TWILIO_ACCOUNT_SID') or config.get('alert.twilio_account_sid')
            auth_token  = os.getenv('TWILIO_AUTH_TOKEN')  or config.get('alert.twilio_auth_token')
            self.from_number = os.getenv('TWILIO_FROM_NUMBER') or config.get('alert.twilio_from_number')
            self.to_number   = os.getenv('TWILIO_TO_NUMBER')   or config.get('alert.twilio_to_number')

            if account_sid and auth_token:
                try:
                    self.twilio_client = TwilioClient(account_sid, auth_token)
                    logger.info("Twilio SMS client initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize Twilio: {e}")
                    self.sms_enabled = False
            else:
                logger.warning(
                    "Twilio credentials not configured. "
                    "Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN environment variables."
                )
                self.sms_enabled = False

    def create_alert(self, alert_type: str, detection, video_path: str = ""):
        message = f"{alert_type}: {detection.label} detected (confidence: {detection.confidence:.2f})"

        detection_id = self.database.save_detection(
            label=detection.label,
            confidence=detection.confidence,
            video_path=video_path,
            metadata={'type': alert_type}
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
