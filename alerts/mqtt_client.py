# ============================================================
# FILE: alerts/mqtt_client.py
# ============================================================

import json
import logging
from typing import Dict, Any

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

logger = logging.getLogger(__name__)

class MQTTClient:
    def __init__(self, broker: str, port: int, topics: Dict[str, str], enabled: bool = True):
        self.broker = broker
        self.port = port
        self.topics = topics
        self.enabled = enabled and MQTT_AVAILABLE
        self.client = None

        if self.enabled:
            self._initialize()
        else:
            if not MQTT_AVAILABLE:
                logger.warning("paho-mqtt not installed, MQTT disabled")
            else:
                logger.info("MQTT disabled in configuration")

    def _initialize(self):
        try:
            self.client = mqtt.Client()
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect

            # paho's own backoff, retrying indefinitely. The previous code gave
            # up permanently after 10 attempts, so a broker that outlasted a few
            # minutes of downtime meant no MQTT until the next restart.
            self.client.reconnect_delay_set(min_delay=1, max_delay=60)

            # connect_async never blocks: at the pond the broker may be
            # unreachable at boot, and startup must not stall waiting for it.
            self.client.connect_async(self.broker, self.port, keepalive=60)
            self.client.loop_start()
            logger.info(
                f"MQTT connecting in background to {self.broker}:{self.port} "
                f"(retries indefinitely, 1-60s backoff)"
            )
        except Exception as e:
            logger.error(f"Failed to initialize MQTT: {e}")
            self.enabled = False

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected to MQTT broker")
        else:
            logger.error(f"MQTT connection refused with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        # paho's network loop reconnects on its own; nothing to do but note it.
        if rc != 0:
            logger.warning("Lost MQTT connection — paho will keep retrying")
        else:
            logger.info("Disconnected from MQTT broker")

    def publish(self, topic_key: str, payload: Dict[str, Any]):
        if not self.enabled or not self.client:
            return

        topic = self.topics.get(topic_key)
        if not topic:
            logger.error(f"Unknown MQTT topic key: {topic_key}")
            return

        try:
            message = json.dumps(payload)
            self.client.publish(topic, message)
            logger.debug(f"Published to {topic}: {message}")
        except Exception as e:
            logger.error(f"Failed to publish MQTT message: {e}")

    def disconnect(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
