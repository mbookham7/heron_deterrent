# ============================================================
# FILE: alerts/mqtt_client.py
# ============================================================

import json
import logging
from typing import Dict, Any, Optional

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
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
            logger.info(f"MQTT client initialized: {self.broker}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to initialize MQTT: {e}")
            self.enabled = False
    
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected to MQTT broker")
        else:
            logger.error(f"MQTT connection failed with code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        logger.warning("Disconnected from MQTT broker")
    
    def publish(self, topic_key: str, payload: Dict[str, Any]):
        if not self.enabled or not self.client:
            return
        
        topic = self.topics.get(topic_key)
        if not topic:
            logger.error(f"Unknown topic key: {topic_key}")
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