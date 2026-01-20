# ============================================================
# FILE: ai/model_loader.py
# ============================================================

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_path: str, use_edge_tpu: bool = False):
        self.model_path = Path(model_path)
        self.use_edge_tpu = use_edge_tpu
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._load_model()
    
    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            if self.use_edge_tpu:
                # Try to load with Edge TPU
                try:
                    from tflite_runtime.interpreter import Interpreter
                    from tflite_runtime.interpreter import load_delegate
                    
                    self.interpreter = Interpreter(
                        model_path=str(self.model_path),
                        experimental_delegates=[load_delegate('libedgetpu.so.1')]
                    )
                    logger.info("Model loaded with Edge TPU acceleration")
                except Exception as e:
                    logger.warning(f"Edge TPU not available, falling back to CPU: {e}")
                    self.use_edge_tpu = False
            
            if not self.use_edge_tpu:
                # Fall back to CPU
                try:
                    from tflite_runtime.interpreter import Interpreter
                except ImportError:
                    import tensorflow as tf
                    Interpreter = tf.lite.Interpreter
                
                self.interpreter = Interpreter(model_path=str(self.model_path))
                logger.info("Model loaded on CPU")
            
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            logger.info(f"Model input shape: {self.input_details[0]['shape']}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def get_interpreter(self):
        return self.interpreter
    
    def get_input_details(self):
        return self.input_details
    
    def get_output_details(self):
        return self.output_details