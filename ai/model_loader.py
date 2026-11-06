# ============================================================
# FILE: ai/model_loader.py
#
# Hardware target selection. `ai.target` in config.yaml is the
# only switch: it determines both the model filename and the
# TFLite backend. There is no model_path key and no silent
# fallback — a misconfigured target fails at startup with the
# command needed to fix it.
# ============================================================

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ai.target names the INFERENCE BACKEND, not the board. A Coral Dev Board runs
# 'cpu' perfectly well, and currently has to: the compiled Edge TPU model's
# classification head was destroyed by full-integer quantisation, so it scores
# ~1e-5 on every anchor and detects nothing. Until a working compiled model
# exists, 'cpu' is the only usable target on either board.
#
# target -> (model filename in models/, needs Edge TPU delegate, source artefact)
TARGETS = {
    'cpu':     ('heron_cpu_int8.tflite', False, 'best_int8_edgetpu.tflite'),
    'edgetpu': ('heron_edgetpu.tflite',  True,  'best_full_int8_edgetpu.tflite'),
}

# Registered models, by the name modes.<mode>.model uses. Each maps to its own
# target table. 'heron' is the daylight model and its table is TARGETS above,
# unchanged. A future IR model is one new entry here — ai.target still picks
# the backend for every model, so it stays the only hardware switch.
MODELS = {
    'heron': TARGETS,
}

EDGETPU_LIB = 'libedgetpu.so.1'


def model_path(model: str, target: str, model_dir: str = './models') -> Path:
    return Path(model_dir) / MODELS[model][target][0]


class ModelLoadError(RuntimeError):
    """The configured target cannot be loaded. The message is shown to the operator."""


class ModelLoader:
    def __init__(self, target: str, num_threads: int = 4, model_dir: str = './models',
                 model: str = 'heron'):
        if model not in MODELS:
            raise ModelLoadError(
                f"model '{model}' is not registered. Registered: {', '.join(sorted(MODELS))}.\n"
                f"  Fix: edit config.yaml -> modes.<mode>.model"
            )
        targets = MODELS[model]
        if target not in targets:
            raise ModelLoadError(
                f"ai.target is '{target}' but must be one of: {', '.join(sorted(targets))}.\n"
                f"  Fix: edit config.yaml -> ai.target"
            )

        filename, self.use_edge_tpu, source_name = targets[target]
        self.model = model
        self.target = target
        self.model_path = Path(model_dir) / filename
        self.num_threads = num_threads

        if not self.model_path.exists():
            raise ModelLoadError(
                f"ai.target is '{target}', so the model must be at: {self.model_path}\n"
                f"  That file does not exist. Copy it from the training repo:\n"
                f"    scp heron-detector/models/{source_name} \\\n"
                f"        <board>:~/heron_deterrent/{self.model_path}\n"
                f"  Models are gitignored — cloning the repo does not bring them."
            )

        self.interpreter = self._build_interpreter()
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        backend = 'Edge TPU' if self.use_edge_tpu else f'CPU ({self.num_threads} threads)'
        shape = lambda d: 'x'.join(str(int(v)) for v in d['shape'])
        logger.info(
            f"Model loaded: {self.model_path.name} | target={target} | backend={backend} | "
            f"input={shape(self.input_details[0])} {self.input_details[0]['dtype'].__name__} | "
            f"output={shape(self.output_details[0])}"
        )

    def _build_interpreter(self):
        return self._build_edge_tpu() if self.use_edge_tpu else self._build_cpu()

    def _build_edge_tpu(self):
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate
        except ImportError as e:
            raise ModelLoadError(
                "ai.target is 'edgetpu' but tflite_runtime is not installed.\n"
                "  On Mendel Linux:  sudo apt-get install -y python3-pycoral libedgetpu1-std\n"
                "  No Edge TPU?      set ai.target: cpu in config.yaml"
            ) from e

        try:
            delegate = load_delegate(EDGETPU_LIB)
        except Exception as e:
            raise ModelLoadError(
                f"ai.target is 'edgetpu' but the delegate ({EDGETPU_LIB}) would not load.\n"
                f"  Is the TPU visible?   lspci -nn | grep 089a\n"
                f"  Is the driver up?     dmesg | grep apex\n"
                f"  Reinstall runtime:    sudo apt-get install --reinstall libedgetpu1-std\n"
                f"  No Edge TPU?          set ai.target: cpu in config.yaml\n"
                f"  Underlying error: {e}"
            ) from e

        # Deliberately no CPU fallback: the Coral model has uint8 I/O compiled for
        # the TPU. Silently running it elsewhere hides a hardware fault in the field.
        return Interpreter(
            model_path=str(self.model_path),
            experimental_delegates=[delegate],
        )

    def _build_cpu(self):
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                import tensorflow as tf  # dev machines only; too heavy for a board
                Interpreter = tf.lite.Interpreter
            except ImportError as e:
                raise ModelLoadError(
                    "ai.target is 'cpu' but no TFLite runtime is installed.\n"
                    "  On a Raspberry Pi:  pip install tflite-runtime\n"
                    "  On Mendel/Coral:    sudo apt-get install -y python3-tflite-runtime\n"
                    "                      (and create the venv with --system-site-packages)\n"
                    "  On a dev machine:   pip install tensorflow"
                ) from e

        return Interpreter(model_path=str(self.model_path), num_threads=self.num_threads)

    def get_interpreter(self):
        return self.interpreter

    def get_input_details(self):
        return self.input_details

    def get_output_details(self):
        return self.output_details
