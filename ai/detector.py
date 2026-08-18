# ============================================================
# FILE: ai/detector.py
# YOLOv8 TFLite inference.
#
# The models in models/ are single-class (heron) and emit
# [1, 4 + num_classes, num_anchors] — e.g. [1, 5, 8400] at 640px.
# The class count is checked against ai.labels at startup so a
# config/model mismatch cannot silently mislabel detections.
#
# Frames are LETTERBOXED to the model's square input, not stretched
# to it — see ai/letterbox.py for why that is not optional. Boxes come
# out of the model in the padded square's coordinate system and are
# mapped back to the original frame before leaving infer(), so a
# Detection's bbox always describes the frame that was passed in.
# ============================================================

import logging
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .letterbox import LetterboxParams, letterbox, undo_xyxy

logger = logging.getLogger(__name__)

# Cap boxes entering NMS. Above the threshold there are rarely more than a
# handful; this only guards against a pathological frame pegging the CPU.
MAX_BOXES_FOR_NMS = 100


class ClassCountMismatch(RuntimeError):
    """ai.labels does not describe the loaded model's classes."""


class Detection:
    def __init__(self, label: str, confidence: float, bbox: Optional[Tuple] = None):
        self.label = label
        self.confidence = confidence
        # (x, y, w, h) normalized 0-1, corner origin.
        # As returned by infer(), these are relative to the ORIGINAL frame —
        # the letterbox padding has already been undone. Between parse_output()
        # and rescale_to_frame() they are relative to the padded square; that
        # window is internal to infer() and nothing outside sees it.
        self.bbox = bbox

    def __repr__(self):
        return f"Detection(label={self.label}, confidence={self.confidence:.2f})"


class ObjectDetector:
    def __init__(self, model_loader, confidence_threshold: float = 0.6,
                 iou_threshold: float = 0.45, labels: Dict[int, str] = None):
        self.interpreter = model_loader.get_interpreter()
        self.input_details = model_loader.get_input_details()
        self.output_details = model_loader.get_output_details()
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.labels = labels or {0: "heron"}

        # A TFLite Interpreter is not thread-safe, and infer() is reachable from
        # the detection thread and the dashboard's test-upload endpoint at the
        # same time. Concurrent invoke() is undefined behaviour and can hard-fault
        # the process under an Edge TPU delegate.
        self._lock = threading.Lock()

        in_detail = self.input_details[0]
        self.input_height = in_detail['shape'][1]
        self.input_width = in_detail['shape'][2]
        self.input_dtype = in_detail['dtype']
        quant = in_detail.get('quantization', (0.0, 0))
        self.input_scale, self.input_zero_point = quant[0], quant[1]

        # 4 bbox coords + one score per class
        self.num_features = 4 + len(self.labels)
        model_classes = self._model_class_count()
        if model_classes != len(self.labels):
            raise ClassCountMismatch(
                f"ai.labels declares {len(self.labels)} class(es) "
                f"{sorted(self.labels.values())} but {model_loader.model_path.name} "
                f"emits {model_classes}.\n"
                f"  The trained model is single-class (heron). Fix config.yaml -> ai.labels "
                f"so it lists exactly {model_classes} entry/entries."
            )

        logger.info(
            f"Detector ready: input {self.input_width}x{self.input_height} "
            f"{self.input_dtype.__name__} | classes={len(self.labels)} "
            f"| confidence>={confidence_threshold} | iou={iou_threshold}"
        )

    def _model_class_count(self) -> int:
        """Infer class count from the output tensor: features = 4 bbox + classes."""
        shape = [d for d in self.output_details[0]['shape'] if d != 1]
        if len(shape) != 2:
            raise ClassCountMismatch(
                f"Unexpected model output shape {list(self.output_details[0]['shape'])}; "
                f"expected [1, 4+classes, anchors]."
            )
        # The feature axis is the short one; anchors number in the thousands.
        return min(shape) - 4

    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, LetterboxParams]:
        """Letterbox the frame to the model's input and quantize as required.

        Returns the input tensor AND the letterbox parameters — the caller needs
        the latter to map detections back to the frame it passed in. Plain
        cv2.resize() here would stretch a 4:3 frame into the square input and
        show the model a distortion it never saw in training.
        """
        img, params = letterbox(frame, (self.input_height, self.input_width))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)

        if self.input_dtype == np.float32:
            return img.astype(np.float32) / 255.0, params

        # Quantized input: map normalized pixels through the tensor's own
        # scale/zero_point. Handles uint8 (Coral) and int8 alike.
        scale = self.input_scale or (1.0 / 255.0)
        q = (img.astype(np.float32) / 255.0) / scale + self.input_zero_point
        limits = np.iinfo(self.input_dtype)
        tensor = np.clip(np.round(q), limits.min, limits.max).astype(self.input_dtype)
        return tensor, params

    def dequantize(self, data: np.ndarray, output_detail: dict) -> np.ndarray:
        if data.dtype not in (np.uint8, np.int8):
            return data
        quant = output_detail.get('quantization', (0.0, 0))
        scale, zero_point = quant[0], quant[1]
        if not scale:
            return data.astype(np.float32)
        return (data.astype(np.float32) - zero_point) * scale

    def parse_output(self, output: np.ndarray) -> List[Detection]:
        """Vectorised parse of the full anchor grid.

        Scores every anchor — a heron anywhere in frame is found, not just one
        in the top-left corner.

        Boxes come back normalized against the PADDED SQUARE that was fed to the
        model. rescale_to_frame() converts them to the original frame; call it
        before handing a Detection to anything else.
        """
        arr = self.dequantize(output, self.output_details[0])
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            logger.error(f"Cannot parse model output of shape {arr.shape}")
            return []

        # Orient to [num_anchors, num_features] using the known feature count.
        if arr.shape[0] == self.num_features and arr.shape[1] != self.num_features:
            arr = arr.T
        if arr.shape[1] != self.num_features:
            logger.error(
                f"Model output {arr.shape} has no axis of length {self.num_features} "
                f"(4 bbox + {len(self.labels)} class)"
            )
            return []

        boxes = arr[:, :4]
        class_scores = arr[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

        keep = np.flatnonzero(scores >= self.confidence_threshold)
        if keep.size == 0:
            logger.debug(f"No anchors above {self.confidence_threshold} (max was {scores.max():.3f})")
            return []

        if keep.size > MAX_BOXES_FOR_NMS:
            keep = keep[np.argsort(scores[keep])[::-1][:MAX_BOXES_FOR_NMS]]

        detections = []
        for i in keep:
            x_center, y_center, w, h = boxes[i]
            detections.append(Detection(
                label=self.labels.get(int(class_ids[i]), "unknown"),
                confidence=float(scores[i]),
                bbox=(float(x_center - w / 2), float(y_center - h / 2), float(w), float(h)),
            ))

        logger.info(f"{len(detections)} anchor(s) above threshold (best {scores[keep].max():.3f})")
        return detections

    def non_max_suppression(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return []

        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = np.argsort(scores)[::-1]

        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            if len(indices) == 1:
                break
            ious = self.compute_iou(boxes[current], boxes[indices[1:]])
            indices = indices[1:][ious < self.iou_threshold]

        return [detections[i] for i in keep]

    def rescale_to_frame(self, detections: List[Detection],
                         params: LetterboxParams) -> List[Detection]:
        """Rewrite bboxes from padded-square space into original-frame space.

        The model's boxes are normalized against the padded SQUARE it was fed.
        Around a 640x480 frame that square is a third grey padding, so a box
        left in that space is wrong in the vertical axis by exactly the amount
        of that padding — and anything downstream that writes an annotation
        file from it would be wrong by the same amount.

        Runs AFTER non-max suppression, deliberately: NMS is IoU-based, and IoU
        is only meaningful within one coordinate system. Ultralytics does the
        same in the same order (NMS in letterboxed space, then scale_boxes).

        The resulting bbox is (x, y, w, h) normalized 0-1 against the ORIGINAL
        frame, corner origin — the same units and layout as before, now
        measured against the frame the caller actually handed in.

        Assumes the model's box outputs are normalized 0-1 against its input,
        which is what the CPU int8 model emits. The compiled Edge TPU model
        reportedly emits pixel units instead (see README, "Edge TPU status") —
        if that target is ever revived, check this before trusting a box from
        it. It detects nothing at all today, so the point is moot.
        """
        if not detections:
            return detections

        square = np.array([[params.new_w, params.new_h, params.new_w, params.new_h]])
        # normalized xywh on the square -> xyxy in square pixels
        xywh = np.array([d.bbox for d in detections], dtype=np.float64)
        xyxy = np.stack([xywh[:, 0], xywh[:, 1],
                         xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]], axis=1) * square

        frame_xyxy = undo_xyxy(xyxy, params)

        for det, (x1, y1, x2, y2) in zip(detections, frame_xyxy):
            # Plain floats, not numpy scalars: a bbox can end up in JSON.
            det.bbox = (float(x1 / params.orig_w), float(y1 / params.orig_h),
                        float((x2 - x1) / params.orig_w), float((y2 - y1) / params.orig_h))
        return detections

    def compute_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2])
        y2 = np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        union = box[2] * box[3] + boxes[:, 2] * boxes[:, 3] - intersection
        return intersection / (union + 1e-6)

    def infer(self, frame: np.ndarray) -> Detection:
        try:
            input_data, params = self.preprocess_frame(frame)
            with self._lock:
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                output = self.interpreter.get_tensor(self.output_details[0]['index'])

            detections = self.non_max_suppression(self.parse_output(output))
            detections = self.rescale_to_frame(detections, params)
            if detections:
                best = max(detections, key=lambda d: d.confidence)
                logger.info(f"Best detection: {best}")
                return best

            return Detection(label="unknown", confidence=0.0)

        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            return Detection(label="unknown", confidence=0.0)

    def aggregate_detections(self, detections: List[Detection],
                             mode: str = "max",
                             alert_threshold: float = 0.0,
                             vote_k: int = 1) -> Detection:
        """Combine per-frame detections into one clip-level verdict.

        mode='max' (default, deter path): return the best heron detection
        across frames. Max-pooling is the right choice when latency matters
        and a false positive is cheap.

        mode='vote' (alert path): return heron ONLY if at least `vote_k` of
        the input frames scored `alert_threshold` or higher on heron. Anything
        less is downgraded to 'unknown' so it lands in the recording bucket
        but does not fire an alert.

        Why voting for the alert path: max-pooling over five key frames gives
        a spurious spike five independent chances to clear the alert threshold
        — the opposite of temporal filtering. If a heron is really there it
        will register on several frames; a wobble on one frame will not.
        Nothing about this changes the deter path, which still fires on a
        single trigger frame outside aggregate_detections.
        """
        if mode == "vote":
            heron_hits = [d for d in detections
                          if d.label == "heron" and d.confidence >= alert_threshold]
            if len(heron_hits) >= vote_k:
                return max(heron_hits, key=lambda d: d.confidence)
            # Not enough sustained heron frames — do not upgrade to a heron
            # alert. Preserve any other known label if the detector saw one.
            known = [d for d in detections
                     if d.label not in ("heron", "unknown")]
            if known:
                return max(known, key=lambda d: d.confidence)
            return Detection(label="unknown", confidence=0.0)

        # Legacy max-pool behaviour, still used by the deter path.
        heron = [d for d in detections if d.label == "heron"]
        if heron:
            return max(heron, key=lambda d: d.confidence)

        known = [d for d in detections if d.label != "unknown"]
        if known:
            return max(known, key=lambda d: d.confidence)

        return Detection(label="unknown", confidence=0.0)
