# ============================================================
# FILE: ui/app.py
# ============================================================

import os
import tempfile
import time
from flask import Flask, render_template, jsonify, request, send_file, Response
import cv2
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

app = Flask(__name__)
system = None

def init_app(heron_system):
    global system
    system = heron_system

@app.after_request
def add_security_headers(response):
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

@app.route('/')
def index():
    return render_template('index.html')

# Deliberately below the capture rate: the stream shares one camera (and one
# lock) with the detection pipeline, and an unthrottled loop pegged a core and
# starved detection of frames.
STREAM_FPS = 2


@app.route('/api/live-stream')
def live_stream():
    def generate():
        delay = 1.0 / STREAM_FPS
        while True:
            if not (system and system.camera):
                break
            frame = system.camera.read()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            # Always sleep, including when the read failed, so a disconnected
            # camera cannot turn this into a busy loop.
            time.sleep(delay)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/alerts')
def get_alerts():
    if not system:
        return jsonify([])

    limit = min(int(request.args.get('limit', 50)), 500)
    alerts = system.database.get_recent_alerts(limit=limit)
    return jsonify(alerts)

@app.route('/api/detections')
def get_detections():
    if not system:
        return jsonify([])

    label = request.args.get('label')
    limit = min(int(request.args.get('limit', 100)), 500)

    detections = system.database.get_detections(label=label, limit=limit)
    return jsonify(detections)

@app.route('/api/unclassified')
def get_unclassified():
    if not system:
        return jsonify([])

    unclassified = system.database.get_unclassified_detections()
    return jsonify(unclassified)

# Labels that require a bounding box (only class the model is trained on).
# Anything else is written as a background image (empty annotation), which is
# YOLO's convention for a negative example — see save_labeled_image.
LABELS_REQUIRING_BBOX = {'heron'}
ALLOWED_LABELS = {'heron', 'cat', 'dog', 'bird', 'person', 'unknown'}


def _detection_by_id(detection_id: int):
    """The Detection row for `detection_id`, or None. Fetches enough rows to
    cover the recent history the UI would be labelling; a lookup by id would be
    cleaner but that would need a new database method."""
    for d in system.database.get_detections(limit=500):
        if d['id'] == detection_id:
            return d
    return None


def _open_clip(detection_id: int):
    """Return (cv2.VideoCapture, native_w, native_h, frame_count) or None."""
    detection = _detection_by_id(detection_id)
    if not detection or not detection.get('video_path'):
        return None
    path = Path(detection['video_path'])
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, w, h, count


@app.route('/api/detection/<int:detection_id>/frame-info')
def frame_info(detection_id):
    """Native dimensions and frame count for the clip. The client needs the
    native dimensions to scale mouse coordinates into pixel-space bboxes; a
    guess like "640x480 always" is exactly the bug this endpoint prevents."""
    if not system:
        return jsonify({'error': 'System not initialized'}), 500
    opened = _open_clip(detection_id)
    if not opened:
        return jsonify({'error': 'clip not found'}), 404
    cap, w, h, count = opened
    cap.release()
    return jsonify({'width': w, 'height': h, 'frame_count': max(count, 1)})


@app.route('/api/detection/<int:detection_id>/frame/<int:index>')
def frame_jpeg(detection_id, index):
    """Serve one frame of the clip as a JPEG, at native resolution.

    Native, not display size: the client owns display scaling and must know
    the native dimensions independently. Any resize here would be a second
    source of scale, which is exactly how coordinate transforms drift.
    """
    if not system:
        return "System not initialized", 500
    opened = _open_clip(detection_id)
    if not opened:
        return "clip not found", 404
    cap, _w, _h, count = opened
    try:
        idx = max(0, min(index, count - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            return "cannot decode frame", 500
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ret:
            return "encode failed", 500
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    finally:
        cap.release()


@app.route('/api/label', methods=['POST'])
def submit_label():
    """Persist a label for a detection and write the corresponding training file.

    bboxes are (x, y, w, h) in NATIVE pixel coords — the client scales from its
    display size to the frame's true dimensions before sending. Two vocabularies
    to keep straight: Detection.bbox elsewhere is normalised 0-1, but
    save_labeled_image expects pixels because it converts to YOLO's normalised
    form itself. Don't cross them.

    Empty bboxes with label='heron' is refused: an empty YOLO .txt is a
    background/negative example, so "labelled heron with no box" would poison
    the training set. Non-heron labels correctly write an empty file — that IS
    the intended background example.
    """
    if not system:
        return jsonify({'success': False, 'error': 'System not initialized'}), 500

    try:
        data = request.json or {}
        detection_id = data.get('detection_id')
        label = data.get('label')
        bboxes = data.get('bboxes', []) or []
        frame_index = int(data.get('frame_index', 0))

        if label not in ALLOWED_LABELS:
            return jsonify({'success': False, 'error': f'Invalid label: {label}'}), 400

        if label in LABELS_REQUIRING_BBOX and not bboxes:
            # Fail visibly. Silent empty annotations are the whole reason this
            # endpoint was rewritten.
            return jsonify({
                'success': False,
                'error': f"'{label}' requires at least one bounding box — "
                         f"an empty label file would train the model that "
                         f"this frame contains no heron."
            }), 400

        system.database.update_manual_label(detection_id, label)

        opened = _open_clip(detection_id)
        if opened:
            cap, native_w, native_h, count = opened
            try:
                idx = max(0, min(frame_index, count - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    return jsonify({'success': False,
                                    'error': f'cannot decode frame {idx}'}), 500

                # For labels that do not require a bbox (cat/dog/bird/person/
                # unknown), any bboxes the client sent are dropped and an empty
                # .txt is written. That IS a valid YOLO background example.
                write_boxes = bboxes if label in LABELS_REQUIRING_BBOX else []
                _validate_pixel_boxes(write_boxes, native_w, native_h)
                system.media_store.save_labeled_image(
                    frame, label, write_boxes, detection_id
                )
            finally:
                cap.release()

        logger.info(f"Label submitted: detection {detection_id} -> {label} "
                    f"(frame {frame_index}, {len(bboxes)} box(es))")
        return jsonify({'success': True})

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error submitting label: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


def _validate_pixel_boxes(bboxes, native_w, native_h):
    """Reject boxes that would produce a garbage YOLO line.

    Refuses NaN/negative/zero-area/off-frame boxes. Cheap upfront check is
    much better than a corrupted annotation shipped into training silently.
    """
    for i, b in enumerate(bboxes):
        if len(b) != 4:
            raise ValueError(f"bbox {i} must be [x, y, w, h]")
        x, y, w, h = (float(v) for v in b)
        if w <= 0 or h <= 0:
            raise ValueError(f"bbox {i} has non-positive size {w}x{h}")
        if x < 0 or y < 0 or x + w > native_w + 0.5 or y + h > native_h + 0.5:
            raise ValueError(
                f"bbox {i} ({x:.1f},{y:.1f},{w:.1f},{h:.1f}) lies outside the "
                f"{native_w}x{native_h} frame — client coordinate scaling may "
                f"be wrong"
            )

@app.route('/api/trigger-deterrent', methods=['POST'])
def trigger_deterrent():
    if not system:
        return jsonify({'success': False, 'error': 'System not initialized'})

    try:
        system.trigger_deterrent()
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error triggering deterrent: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/upload-test-video', methods=['POST'])
def upload_test_video():
    if not system:
        return jsonify({'success': False, 'error': 'System not initialized'})

    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'})

        video_file = request.files['video']

        filename = video_file.filename or 'upload.mp4'
        file_ext = Path(filename).suffix.lower() or '.mp4'
        if file_ext not in {'.mp4', '.mov', '.avi', '.mkv'}:
            return jsonify({'success': False, 'error': 'Unsupported file type'})

        # Write to a proper temp file; always clean up regardless of errors
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=file_ext)
        try:
            os.close(tmp_fd)
            video_file.save(tmp_path)

            cap = cv2.VideoCapture(tmp_path)
            frames = []
            frame_count = 0
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                    frame_count += 1
            finally:
                cap.release()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if frames:
            system.process_detection_event(frames)
            detections = system.database.get_detections(limit=10)
            heron_detections = [d for d in detections if d.get('label') == 'heron']

            return jsonify({
                'success': True,
                'frames_processed': frame_count,
                'herons_detected': len(heron_detections),
                'detections': heron_detections
            })
        else:
            return jsonify({'success': False, 'error': 'No frames extracted'})

    except Exception as e:
        logger.error(f"Error processing uploaded video: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/heron-detections')
def get_heron_detections():
    if not system:
        return jsonify([])

    try:
        limit = min(int(request.args.get('limit', 50)), 500)
        detections = system.database.get_detections(label='heron', limit=limit)
        return jsonify(detections)
    except Exception as e:
        logger.error(f"Error getting heron detections: {e}")
        return jsonify([])

@app.route('/api/system-status')
def system_status():
    if not system:
        return jsonify({'status': 'offline'})

    try:
        status = {
            'status': 'online' if system.running else 'stopped',
            'camera_connected': system.camera.is_connected(),
            'active_hours': system.is_within_active_hours(),
            'cooldown_elapsed': system.cooldown_elapsed(),
            'model_loaded': system.detector is not None
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/api/config', methods=['GET', 'POST'])
def config_endpoint():
    if not system:
        return jsonify({'success': False, 'error': 'System not initialized'})

    if request.method == 'GET':
        # Strip sensitive keys before returning
        safe_config = {k: v for k, v in system.config.config.items()
                       if k != 'alert'}
        return jsonify(safe_config)

    try:
        new_config = request.json
        import yaml
        with open(system.config.config_path, 'w') as f:
            yaml.dump(new_config, f)
        system.config.reload()
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/media/<path:filename>')
def serve_media(filename):
    if not system:
        return "System not initialized", 404

    filepath = system.media_store.get_file_path(filename)
    if not filepath or not filepath.exists():
        return "File not found", 404

    # Prevent path traversal: resolved path must be inside the media directory
    base = system.media_store.base_path.resolve()
    try:
        filepath.resolve().relative_to(base)
    except ValueError:
        logger.warning(f"Path traversal attempt blocked: {filename}")
        return "Forbidden", 403

    return send_file(filepath)

def start_web_ui(heron_system, host='0.0.0.0', port=5000):
    init_app(heron_system)
    logger.info(f"Starting web UI on {host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)
