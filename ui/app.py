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

from capture.frame_classifier import FRAME_TYPES, classify_frame, measure, resolve_label_domain

logger = logging.getLogger(__name__)

# Values the dashboard may filter on. 'unknown' = no settled domain: legacy
# rows, and clips recorded while establishing or switching.
FRAME_TYPE_FILTERS = set(FRAME_TYPES) | {'unknown'}


def _frame_type_arg():
    ft = request.args.get('frame_type') or None
    if ft is not None and ft not in FRAME_TYPE_FILTERS:
        raise ValueError(f"frame_type must be one of {sorted(FRAME_TYPE_FILTERS)}")
    return ft

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
    try:
        frame_type = _frame_type_arg()
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    detections = system.database.get_detections(label=label, limit=limit,
                                                frame_type=frame_type)
    return jsonify(detections)


@app.route('/api/stats')
def get_stats():
    """Counts for the stats strip. Motion-only rows are reported separately
    and never as detections — no model looked at them."""
    if not system:
        return jsonify({})
    return jsonify(system.database.get_stats())

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
    """The Detection row for `detection_id`, or None. A direct lookup: round
    the clock, night motion rows would push older clips out of any
    "recent 500" window long before they were labelled."""
    return system.database.get_detection(detection_id)


def _recorded_frame_type(detection, index: int, frame_count: int):
    """This frame's type as classified live at capture, or None if the clip
    has no record (legacy rows, uploads). Raises ValueError if the record and
    the decoded clip disagree on length — then the index cannot be trusted to
    point at the same frame, and nothing should be filed."""
    try:
        types = json.loads(detection.get('metadata') or '{}').get('clip_frame_types')
    except (TypeError, ValueError):
        types = None
    if not types:
        return None
    if len(types) != frame_count:
        raise ValueError(f"clip has {frame_count} frames but {len(types)} recorded "
                         f"frame types — cannot tell which frame is which")
    return types[index]


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


@app.route('/api/detection/<int:detection_id>/frame/<int:index>/domain')
def frame_domain(detection_id, index):
    """Which training set this frame would be filed under, BEFORE saving.

    The labelling modal shows this so an IR frame cannot be dropped into the
    daylight set by mistake. `domain` is null when the frame cannot be filed
    as things stand; `needs_confirmation` is true when the clip has no settled
    frame type and the operator must pick one (?requested=day|ir|dark).
    """
    if not system:
        return jsonify({'error': 'System not initialized'}), 500
    detection = _detection_by_id(detection_id)
    opened = _open_clip(detection_id)
    if not detection or not opened:
        return jsonify({'error': 'clip not found'}), 404
    cap, _w, _h, count = opened
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(index, count - 1)))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        return jsonify({'error': 'cannot decode frame'}), 500

    idx = max(0, min(index, count - 1))
    requested = request.args.get('requested') or None
    try:
        recorded = _recorded_frame_type(detection, idx, count)
    except ValueError as e:
        return jsonify({'domain': None, 'reason': str(e), 'needs_confirmation': False})
    domain, why = resolve_label_domain(detection.get('frame_type'), recorded, requested)

    # Advisory only: the decoded frame carries the encoder's colour cast, so
    # this can read an IR frame as day. Shown to help a human confirm by eye
    # when there is no capture-time record; never used to route anything.
    cfg = system.frame_tracker.cfg
    chroma, luma = measure(frame, cfg.sample_step)
    return jsonify({
        'domain': domain,
        'reason': why,
        'row_frame_type': detection.get('frame_type'),
        'recorded_frame_type': recorded,
        'needs_confirmation': domain is None and (
            recorded is None or detection.get('frame_type') not in FRAME_TYPES),
        'advisory': {'decoded_reads_as': classify_frame(frame, cfg) or 'ambiguous',
                     'chroma': round(chroma, 2), 'luma': round(luma, 1)},
    })


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
        # Only needed (and only honoured) for a clip with no settled frame
        # type; see resolve_label_domain.
        requested_domain = data.get('domain') or None

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

        detection = _detection_by_id(detection_id)
        if not detection:
            return jsonify({'success': False, 'error': 'detection not found'}), 404

        # The row is marked labelled only AFTER the training file is written,
        # so a refused domain leaves the clip in the triage queue.
        domain = None
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

                # Day and night training data must never mix. The domain comes
                # from the clip's frame type, cross-checked against THIS frame's
                # type as recorded live at capture; a clip with no settled type
                # or no record needs the operator to name one.
                recorded = _recorded_frame_type(detection, idx, count)
                domain, why = resolve_label_domain(
                    detection.get('frame_type'), recorded, requested_domain)
                if domain is None:
                    return jsonify({'success': False, 'error': why}), 409

                # For labels that do not require a bbox (cat/dog/bird/person/
                # unknown), any bboxes the client sent are dropped and an empty
                # .txt is written. That IS a valid YOLO background example.
                write_boxes = bboxes if label in LABELS_REQUIRING_BBOX else []
                _validate_pixel_boxes(write_boxes, native_w, native_h)
                system.media_store.save_labeled_image(
                    frame, label, write_boxes, detection_id, frame_type=domain
                )
            finally:
                cap.release()

        system.database.update_manual_label(detection_id, label)

        logger.info(f"Label submitted: detection {detection_id} -> {label} "
                    f"(frame {frame_index}, {len(bboxes)} box(es), "
                    f"domain {domain or 'none — no clip'})")
        return jsonify({'success': True, 'domain': domain})

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
        # Manual speaker test. Plays whatever the mode says — modes gate only
        # the AUTOMATIC deterrent.
        system.play_deterrent_manually()
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
            # Same rule as the live loop: a model runs only on the domain it
            # was trained for. An IR or dark upload would get a confident,
            # meaningless daylight score — refuse it instead.
            frame_type = system.classify_clip(frames)
            mode = system.modes.get(frame_type)
            if mode is None or mode.motion_only:
                return jsonify({
                    'success': False,
                    'error': f"clip classifies as {frame_type or 'mixed/ambiguous'}; "
                             f"no model is configured for that frame type",
                })
            system.process_detection_event(frames, mode, {'source': 'upload'})
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
        detections = system.database.get_detections(label='heron', limit=limit,
                                                    frame_type=_frame_type_arg())
        return jsonify(detections)
    except Exception as e:
        logger.error(f"Error getting heron detections: {e}")
        return jsonify([])

@app.route('/api/system-status')
def system_status():
    if not system:
        return jsonify({'status': 'offline'})

    try:
        state = system.frame_state()
        mode = system.modes.get(state)
        last = system.frame_tracker.last
        status = {
            'status': 'online' if system.running else 'stopped',
            'camera_connected': system.camera.is_connected(),
            'frame_state': state,
            'mode': mode.describe() if mode else None,
            'motion_only': mode is None or mode.motion_only,
            'deterrent_armed': bool(mode and mode.deterrent),
            'frame_metrics': last.as_metadata() if last else None,
            'cooldown_elapsed': system.cooldown_elapsed(),
            'models_loaded': sorted(system.detectors),
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
