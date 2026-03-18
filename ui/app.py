# ============================================================
# FILE: ui/app.py
# ============================================================

import os
import tempfile
from flask import Flask, render_template, jsonify, request, send_file, Response
import cv2
import json
import logging
from pathlib import Path
import numpy as np
from io import BytesIO

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

@app.route('/api/live-stream')
def live_stream():
    def generate():
        while True:
            if system and system.camera:
                frame = system.camera.read()
                if frame is not None:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                break

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

@app.route('/api/label', methods=['POST'])
def submit_label():
    if not system:
        return jsonify({'success': False, 'error': 'System not initialized'})

    try:
        data = request.json
        detection_id = data.get('detection_id')
        label = data.get('label')
        bboxes = data.get('bboxes', [])

        allowed_labels = {'heron', 'cat', 'dog', 'bird', 'person', 'unknown'}
        if label not in allowed_labels:
            return jsonify({'success': False, 'error': f'Invalid label: {label}'})

        system.database.update_manual_label(detection_id, label)

        detections = system.database.get_detections()
        detection = next((d for d in detections if d['id'] == detection_id), None)

        if detection and detection['video_path']:
            video_path = Path(detection['video_path'])
            if video_path.exists():
                cap = cv2.VideoCapture(str(video_path))
                try:
                    ret, frame = cap.read()
                    if ret:
                        system.media_store.save_labeled_image(frame, label, bboxes, detection_id)
                finally:
                    cap.release()

        logger.info(f"Label submitted: detection {detection_id} -> {label}")
        return jsonify({'success': True})

    except Exception as e:
        logger.error(f"Error submitting label: {e}")
        return jsonify({'success': False, 'error': str(e)})

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
