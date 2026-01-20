# ============================================================
# FILE: ui/app.py
# ============================================================

from flask import Flask, render_template, jsonify, request, send_file, Response
import cv2
import json
import logging
from pathlib import Path
import numpy as np
from io import BytesIO

logger = logging.getLogger(__name__)

app = Flask(__name__)
system = None  # Will be set when starting the UI

def init_app(heron_system):
    global system
    system = heron_system

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
                    # Encode frame as JPEG
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
    
    alerts = system.database.get_recent_alerts(limit=50)
    return jsonify(alerts)

@app.route('/api/detections')
def get_detections():
    if not system:
        return jsonify([])
    
    label = request.args.get('label')
    limit = int(request.args.get('limit', 100))
    
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
        
        # Update database
        system.database.update_manual_label(detection_id, label)
        
        # Get the detection
        detections = system.database.get_detections()
        detection = next((d for d in detections if d['id'] == detection_id), None)
        
        if detection and detection['video_path']:
            # Extract first frame from video
            video_path = Path(detection['video_path'])
            if video_path.exists():
                cap = cv2.VideoCapture(str(video_path))
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    # Save labeled data in YOLO format
                    system.media_store.save_labeled_image(
                        frame, label, bboxes, detection_id
                    )
        
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
        
        # Save temporarily
        temp_path = Path('./temp_upload.mp4')
        video_file.save(temp_path)
        
        # Process video
        cap = cv2.VideoCapture(str(temp_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        temp_path.unlink()  # Delete temp file
        
        if frames:
            # Process as detection event
            system.process_detection_event(frames)
            return jsonify({'success': True, 'frames_processed': len(frames)})
        else:
            return jsonify({'success': False, 'error': 'No frames extracted'})
        
    except Exception as e:
        logger.error(f"Error processing uploaded video: {e}")
        return jsonify({'success': False, 'error': str(e)})

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
        return jsonify(system.config.config)
    
    elif request.method == 'POST':
        try:
            # Update config (simplified - would need validation)
            new_config = request.json
            
            # Save to file
            import yaml
            with open(system.config.config_path, 'w') as f:
                yaml.dump(new_config, f)
            
            # Reload
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
    if filepath and filepath.exists():
        return send_file(filepath)
    else:
        return "File not found", 404

def start_web_ui(heron_system, host='0.0.0.0', port=5000):
    init_app(heron_system)
    logger.info(f"Starting web UI on {host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)