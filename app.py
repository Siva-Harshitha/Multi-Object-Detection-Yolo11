from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm', 'flv', 'wmv', 'mxf'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max

# Load your local YOLO11 model
model = YOLO("yolo11n.pt")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return send_from_directory("templates", "app.html")

@app.route("/api/process", methods=["POST"])
def process_video():
    """Process uploaded video and return detections per frame"""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file format"}), 400
        
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)
        
        # Extract key frames from video
        frames = []
        try:
            cap = cv2.VideoCapture(input_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Sample 6 evenly-spaced frames
            sample_count = min(6, max(1, total_frames // 5))
            frame_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    time_sec = idx / fps if fps > 0 else 0
                    # Run YOLO11 tracking on this frame
                    results = model.track(frame, persist=True, conf=0.5, iou=0.5)
                    
                    detections = []
                    if results and len(results) > 0:
                        r = results[0]
                        if r.boxes is not None:
                            for box in r.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = float(box.conf[0])
                                cls = int(box.cls[0])
                                track_id = int(box.id[0]) if box.id is not None else 0
                                label = model.names[cls]
                                
                                # Normalize to 0-1
                                h, w = frame.shape[:2]
                                x = float(x1 / w)
                                y = float(y1 / h)
                                box_w = float((x2 - x1) / w)
                                box_h = float((y2 - y1) / h)
                                
                                detections.append({
                                    "id": track_id,
                                    "label": str(label),
                                    "conf": float(conf),
                                    "x": x,
                                    "y": y,
                                    "w": box_w,
                                    "h": box_h,
                                    "color": ["#00FFB2","#FF6B35","#FFD600","#00B4FF","#FF3CAC","#7CFF00","#FF0055","#A855F7","#FF9F0A","#34D399"][track_id % 10]
                                })
                    
                    frames.append({
                        "time": float(time_sec),
                        "detections": detections
                    })
            
            cap.release()
        except Exception as e:
            return jsonify({"error": f"Video processing failed: {str(e)}"}), 500
        
        return jsonify({"frames": frames}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)