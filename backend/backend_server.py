import os
import uuid
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import ai_pipeline

app = Flask(__name__)
CORS(app)

JOBS = {}

UPLOAD_DIR = "static/uploads"
RESULT_DIR = "static/results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

@app.route("/api/jobs", methods=["POST"])
def submit_job():
    """
    Endpoint to receive an image and motion_id, and start the background pipeline.
    """
    print("Received a new job request...")
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']
    motion_id = request.form['motion_id']
    
    job_id = str(uuid.uuid4())
    image_filename = f"{job_id}_{image_file.filename}"
    image_path = os.path.join(UPLOAD_DIR, image_filename)
    image_file.save(image_path)
    
    print(f"New job created with ID: {job_id}. Image saved to {image_path}")

    JOBS[job_id] = {'status': 'processing', 'step': 'Waiting in queue...'}
    
    thread = threading.Thread(
        target=ai_pipeline.run_full_pipeline, 
        args=(job_id, image_path, motion_id, JOBS)
    )
    thread.start()
    
    return jsonify({"job_id": job_id})


@app.route("/api/jobs/<job_id>/status", methods=["GET"])
def get_job_status(job_id):
    """
    Endpoint for the frontend to poll for job progress updates.
    """
    print(f"Checking status for job {job_id}...")
    job = JOBS.get(job_id)
    
    if not job:
        return jsonify({"error": "Job not found"}), 404
        
    return jsonify(job)


@app.route('/api/results/<filename>')
def get_result_file(filename):
    """
    Endpoint to serve the final generated video file.
    """
    return send_from_directory(RESULT_DIR, filename)

if __name__ == "__main__":
    app.run(port=8000, debug=True)