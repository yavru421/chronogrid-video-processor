#!/usr/bin/env python3
"""
Chronogrid API Server - REST API for automation and integration

Provides endpoints for job submission, status checking, and result retrieval.
"""

from flask import Flask, request, jsonify
import threading
import uuid
from pathlib import Path
import json
from datetime import datetime
from functools import wraps
import time

app = Flask(__name__)

# In-memory job storage (for demo; use database in production)
jobs = {}

# Simple rate limiting (in production, use Redis or similar)
request_counts = {}
RATE_LIMIT = 10  # requests per minute
RATE_WINDOW = 60  # seconds

def rate_limited(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_ip = request.remote_addr
        current_time = time.time()

        if client_ip not in request_counts:
            request_counts[client_ip] = []

        # Clean old requests
        request_counts[client_ip] = [
            req_time for req_time in request_counts[client_ip]
            if current_time - req_time < RATE_WINDOW
        ]

        if len(request_counts[client_ip]) >= RATE_LIMIT:
            return jsonify({'error': 'Rate limit exceeded'}), 429

        request_counts[client_ip].append(current_time)
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/v1/jobs', methods=['POST'])
@rate_limited
def submit_job():
    """Submit a new processing job."""
    try:
        data = request.get_json()
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Invalid JSON payload'}), 400

        video_path = data.get('video_path', '').strip()
        if not video_path:
            return jsonify({'error': 'video_path required and cannot be empty'}), 400

        # Security: Validate path doesn't contain dangerous characters
        if '..' in video_path or video_path.startswith('/'):
            return jsonify({'error': 'Invalid video_path'}), 400

        # Security: Check file exists and is accessible
        if not Path(video_path).expanduser().resolve().exists():
            return jsonify({'error': 'Video file not found'}), 404

        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            'id': job_id,
            'status': 'queued',
            'video_path': video_path,
            'options': data.get('options', {}),
            'created_at': datetime.now().isoformat(),
            'result': None
        }

        # Start processing in background
        threading.Thread(target=process_job, args=(job_id,), daemon=True).start()

        return jsonify({'job_id': job_id, 'status': 'queued'}), 201

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/v1/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get job status and results."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    return jsonify(jobs[job_id])

@app.route('/api/v1/jobs', methods=['GET'])
def list_jobs():
    """List all jobs."""
    return jsonify(list(jobs.values()))

def process_job(job_id):
    """Process a job (placeholder)."""
    job = jobs[job_id]
    job['status'] = 'processing'

    try:
        # Import and run processing
        from chronogrid.core.processing import process_video
        result = process_video(Path(job['video_path']), **job['options'])
        job['status'] = 'completed'
        job['result'] = {
            'chronogrid_path': str(result.chronogrid_path),
            'analysis': result.analysis_text
        }
    except Exception as e:
        job['status'] = 'failed'
        job['error'] = str(e)

if __name__ == '__main__':
    # Security: Never run with debug=True in production
    # Security: Bind to localhost only, not 0.0.0.0
    app.run(host='127.0.0.1', port=5000, debug=False)