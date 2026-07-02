# app.py

import os
import sys
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mimetypes
from pathlib import Path
from threading import Thread

from flask import Flask, jsonify, send_file, abort, make_response, request
from flask_session import Session
from werkzeug.exceptions import NotFound

from routes.auth_routes import auth_bp
from routes.system_routes import system_bp
from routes.document_routes import document_bp
from routes.chat_routes import chat_bp
from config import config as app_config

# -------------------------------------------------------
# Flask App
# -------------------------------------------------------
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Sessions
_flask_secret = os.environ.get('FLASK_SECRET_KEY')
if not _flask_secret:
    raise RuntimeError(
        "FLASK_SECRET_KEY environment variable must be set before starting the server. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )
app.secret_key = _flask_secret
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
app.config['SESSION_FILE_DIR'] = os.path.join(os.path.dirname(__file__), 'sessions')
app.config['SESSION_FILE_THRESHOLD'] = 100
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
# Secure cookie flags
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
# Set SECURE only in production (HTTPS=true env var); default off for local dev
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('HTTPS', 'false').lower() == 'true'

# -------------------------------------------------------
# Public Embedding Asset Route (no auth)
# Example: /embedding/<doc_name>/images/<file.png>
# Serves from: {PROJECT_ROOT}/embedding/...
# -------------------------------------------------------
@app.route('/embedding/<path:filename>')
def serve_embedding_asset(filename: str):
    """
    Publicly serves files from EMBEDDINGS_DIR, e.g.
      /embedding/<doc_name>/images/<file.png>

    - Prevents path traversal
    - Guesses content-type
    - Adds public cache headers
    - Prints resolved absolute path on 404 for easy debugging
    """
    base_dir = Path(app_config.EMBEDDINGS_DIR).resolve()
    # normalize input and prevent traversal
    safe_rel = Path(filename.lstrip("/\\"))
    abs_path = (base_dir / safe_rel).resolve()

    # stay inside base_dir
    try:
        abs_path.relative_to(base_dir)
    except ValueError:
        abort(403)

    if not abs_path.is_file():
        print(f"[embedding 404] Tried: {abs_path}")  # helpful in console
        raise NotFound()

    ctype, _ = mimetypes.guess_type(str(abs_path))
    resp = make_response(send_file(str(abs_path), mimetype=ctype or 'application/octet-stream'))
    # Cache (tune if needed)
    resp.headers['Cache-Control'] = 'public, max-age=86400, immutable'
    return resp

# -------------------------------------------------------
# Initialize session
# -------------------------------------------------------
Session(app)

# -------------------------------------------------------
# Health Check (no auth required)
# -------------------------------------------------------
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint that shows system status without requiring login"""
    import app_state
    from app_state import get_active_llm
    version = os.environ.get('RENDER_GIT_COMMIT', 'local')[:9]
    if app_state.system_initialized:
        try:
            _, backend = get_active_llm()
        except Exception:
            backend = "none"
        embed_model = getattr(app_state.embedding_model, '_model', None) or 'local'
        return jsonify({
            'status': 'healthy',
            'initialized': True,
            'llm_backend': backend,
            'embed_model': embed_model,
            'version': version,
        })
    else:
        err = app_state.initialization_error or "System is still starting up, please wait…"
        return jsonify({
            'status': 'initializing',
            'initialized': False,
            'error': err,
            'version': version,
        }), 503

# -------------------------------------------------------
# Temporary diagnostic: probe which HF embedding routes work
# with the server's API key. Read-only; never reveals the key.
# -------------------------------------------------------
@app.route('/api/embed-diag', methods=['GET'])
def embed_diag():
    import requests as _rq
    key = os.environ.get('HF_API_KEY', '')
    if not key:
        return jsonify({'error': 'HF_API_KEY not set'}), 400
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    candidates = [
        ('router_v1_bge_small', 'https://router.huggingface.co/v1/embeddings',
         {'model': 'BAAI/bge-small-en-v1.5', 'input': 'hello world'}),
        ('router_v1_qwen3_06b', 'https://router.huggingface.co/v1/embeddings',
         {'model': 'Qwen/Qwen3-Embedding-0.6B', 'input': 'hello world'}),
        ('router_v1_e5_large', 'https://router.huggingface.co/v1/embeddings',
         {'model': 'intfloat/multilingual-e5-large', 'input': 'hello world'}),
        ('router_v1_minilm', 'https://router.huggingface.co/v1/embeddings',
         {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'input': 'hello world'}),
        ('pipeline_bge_small',
         'https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction',
         {'inputs': 'hello world'}),
    ]
    results = {}
    for name, url, payload in candidates:
        try:
            r = _rq.post(url, headers=headers, json=payload, timeout=20)
            detail = r.text[:200]
            if r.status_code == 200:
                try:
                    j = r.json()
                    if isinstance(j, dict) and 'data' in j:
                        detail = f"dim={len(j['data'][0]['embedding'])}"
                    elif isinstance(j, list):
                        inner = j[0] if j and isinstance(j[0], list) else j
                        detail = f"list len={len(j)} inner={len(inner) if hasattr(inner, '__len__') else '?'}"
                except Exception:
                    pass
            results[name] = {'status': r.status_code, 'ok': r.status_code == 200, 'detail': detail}
        except Exception as e:
            results[name] = {'status': None, 'ok': False, 'detail': str(e)[:200]}
    return jsonify(results)

# -------------------------------------------------------
# Blueprints
# -------------------------------------------------------
app.register_blueprint(auth_bp)
app.register_blueprint(system_bp)
app.register_blueprint(document_bp, url_prefix='/api')
app.register_blueprint(chat_bp)

# Register Embedding Routes
from routes.embedding_routes import embedding_bp
app.register_blueprint(embedding_bp)

# Stage 2: Agent management routes
from routes.agent_routes import agent_bp
app.register_blueprint(agent_bp)

# Phase 2: File viewing and editing routes
from routes.file_routes import file_bp
app.register_blueprint(file_bp)

# -------------------------------------------------------
# CORS / CSRF — allowed origins (must be defined before hooks)
# -------------------------------------------------------
_ALLOWED_ORIGINS = {
    o.strip()
    for o in os.environ.get('ALLOWED_ORIGINS', 'http://localhost:9072').split(',')
    if o.strip()
}

# -------------------------------------------------------
# CSRF protection — reject cross-origin mutating requests
# -------------------------------------------------------
@app.before_request
def csrf_origin_check():
    if request.method in ('GET', 'HEAD', 'OPTIONS'):
        return
    origin = request.headers.get('Origin', '')
    
    # Allow same-origin requests automatically
    if origin:
        from urllib.parse import urlparse
        parsed_origin = urlparse(origin)
        if parsed_origin.netloc == request.host:
            return
            
    if origin and origin not in _ALLOWED_ORIGINS:
        return jsonify({'error': 'Forbidden'}), 403

# -------------------------------------------------------
# CORS — restrict to known origins only
# -------------------------------------------------------

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get('Origin', '')
    
    is_same_origin = False
    if origin:
        from urllib.parse import urlparse
        parsed_origin = urlparse(origin)
        if parsed_origin.netloc == request.host:
            is_same_origin = True
            
    if origin in _ALLOWED_ORIGINS or is_same_origin:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        response.headers['Vary'] = 'Origin'
    return response

# -------------------------------------------------------
# Errors
# -------------------------------------------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

# -------------------------------------------------------
# -------------------------------------------------------
# System init — runs for both gunicorn (cloud) and direct python app.py (local)
# -------------------------------------------------------
from app_state import initialize_system
import threading as _threading
import os

_init_lock = _threading.Lock()
_init_pid = None
_init_thread = None

def start_system_init():
    global _init_pid, _init_thread
    with _init_lock:
        current_pid = os.getpid()
        if _init_pid != current_pid:
            _init_pid = current_pid
            print(f"[init] starting system-init thread in pid {current_pid}", flush=True)
            _init_thread = _threading.Thread(target=initialize_system, daemon=True, name="system-init")
            _init_thread.start()
    return _init_thread

# IMPORTANT: do NOT call start_system_init() at import time.
# Gunicorn's master imports this module before forking workers; a background
# thread started here runs in the MASTER, and forking while that thread holds
# import/SSL/threading locks deadlocks the worker's own init thread — the
# worker (the only process that serves requests) then reports "initializing"
# forever. Init is started per-worker via gunicorn.conf.py post_fork, with
# the before_request guard as a backup, and explicitly in __main__ for
# direct `python app.py` runs.

@app.before_request
def ensure_initialized_in_worker():
    start_system_init()

if __name__ == '__main__':
    import sys
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 60)
    print("🤖 RAG CHATBOT SERVER")
    print("=" * 60)
    print("🌐 Starting Flask server...")
    print("📱 Open http://localhost:9072 in your browser")

    # Ensure required directories exist
    embeddings_dir = Path(app_config.EMBEDDINGS_DIR)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    Path(app.config['SESSION_FILE_DIR']).mkdir(parents=True, exist_ok=True)

    # Log where we’re serving embedding assets from
    print(f"📂 EMBEDDINGS_DIR = {embeddings_dir.resolve()}")

    print("📄 System initializing in background thread...")
    init_thread = start_system_init()
    if init_thread is not None:
        init_thread.join(timeout=2)  # give it a moment to start

    try:
        app.run(host='0.0.0.0', port=9072, debug=False, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        raise
