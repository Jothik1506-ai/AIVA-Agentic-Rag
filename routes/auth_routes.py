# routes/auth_routes.py
"""
Authentication routes module.
Handles login, logout, and access data endpoints.
"""

import os
import logging
import time
import threading
from flask import Blueprint, request, jsonify, render_template, session, redirect, url_for
from modules.auth import (
    load_access_data,
    validate_designation,
    set_user_session,
    clear_user_session,
    require_login,
    create_local_user,
    verify_local_user,
    login_session_user,
    google_login_from_credential,
)

logger = logging.getLogger(__name__)

auth_bp = Blueprint('auth', __name__)

# ── Simple in-memory rate limiter for login ───────────────────────────────────
_login_attempts: dict = {}
_login_lock = threading.Lock()
_RATE_LIMIT_WINDOW = 60   # seconds
_RATE_LIMIT_MAX    = 5    # attempts per window per IP

def _login_rate_limit_ok(ip: str) -> bool:
    now = time.time()
    with _login_lock:
        history = [t for t in _login_attempts.get(ip, []) if now - t < _RATE_LIMIT_WINDOW]
        if len(history) >= _RATE_LIMIT_MAX:
            _login_attempts[ip] = history
            return False
        history.append(now)
        _login_attempts[ip] = history
        return True


@auth_bp.route('/')
def index():
    """Redirect directly to AIVA chat"""
    return redirect(url_for('auth.aiva'))


@auth_bp.route('/login')
def login():
    """Serve the AIVA login page (local accounts + optional Google Sign-In)."""
    if session.get('aiva_user_id'):
        return redirect(url_for('auth.aiva'))
    try:
        return render_template(
            'aiva_login.html',
            google_client_id=os.environ.get('GOOGLE_CLIENT_ID', ''),
        )
    except Exception as e:
        return f"Template error: {e}", 500


@auth_bp.route('/chatbot')
@require_login
def chatbot():
    """Serve the main chatbot interface"""
    try:
        from flask import make_response
        response = make_response(render_template('chatbot_new.html'))
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    except Exception as e:
        return f"Template error: {e}", 500


@auth_bp.route('/aiva')
def aiva():
    """Serve the AIVA AI chat interface — requires a logged-in user."""
    if not session.get('aiva_user_id'):
        return redirect(url_for('auth.login'))
    if not session.get('user_designation'):
        session['user_designation'] = 'Admin'
    display_name = session.get('aiva_display_name') or \
        session.get('aiva_user_id', '').split(':', 1)[-1] or 'AIVA User'
    try:
        from flask import make_response
        response = make_response(render_template(
            'graphrag_chat_ui.html',
            user_name=display_name,
            user_initial=(display_name[:1] or 'A').upper(),
        ))
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    except Exception as e:
        return f"Template error: {e}", 500


@auth_bp.route('/api/login', methods=['POST'])
def api_login():
    """Handle user login"""
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr or '').split(',')[0].strip()
    if not _login_rate_limit_ok(client_ip):
        return jsonify({'error': 'Too many login attempts. Please wait and try again.'}), 429
    try:
        data = request.get_json()
        if not data or 'designation' not in data:
            return jsonify({'error': 'No designation provided'}), 400
        
        designation = data['designation'].strip()
        if not designation:
            return jsonify({'error': 'Empty designation'}), 400
        
        if not validate_designation(designation):
            return jsonify({'error': 'Invalid designation'}), 400
        
        set_user_session(designation)
        
        return jsonify({
            'message': 'Login successful',
            'designation': designation,
            'redirect_url': url_for('auth.chatbot')
        })
        
    except Exception as e:
        logger.error(f"Login API error: {e}")
        return jsonify({'error': 'Login failed'}), 500


# ── AIVA account auth (login page) ───────────────────────────────────────────

@auth_bp.route('/api/auth/register', methods=['POST'])
def api_auth_register():
    """Create a local account and log the user in."""
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr or '').split(',')[0].strip()
    if not _login_rate_limit_ok(client_ip):
        return jsonify({'error': 'Too many attempts. Please wait and try again.'}), 429
    data = request.get_json(silent=True) or {}
    user_id, err = create_local_user(data.get('username', ''), data.get('password', ''))
    if err:
        return jsonify({'error': err}), 400
    login_session_user(user_id, user_id.split(':', 1)[-1])
    return jsonify({'message': 'Account created', 'user_id': user_id,
                    'redirect_url': url_for('auth.aiva')}), 201


@auth_bp.route('/api/auth/login', methods=['POST'])
def api_auth_login():
    """Log in with a local account."""
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr or '').split(',')[0].strip()
    if not _login_rate_limit_ok(client_ip):
        return jsonify({'error': 'Too many attempts. Please wait and try again.'}), 429
    data = request.get_json(silent=True) or {}
    user_id = verify_local_user(data.get('username', ''), data.get('password', ''))
    if not user_id:
        return jsonify({'error': 'Invalid username or password'}), 401
    login_session_user(user_id, user_id.split(':', 1)[-1])
    return jsonify({'message': 'Login successful', 'user_id': user_id,
                    'redirect_url': url_for('auth.aiva')})


@auth_bp.route('/api/auth/google', methods=['POST'])
def api_auth_google():
    """Log in with a Google Sign-In credential (ID token), verified server-side."""
    data = request.get_json(silent=True) or {}
    verified = google_login_from_credential(data.get('credential', ''))
    if not verified:
        return jsonify({'error': 'Google sign-in could not be verified'}), 401
    user_id, display_name = verified
    login_session_user(user_id, display_name)
    return jsonify({'message': 'Login successful', 'user_id': user_id,
                    'redirect_url': url_for('auth.aiva')})


@auth_bp.route('/api/auth/me', methods=['GET'])
def api_auth_me():
    """Return the logged-in user (login page uses this to skip itself)."""
    uid = session.get('aiva_user_id')
    return jsonify({'logged_in': bool(uid), 'user_id': uid})


@auth_bp.route('/logout')
def logout_page():
    """Log out and return to the login page."""
    clear_user_session()
    return redirect(url_for('auth.login'))


@auth_bp.route('/api/logout', methods=['POST'])
def api_logout():
    """Handle user logout"""
    try:
        clear_user_session()
        return jsonify({'message': 'Logged out successfully', 'redirect_url': url_for('auth.login')})
    except Exception as e:
        logger.error(f"Logout API error: {e}")
        return jsonify({'error': 'Logout failed'}), 500


@auth_bp.route('/api/access-data')
def api_access_data():
    """Return valid designation names for the login page dropdown — no access matrix."""
    try:
        access_data = load_access_data()
        return jsonify({'designations': list(access_data.keys())})
    except Exception as e:
        logger.error(f"Access data API error: {e}")
        return jsonify({'error': 'Failed to load access data'}), 500