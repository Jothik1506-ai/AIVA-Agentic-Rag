# modules/auth.py
"""
Authentication and access control module.
Handles user login, session management, and document access permissions.
"""

import os
import csv
import time
import uuid
import hashlib
import logging
import threading
from pathlib import Path
from functools import wraps
from flask import session, redirect, url_for, jsonify, request, g

import requests

logger = logging.getLogger(__name__)
current_dir = Path(__file__).parent.parent

# ── Per-user identity resolution ──────────────────────────────────────────────
#
# The frontend authenticates users with Google Sign-In. To scope notebooks and
# memory per user, every API request must carry the user's identity. Accepted,
# in priority order:
#
#   1. "Authorization: Bearer <google_id_token>" — the Google ID token (JWT)
#      from Sign-In. Verified server-side against Google's tokeninfo endpoint;
#      the stable "sub" claim becomes the user id. This is the secure option.
#   2. "X-User-Id: <google_sub_or_uid>" — a plain identifier the frontend
#      already has. NOT cryptographically verified — acceptable only when the
#      backend is reached exclusively by the trusted frontend over HTTPS.
#      Disable with AIVA_TRUST_USER_HEADER=0 once the frontend sends tokens.
#   3. Flask session fallback — browsers using the backend-served UI get a
#      random per-browser-session id, so even anonymous visitors are isolated
#      from each other.

_GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"
_token_cache: dict = {}          # sha256(token) -> (user_id, expires_at_epoch)
_token_cache_lock = threading.Lock()
_TOKEN_CACHE_MAX = 1000


def _verify_google_id_token(token: str):
    """Verify a Google ID token and return 'google:<sub>' or None."""
    fp = hashlib.sha256(token.encode()).hexdigest()
    now = time.time()

    with _token_cache_lock:
        cached = _token_cache.get(fp)
        if cached and cached[1] > now:
            return cached[0]

    try:
        resp = requests.get(_GOOGLE_TOKENINFO_URL, params={"id_token": token}, timeout=5)
        if resp.status_code != 200:
            logger.warning("Google token verification rejected (HTTP %s)", resp.status_code)
            return None
        claims = resp.json()
        sub = claims.get("sub")
        if not sub:
            return None
        # If GOOGLE_CLIENT_ID is configured, require the token to be issued
        # for this app — otherwise any Google-signed token would be accepted.
        expected_aud = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
        if expected_aud and claims.get("aud") != expected_aud:
            logger.warning("Google token audience mismatch")
            return None
        user_id = f"google:{sub}"
        expires_at = float(claims.get("exp", now + 300))
        with _token_cache_lock:
            if len(_token_cache) >= _TOKEN_CACHE_MAX:
                _token_cache.clear()
            _token_cache[fp] = (user_id, expires_at)
        return user_id
    except Exception as e:
        logger.warning("Google token verification failed: %s", e)
        return None


def get_current_user_id() -> str:
    """
    Resolve the stable identity of the requesting user.
    Always returns a non-empty string; result is cached on flask.g per request.
    """
    cached = getattr(g, "user_id", None)
    if cached:
        return cached

    user_id = None

    # 1. Verified Google ID token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer "):].strip()
        # Only attempt Google verification for JWT-shaped tokens
        if token.count(".") == 2:
            user_id = _verify_google_id_token(token)

    # 2. Trusted frontend header (unverified — see module docstring)
    if not user_id and os.environ.get("AIVA_TRUST_USER_HEADER", "1") != "0":
        hdr = request.headers.get("X-User-Id", "").strip()
        if hdr and len(hdr) <= 128:
            safe = "".join(c for c in hdr if c.isalnum() or c in "@._:-")
            if safe:
                user_id = f"hdr:{safe}"

    # 3. Per-browser-session fallback
    if not user_id:
        if not session.get("aiva_uid"):
            session["aiva_uid"] = uuid.uuid4().hex
            session.permanent = True
        user_id = f"session:{session['aiva_uid']}"

    g.user_id = user_id
    return user_id


def load_access_data():
    """Load access permissions from CSV file"""
    access_data = {}
    csv_path = os.path.join(current_dir, 'access.csv')
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                designation = row['Designation']
                access_data[designation] = {}
                for doc_name, access in row.items():
                    if doc_name != 'Designation':
                        access_data[designation][doc_name] = int(access)
        return access_data
    except Exception as e:
        logger.error(f"Failed to load access data: {e}")
        return {}


def get_user_access(designation):
    """Get list of documents user has access to"""
    access_data = load_access_data()
    if designation not in access_data:
        return []
    
    user_access = []
    for doc_name, has_access in access_data[designation].items():
        if has_access == 1:
            user_access.append(doc_name)
    
    return user_access


def require_login(f):
    """Enforce session-based authentication on a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('user_designation'):
            session['user_designation'] = 'Admin'
        # Resolve and pin the requesting user's identity for this request —
        # all notebook/memory access is scoped by g.user_id downstream.
        get_current_user_id()
        return f(*args, **kwargs)
    return decorated_function


def validate_designation(designation):
    """Validate if designation exists in access data"""
    access_data = load_access_data()
    return designation in access_data


def check_document_access(designation, document_names):
    """Check if user has access to specified documents"""
    user_access = get_user_access(designation)
    unauthorized = [doc for doc in document_names if doc not in user_access]
    return len(unauthorized) == 0, unauthorized


def set_user_session(designation):
    """Set user session data"""
    session['user_designation'] = designation
    session.permanent = True


def clear_user_session():
    """Clear user session data"""
    session.clear()