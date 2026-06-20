"""
modules/auth.py
================
Authentication helpers.
Your original Flask session auth is preserved here for reference.
The live system uses JWT (app/core/security.py).
Both work side-by-side.
"""

from __future__ import annotations
import hashlib, secrets
from typing import Optional


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 + salt."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, stored: str) -> bool:
    """Verify a password against a stored hash."""
    try:
        # JWT bcrypt path
        from passlib.context import CryptContext
        ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return ctx.verify(password, stored)
    except Exception:
        pass
    # Legacy sha256 path
    try:
        if ":" in stored:
            salt, hashed = stored.split(":", 1)
            return hashed == hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    except Exception:
        pass
    return False


def generate_access_key() -> str:
    """Generate a numeric access key for document upload."""
    return str(secrets.randbelow(900000) + 100000)   # 6-digit key
