"""
auth.py — Authentication & Security Utilities

Responsibilities:
    - Password hashing and verification via bcrypt (direct)
    - JWT token creation and decoding (python-jose)
    - FastAPI dependency: get_current_user, require_admin
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from src.core.config import JWT_ALGORITHM, JWT_EXPIRE_HOURS, JWT_SECRET_KEY

logger = logging.getLogger(__name__)

# ── OAuth2 scheme — reads token from Authorization: Bearer <token> ────────────
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ── Password helpers ──────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Return a bcrypt hash of *plain* (truncated to 72 bytes as per bcrypt spec)."""
    return bcrypt.hashpw(plain[:72].encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches the bcrypt *hashed* string."""
    return bcrypt.checkpw(plain[:72].encode("utf-8"), hashed.encode("utf-8"))


# ── JWT helpers ───────────────────────────────────────────────────────────────

def create_access_token(data: dict[str, Any]) -> str:
    """Encode *data* as a signed JWT with an expiry of JWT_EXPIRE_HOURS hours.

    Args:
        data: Payload dict. Must include at minimum ``sub`` (username).

    Returns:
        Signed JWT string.
    """
    payload = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    payload["exp"] = expire
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT string.

    Args:
        token: The raw JWT string.

    Returns:
        Decoded payload dict.

    Raises:
        HTTPException(401): If the token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as exc:
        logger.warning("JWT decode failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI Dependencies ──────────────────────────────────────────────────────

def get_current_user(token: str = Depends(oauth2_scheme)) -> dict[str, Any]:
    """FastAPI dependency: decode the Bearer token and return the user payload.

    Inject this into any endpoint that requires authentication:
        current_user: dict = Depends(get_current_user)

    Returns dict with keys: sub (username), user_id (int), role (str).
    """
    return decode_token(token)


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """FastAPI dependency: same as get_current_user but enforces role == 'admin'.

    Raises:
        HTTPException(403): If the authenticated user is not an admin.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return current_user
