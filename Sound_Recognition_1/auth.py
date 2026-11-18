"""SoundSense auth: one login for everyone, a `role` column decides developer access.

Wire it into main.py:

    from auth import router as auth_router, require_developer, get_current_user, save_prediction
    app.include_router(auth_router)

    # developer-only endpoints
    @app.post("/api/train", dependencies=[Depends(require_developer)])
    ...

    # in /api/predict: user is None for guests, a dict for logged-in people
    async def predict(..., user=Depends(get_current_user)):
        ...
        if user:
            save_prediction(user["id"], file.filename, model, [(label, confidence), ...])

Make a developer (nobody can pick this role themselves):
    python auth.py promote someone@example.com
    python auth.py demote  someone@example.com
    python auth.py list

Install:  pip install fastapi uvicorn bcrypt pyjwt python-multipart
"""
import json
import os
import re
import secrets
import sqlite3
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import bcrypt
import jwt
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel

DB_PATH = Path(os.environ.get("SOUNDSENSE_DB", Path(__file__).with_name("soundsense.db")))
SECRET_FILE = DB_PATH.with_name(".jwt_secret")
TOKEN_TTL_SECONDS = 7 * 24 * 3600
ROLES = ("user", "developer")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _load_secret() -> str:
    env = os.environ.get("SOUNDSENSE_SECRET")
    if env:
        return env
    if not SECRET_FILE.exists():
        SECRET_FILE.write_text(secrets.token_hex(32))
    return SECRET_FILE.read_text().strip()


SECRET = _load_secret()


# ───────────────────────── database ─────────────────────────
@contextmanager
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with db() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT NOT NULL UNIQUE,
                name          TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role          TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'developer')),
                created_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                file_name  TEXT NOT NULL,
                model      TEXT NOT NULL,
                top1_label TEXT NOT NULL,
                top1_conf  REAL NOT NULL,
                top5_json  TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id, id DESC);
            """
        )


init_db()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ───────────────────────── passwords and tokens ─────────────────────────
_DUMMY_HASH = bcrypt.hashpw(b"dummy-password", bcrypt.gensalt())


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, stored: Optional[str]) -> bool:
    # Always run a check so a missing account takes as long as a wrong password.
    target = stored.encode() if stored else _DUMMY_HASH
    try:
        ok = bcrypt.checkpw(password.encode()[:72], target)
    except ValueError:
        return False
    return ok and stored is not None


def make_token(user_id: int) -> str:
    now = int(time.time())
    return jwt.encode({"sub": str(user_id), "iat": now, "exp": now + TOKEN_TTL_SECONDS}, SECRET, algorithm="HS256")


def public_user(row) -> dict:
    return {"id": row["id"], "name": row["name"], "email": row["email"], "role": row["role"]}


# ───────────────────────── dependencies ─────────────────────────
def get_current_user(authorization: Optional[str] = Header(default=None)) -> Optional[dict]:
    """Returns the logged-in user, or None for guests / bad tokens."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, SECRET, algorithms=["HS256"])
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        return None
    # Role is read from the database on every request, so promoting or demoting takes effect at once.
    with db() as con:
        row = con.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return public_user(row) if row else None


def require_user(user: Optional[dict] = Depends(get_current_user)) -> dict:
    if not user:
        raise HTTPException(status_code=401, detail="Please sign in.")
    return user


def require_developer(user: dict = Depends(require_user)) -> dict:
    if user["role"] != "developer":
        raise HTTPException(status_code=403, detail="Developer access required.")
    return user


# ───────────────────────── schemas ─────────────────────────
class SignupIn(BaseModel):
    name: str
    email: str
    password: str


class LoginIn(BaseModel):
    username: str  # the email address (kept as "username" to match the existing frontend payload)
    password: str


# ───────────────────────── routes ─────────────────────────
router = APIRouter()


@router.post("/api/auth/signup", status_code=201)
def signup(body: SignupIn):
    name = body.name.strip()
    email = body.email.strip().lower()
    if not 1 <= len(name) <= 60:
        raise HTTPException(400, "Enter your name (up to 60 characters).")
    if not EMAIL_RE.match(email):
        raise HTTPException(400, "Enter a valid email address.")
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")
    if len(body.password.encode()) > 72:
        raise HTTPException(400, "Password must be 72 bytes or fewer.")
    try:
        with db() as con:
            # The role is always 'user' here. It is never taken from the request.
            cur = con.execute(
                "INSERT INTO users (email, name, password_hash, role, created_at) VALUES (?, ?, ?, 'user', ?)",
                (email, name, hash_password(body.password), _now()),
            )
            row = con.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
    except sqlite3.IntegrityError:
        raise HTTPException(409, "An account with this email already exists.")
    return {"token": make_token(row["id"]), "user": public_user(row)}


@router.post("/api/auth/login")
def login(body: LoginIn):
    email = body.username.strip().lower()
    with db() as con:
        row = con.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if not verify_password(body.password, row["password_hash"] if row else None):
        raise HTTPException(401, "Invalid email or password.")
    return {"token": make_token(row["id"]), "user": public_user(row)}


@router.post("/api/auth/logout")
def logout():
    # Tokens are stateless; the browser discards its copy.
    return {"ok": True}


@router.get("/api/me")
def me(user: dict = Depends(require_user)):
    return user


@router.get("/api/history")
def history(limit: int = Query(100, ge=1, le=500), user: dict = Depends(require_user)):
    with db() as con:
        rows = con.execute(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user["id"], limit)
        ).fetchall()
    return {
        "history": [
            {
                "id": r["id"],
                "file_name": r["file_name"],
                "model": r["model"],
                "top1_label": r["top1_label"],
                "top1_conf": r["top1_conf"],
                "top5": json.loads(r["top5_json"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    }


@router.delete("/api/history")
def clear_history(user: dict = Depends(require_user)):
    with db() as con:
        con.execute("DELETE FROM predictions WHERE user_id = ?", (user["id"],))
    return {"ok": True}


def save_prediction(user_id: int, file_name: str, model: str, preds) -> None:
    """preds: ranked list of (label, confidence) with confidence as a fraction (0 to 1) or percent."""
    top5 = [{"label": str(l), "confidence": float(c)} for l, c in list(preds)[:5]]
    if not top5:
        return
    with db() as con:
        con.execute(
            "INSERT INTO predictions (user_id, file_name, model, top1_label, top1_conf, top5_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, file_name or "audio", model or "", top5[0]["label"], top5[0]["confidence"], json.dumps(top5), _now()),
        )


# ───────────────────────── command line ─────────────────────────
def _set_role(email: str, role: str) -> None:
    with db() as con:
        cur = con.execute("UPDATE users SET role = ? WHERE email = ?", (role, email.strip().lower()))
    print(f"{email}: now '{role}'" if cur.rowcount else f"No account found for {email}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd in ("promote", "demote") and len(sys.argv) == 3:
        _set_role(sys.argv[2], "developer" if cmd == "promote" else "user")
    elif cmd == "list":
        with db() as con:
            for r in con.execute("SELECT id, email, name, role, created_at FROM users ORDER BY id"):
                print(f"{r['id']:>3}  {r['role']:<9}  {r['email']}  ({r['name']})")
    else:
        print("usage: python auth.py promote|demote EMAIL   |   python auth.py list")