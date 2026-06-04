"""subnet-server: a single-file user directory for a tuwunel-backed subnet.

Everything lives in this file. There is no background loop, no management
agent, and no frontend. The service does three things:

  1. Auth: stateless ETH-signature auth. Clients sign the static SIGN_MESSAGE
     (`{domain}-matrix-auth`) with their Ethereum private key and pass
     `{address, signature}` in every authed request body. This mirrors what the
     `subnet-client` SDK expects.

  2. Matrix provisioning: when a user is created, we register their account on
     the tuwunel homeserver and invite/join them into the configured rooms
     (default: "General"). The Steward (the ETH-keyed admin from .env) owns the
     rooms and issues invites.

  3. Admin API: admin-privileged users (role == "admin") can add and remove
     users over the HTTP API. There is no self-service join — an admin adds an
     ETH address, and that address becomes a member.

Usage:
    python main.py    # serve the API on API_PORT
"""
from __future__ import annotations

import json
import os
import secrets
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from urllib.parse import urlparse

import psycopg2
import requests
import uvicorn
from dotenv import load_dotenv
from eth_account import Account
from eth_account.messages import encode_defunct
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from peewee import (
    BooleanField,
    CharField,
    DateTimeField,
    InterfaceError,
    Model,
    OperationalError,
    PostgresqlDatabase,
    TextField,
)
from playhouse.migrate import PostgresqlMigrator, migrate
from playhouse.shortcuts import ReconnectMixin
from psycopg2 import sql

load_dotenv()

DB_NAME = os.environ.get("DB_NAME", "subnet_server")
PG_HOST = os.environ.get("DB_HOST", "localhost")
API_PORT = int(os.environ.get("API_PORT", "34182"))

MATRIX_URL = os.environ.get("MATRIX_URL", "http://localhost:6167")
MATRIX_SERVER_NAME = os.environ.get("MATRIX_SERVER_NAME", "localhost")
MATRIX_REGISTRATION_TOKEN = os.environ.get("MATRIX_REGISTRATION_TOKEN", "")
MATRIX_ADMIN_ROOM_ID = os.environ.get("MATRIX_ADMIN_ROOM_ID", "")
SUBNET_API_BASE = os.environ.get("SUBNET_API_BASE", "")

STEWARD_NAME = os.environ.get("STEWARD_NAME", "Steward")
# The Steward IS the ETH-keyed admin from .env — one identity at the protocol
# layer (signs subnet messages with this wallet) and the chat layer (Matrix
# account is @{address}:{MATRIX_SERVER_NAME}). It owns the rooms and issues
# invites when new users are created.
STEWARD_ACCOUNT = Account.from_key(os.environ["ETH_PRIVATE_KEY"])
STEWARD_ADDRESS = STEWARD_ACCOUNT.address.lower()

# Rooms every new user is invited to and force-joined into on creation.
AUTO_JOIN_ROOMS = [
    r.strip() for r in os.environ.get("AUTO_JOIN_ROOMS", "General").split(",") if r.strip()
]


def _derive_sign_message() -> str:
    host = urlparse(SUBNET_API_BASE).hostname or "localhost"
    domain = host[len("subnet."):] if host.startswith("subnet.") else host
    return f"{domain}-matrix-auth"


SIGN_MESSAGE = _derive_sign_message()


# ────────────────────────── models ──────────────────────────
class ReconnectingPostgresqlDatabase(ReconnectMixin, PostgresqlDatabase):
    reconnect_errors = (
        (OperationalError, "terminat"),
        (OperationalError, "closed"),
        (OperationalError, "could not connect"),
        (InterfaceError, "connection already closed"),
        (InterfaceError, "cursor already closed"),
    )


db = ReconnectingPostgresqlDatabase(DB_NAME, host=PG_HOST)


class BaseModel(Model):
    class Meta:
        database = db


class User(BaseModel):
    address = CharField(unique=True)              # lowercase ETH address (0x…)
    matrix_password = CharField(default="")       # password for the Matrix account
    matrix_access_token = TextField(default="")
    name = CharField(default="")
    description = TextField(default="")
    role = CharField(default="user")              # user | admin
    matrix_synced = BooleanField(default=False)
    created_at = DateTimeField(default=datetime.utcnow)


def ensure_database() -> None:
    conn = psycopg2.connect(dbname="postgres", host=PG_HOST)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
        if not cur.fetchone():
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
    conn.close()


def auto_migrate() -> None:
    migrator = PostgresqlMigrator(db)
    for model in [User]:
        table = model._meta.table_name
        cur = db.execute_sql(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
            (table,),
        )
        existing = {row[0] for row in cur.fetchall()}
        for field in model._meta.sorted_fields:
            col = field.column_name
            if col not in existing:
                print(f"[migrate] adding {table}.{col}")
                migrate(migrator.add_column(table, col, field))


def init_db() -> None:
    ensure_database()
    db.connect(reuse_if_open=True)
    db.create_tables([User])
    auto_migrate()


# ────────────────────────── matrix admin (tuwunel) ──────────────────────────
def matrix_login(address: str, password: str) -> str:
    """Password login for an existing Matrix user. Returns access_token."""
    r = requests.post(
        f"{MATRIX_URL}/_matrix/client/v3/login",
        json={
            "type": "m.login.password",
            "identifier": {"type": "m.id.user", "user": address},
            "password": password,
        },
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"login {r.status_code}: {r.text[:200]}")
    return r.json()["access_token"]


def matrix_register(address: str, password: str) -> str:
    """Register a Matrix user via the registration_token flow. Returns access_token.

    First POST returns 401 with a session id; second POST completes auth.
    If the account already exists, falls back to password login so we recover a
    token rather than crashing.
    """
    url = f"{MATRIX_URL}/_matrix/client/v3/register"
    r1 = requests.post(url, json={"username": address, "password": password, "inhibit_login": False}, timeout=30)
    if r1.status_code == 200:
        return r1.json()["access_token"]
    if r1.status_code == 400 and r1.json().get("errcode") == "M_USER_IN_USE":
        return matrix_login(address, password)
    if r1.status_code != 401:
        raise RuntimeError(f"register init {r1.status_code}: {r1.text[:200]}")
    session = r1.json().get("session", "")
    body = {
        "auth": {"type": "m.login.registration_token", "token": MATRIX_REGISTRATION_TOKEN, "session": session},
        "username": address,
        "password": password,
        "inhibit_login": False,
    }
    r2 = requests.post(url, json=body, timeout=30)
    if r2.status_code != 200:
        raise RuntimeError(f"register {r2.status_code}: {r2.text[:200]}")
    return r2.json()["access_token"]


def matrix_set_displayname(access_token: str, user_id: str, name: str) -> None:
    r = requests.put(
        f"{MATRIX_URL}/_matrix/client/v3/profile/{user_id}/displayname",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"displayname": name},
        timeout=30,
    )
    if r.status_code not in (200, 204):
        print(f"[matrix] set_displayname {user_id}: {r.status_code} {r.text[:120]}")


def matrix_deactivate_user(steward_token: str, user_id: str, admin_room_id: str) -> None:
    """Deactivate (delete) a tuwunel Matrix account via the admin command room.

    `user_id` is the full MXID (`@address:server`). Deactivation is tuwunel's
    account-removal operation: it erases credentials and evicts the user from
    its rooms — there is no harder "delete" in the Matrix protocol.
    """
    txn_id = secrets.token_hex(16)
    r = requests.put(
        f"{MATRIX_URL}/_matrix/client/v3/rooms/{admin_room_id}/send/m.room.message/{txn_id}",
        headers={"Authorization": f"Bearer {steward_token}"},
        json={"msgtype": "m.text", "body": f"!admin users deactivate {user_id}"},
        timeout=30,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"deactivate {user_id}: {r.status_code} {r.text[:200]}")


# ────────────────────────── steward + rooms bootstrap ──────────────────────────
def steward_password() -> str:
    """Deterministic Matrix password for the Steward: its own signature over
    SIGN_MESSAGE. We hold the private key, so we can always re-derive it."""
    signed = STEWARD_ACCOUNT.sign_message(encode_defunct(text=SIGN_MESSAGE))
    sig = signed.signature.hex()
    return sig if sig.startswith("0x") else "0x" + sig


def ensure_steward() -> User:
    """Make sure the Steward exists in the DB and on Matrix, and return the row.
    The Steward owns the auto-join rooms and issues invites for new users."""
    user = User.get_or_none(User.address == STEWARD_ADDRESS)
    if user is None:
        user = User.create(
            address=STEWARD_ADDRESS,
            matrix_password=steward_password(),
            name=STEWARD_NAME,
            role="admin",
        )
    elif user.role != "admin" or user.name != STEWARD_NAME:
        user.role = "admin"
        user.name = STEWARD_NAME
        user.save()

    if not user.matrix_synced or not user.matrix_access_token:
        token = matrix_register(user.address, user.matrix_password)
        user.matrix_access_token = token
        user.matrix_synced = True
        user.save()
        matrix_set_displayname(token, f"@{user.address}:{MATRIX_SERVER_NAME}", STEWARD_NAME)
    return user


# ────────────────────────── subnet CLI (acts as the Steward) ──────────────────────────
# Room creation and invites go through the locally installed `subnet` utility,
# which authenticates as the ETH-keyed admin agent (ETH_PRIVATE_KEY +
# SUBNET_API_BASE from the environment). This is the same identity as the
# Steward, so the CLI creates/owns the auto-join rooms and issues their invites.
def _subnet(args: list[str]) -> subprocess.CompletedProcess:
    env = {**os.environ, "SUBNET_SIGN_MESSAGE": SIGN_MESSAGE}
    return subprocess.run(["subnet", *args], capture_output=True, text=True, timeout=60, env=env)


def subnet_joined_rooms() -> dict[str, str]:
    """{room_name: room_id} for rooms the Steward has joined."""
    p = _subnet(["joined-rooms", "--no-spaces"])
    if p.returncode != 0:
        raise RuntimeError(f"subnet joined-rooms failed: {(p.stderr or p.stdout)[:200]}")
    return {r["name"]: r["room_id"] for r in json.loads(p.stdout or "[]")}


def subnet_create_room(name: str) -> str:
    p = _subnet(["create-room", "--name", name, "--public", "--unencrypted"])
    if p.returncode != 0:
        raise RuntimeError(f"subnet create-room {name} failed: {(p.stderr or p.stdout)[:200]}")
    return json.loads(p.stdout)["room_id"]


def subnet_invite_user(room_id: str, user_id: str) -> None:
    p = _subnet(["invite-user", room_id, user_id])
    if p.returncode != 0:
        raise RuntimeError(f"subnet invite-user {user_id} → {room_id} failed: {(p.stderr or p.stdout)[:200]}")


def ensure_rooms() -> dict[str, str]:
    """Resolve {room_name: room_id} for the auto-join rooms via the CLI,
    creating any the Steward hasn't already (e.g. General on first use)."""
    joined = subnet_joined_rooms()
    rooms: dict[str, str] = {}
    for name in AUTO_JOIN_ROOMS:
        room_id = joined.get(name)
        if not room_id:
            room_id = subnet_create_room(name)
            print(f"[subnet] created room {name} → {room_id}")
        rooms[name] = room_id
    return rooms


# ────────────────────────── FastAPI ──────────────────────────
@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    # Register the Steward's Matrix account so the `subnet` CLI can later log in
    # as the admin agent. Rooms are ensured lazily on the first add_user, since
    # the CLI talks to this server, which isn't accepting requests yet here.
    ensure_steward()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_sig(address: str, signature: str) -> bool:
    msg = encode_defunct(text=SIGN_MESSAGE)
    recovered = Account.recover_message(msg, signature=signature)
    return recovered.lower() == address.lower()


async def authed(request: Request) -> tuple[User, dict]:
    body = await request.json()
    address = (body.get("address") or "").strip().lower()
    signature = body.get("signature") or ""
    if not address or not signature:
        raise HTTPException(400, "address and signature required")
    if not verify_sig(address, signature):
        raise HTTPException(401, "invalid signature")
    user = User.get_or_none(User.address == address)
    if user is None:
        raise HTTPException(404, "user not registered")
    return user, body


def require_admin(user: User) -> None:
    if user.role != "admin":
        raise HTTPException(403, "admin only")


def user_row(u: User) -> dict:
    return {
        "address": u.address,
        "name": u.name,
        "description": u.description,
        "role": u.role,
        "matrix_synced": u.matrix_synced,
        "matrix_id": f"@{u.address}:{MATRIX_SERVER_NAME}",
        "created_at_ms": int(u.created_at.replace(tzinfo=timezone.utc).timestamp() * 1000),
    }


def credentials_row(u: User) -> dict:
    return {
        "address": u.address,
        "matrix_url": MATRIX_URL,
        "matrix_username": u.address,
        "matrix_password": u.matrix_password,
    }


def provision_matrix_user(user: User) -> None:
    """Register the user on tuwunel and invite them (as the Steward, via the
    `subnet` CLI) into every auto-join room. The user accepts the invites with
    their own client — we don't force-join on their behalf."""
    ensure_steward()
    rooms = ensure_rooms()

    token = matrix_register(user.address, user.matrix_password)
    user.matrix_access_token = token
    user.matrix_synced = True
    user.save()

    matrix_id = f"@{user.address}:{MATRIX_SERVER_NAME}"
    matrix_set_displayname(token, matrix_id, user.name or user.address)
    for name, room_id in rooms.items():
        subnet_invite_user(room_id, matrix_id)
        print(f"[subnet] invited {matrix_id} → {name}")


# ────────────────────────── public / user endpoints ──────────────────────────
@app.get("/ping")
async def ping():
    return {"ok": True, "sign_message": SIGN_MESSAGE}


@app.post("/api/credentials")
async def credentials(request: Request):
    body = await request.json()
    address = (body.get("address") or "").strip().lower()
    signature = body.get("signature") or ""
    if not address or not signature:
        raise HTTPException(400, "address and signature required")
    if not verify_sig(address, signature):
        raise HTTPException(401, "invalid signature")
    user = User.get_or_none(User.address == address)
    if user is None:
        raise HTTPException(404, "user not registered")
    return credentials_row(user)


@app.post("/api/me")
async def me(request: Request):
    user, _ = await authed(request)
    return user_row(user)


@app.post("/api/update_profile")
async def update_profile(request: Request):
    user, body = await authed(request)
    if "name" in body:
        user.name = (body["name"] or "")[:120]
    if "description" in body:
        user.description = (body["description"] or "")[:2000]
    user.save()
    return user_row(user)


# ────────────────────────── admin endpoints ──────────────────────────
@app.post("/api/admin/users")
async def list_users(request: Request):
    user, _ = await authed(request)
    require_admin(user)
    return {"users": [user_row(u) for u in User.select().order_by(User.created_at)]}


@app.post("/api/admin/add_user")
async def add_user(request: Request):
    """Admin adds a member by ETH address. We generate a Matrix password,
    provision the account on tuwunel, and invite them into the auto-join rooms.
    Returns the new user plus their Matrix credentials."""
    requester, body = await authed(request)
    require_admin(requester)

    address = (body.get("address") or "").strip().lower()
    if not (address.startswith("0x") and len(address) == 42):
        raise HTTPException(400, "address must be a 0x-prefixed ETH address")
    if User.get_or_none(User.address == address) is not None:
        raise HTTPException(409, "address already registered")

    role = body.get("role", "user")
    if role not in ("user", "admin"):
        raise HTTPException(400, "role must be 'user' or 'admin'")
    name = (body.get("name") or "").strip()[:120] or address

    user = User.create(
        address=address,
        matrix_password=secrets.token_urlsafe(24),
        name=name,
        role=role,
    )
    try:
        provision_matrix_user(user)
    except Exception as e:
        user.delete_instance()
        raise HTTPException(502, f"matrix provisioning failed: {e}")

    return {"user": user_row(user), "credentials": credentials_row(user)}


@app.post("/api/admin/remove_user")
async def remove_user(request: Request):
    """Admin removes a member: deactivate their Matrix account, delete the row."""
    requester, body = await authed(request)
    require_admin(requester)

    target_address = (body.get("target_address") or "").strip().lower()
    if not target_address:
        raise HTTPException(400, "target_address required")
    if target_address == STEWARD_ADDRESS:
        raise HTTPException(400, "cannot remove the steward")

    target = User.get_or_none(User.address == target_address)
    if target is None:
        raise HTTPException(404, "user not found")

    # Delete from tuwunel first; only drop the DB row if that succeeds, so we
    # never leave an orphaned Matrix account behind.
    if target.matrix_synced:
        if not MATRIX_ADMIN_ROOM_ID:
            raise HTTPException(500, "MATRIX_ADMIN_ROOM_ID not configured; cannot deactivate Matrix account")
        steward = ensure_steward()
        try:
            matrix_deactivate_user(
                steward.matrix_access_token,
                f"@{target_address}:{MATRIX_SERVER_NAME}",
                MATRIX_ADMIN_ROOM_ID,
            )
        except Exception as e:
            raise HTTPException(502, f"matrix deactivation failed: {e}")
    target.delete_instance()
    return {"removed": target_address}


# ────────────────────────── entrypoint ──────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
