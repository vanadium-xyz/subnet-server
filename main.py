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

import asyncio
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
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
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

# Where the uploaded subnet icon (a png/jpg) is stored on disk. Served at
# /api/icon and surfaced as an absolute url by /api/metadata.
ICON_DIR = os.environ.get("ICON_DIR", os.path.join(os.path.dirname(__file__), "icon"))

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

# Additional ETH addresses that get role=admin at startup. The Steward is always
# an admin; this is for off-host agents (e.g. the bottles provisioner) that need
# to call admin endpoints by signing as themselves. They get a DB row and admin
# role, but no Matrix account is provisioned — admin endpoints use the Steward
# identity for any Matrix-side work.
EXTRA_ADMINS = [
    a.strip().lower() for a in os.environ.get("EXTRA_ADMINS", "").split(",") if a.strip()
]


def _derive_domain() -> str:
    host = urlparse(SUBNET_API_BASE).hostname or "localhost"
    return host[len("subnet."):] if host.startswith("subnet.") else host


SUBNET_DOMAIN = _derive_domain()
SIGN_MESSAGE = f"{SUBNET_DOMAIN}-matrix-auth"
# Human-facing name for the subnet, returned by /api/metadata. Defaults to the
# domain when unset.
SUBNET_NAME = os.environ.get("SUBNET_NAME", "") or SUBNET_DOMAIN


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
    avatar_url = CharField(default="")            # source http(s) link to the png/jpg profile icon
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


def matrix_upload_avatar(access_token: str, image_url: str) -> str:
    """Download a PNG/JPG from image_url and upload it to the Matrix media repo.

    Returns the resulting `mxc://` content URI. The image type is taken from the
    response Content-Type, falling back to the link's extension; only png/jpeg
    are accepted (what Matrix clients render reliably as avatars)."""
    resp = requests.get(image_url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"fetch avatar {image_url}: {resp.status_code}")

    content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if content_type not in ("image/png", "image/jpeg"):
        ext = image_url.lower().rsplit(".", 1)[-1]
        content_type = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "")
        if not content_type:
            raise RuntimeError(f"avatar must be a png or jpg: {image_url}")

    filename = "avatar.png" if content_type == "image/png" else "avatar.jpg"
    up = requests.post(
        f"{MATRIX_URL}/_matrix/media/v3/upload",
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": content_type},
        params={"filename": filename},
        data=resp.content,
        timeout=60,
    )
    if up.status_code != 200:
        raise RuntimeError(f"upload avatar: {up.status_code} {up.text[:200]}")
    return up.json()["content_uri"]


def matrix_set_avatar(access_token: str, user_id: str, image_url: str) -> str:
    """Upload image_url to the media repo and set it as user_id's avatar.
    Returns the `mxc://` URI that was set."""
    mxc = matrix_upload_avatar(access_token, image_url)
    r = requests.put(
        f"{MATRIX_URL}/_matrix/client/v3/profile/{user_id}/avatar_url",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"avatar_url": mxc},
        timeout=30,
    )
    if r.status_code not in (200, 204):
        raise RuntimeError(f"set_avatar {user_id}: {r.status_code} {r.text[:200]}")
    return mxc


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


def ensure_extra_admins() -> None:
    """Bootstrap each EXTRA_ADMINS address as a role=admin user. Creates the DB
    row if missing and upgrades the role if it had drifted. No Matrix account
    is provisioned for them; admin endpoints that touch Matrix use the Steward."""
    for address in EXTRA_ADMINS:
        if not (address.startswith("0x") and len(address) == 42):
            print(f"[admin] skipping invalid EXTRA_ADMINS entry: {address}")
            continue
        user = User.get_or_none(User.address == address)
        if user is None:
            User.create(
                address=address,
                matrix_password=secrets.token_urlsafe(24),
                name=address,
                role="admin",
            )
            print(f"[admin] created extra admin {address}")
        elif user.role != "admin":
            user.role = "admin"
            user.save()
            print(f"[admin] promoted {address} to admin")


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
    ensure_extra_admins()
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


def clean_avatar_link(raw: str | None) -> str:
    """Normalise a user-supplied profile-icon link. Empty is allowed (clears it);
    anything non-empty must be an http(s) URL — the png/jpg content type itself is
    checked when we fetch and upload it to Matrix."""
    link = (raw or "").strip()
    if not link:
        return ""
    if not link.lower().startswith(("http://", "https://")):
        raise HTTPException(400, "avatar_url must be an http(s) link to a png/jpg")
    return link[:1000]


def require_admin(user: User) -> None:
    if user.role != "admin":
        raise HTTPException(403, "admin only")


def require_local(request: Request) -> None:
    """Branding is operator-only: it carries no ETH signature, so we accept it
    only from the loopback interface (the steward running on the box)."""
    host = request.client.host if request.client else ""
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(403, "branding is local-only")


def icon_path() -> str | None:
    """Disk path of the current subnet icon, or None if none has been uploaded."""
    for name in ("icon.png", "icon.jpg"):
        p = os.path.join(ICON_DIR, name)
        if os.path.exists(p):
            return p
    return None


def icon_url() -> str:
    """Absolute url for the subnet icon, or "" when none is set. Absolute so the
    bottles frontend can load it cross-origin from `subnet.<domain>`."""
    if not icon_path():
        return ""
    base = SUBNET_API_BASE.rstrip("/")
    return f"{base}/api/icon" if base else "/api/icon"


def stored_name() -> str:
    """Operator-set subnet name (via /admin), or "" when none has been set."""
    p = os.path.join(ICON_DIR, "name.txt")
    if not os.path.exists(p):
        return ""
    with open(p) as f:
        return f.read().strip()


def subnet_name() -> str:
    """The subnet's display name: an operator override if set, else SUBNET_NAME."""
    return stored_name() or SUBNET_NAME


def save_icon(data: bytes, filename: str, content_type: str) -> None:
    """Persist uploaded image bytes as the subnet icon, replacing any existing one.
    Accepts png/jpg, inferring from the content type and falling back to the
    filename extension."""
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct not in ("image/png", "image/jpeg"):
        ext = (filename or "").lower().rsplit(".", 1)[-1]
        ct = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "")
        if not ct:
            raise HTTPException(400, "icon must be a png or jpg")
    os.makedirs(ICON_DIR, exist_ok=True)
    for name in ("icon.png", "icon.jpg"):
        old = os.path.join(ICON_DIR, name)
        if os.path.exists(old):
            os.remove(old)
    dest = os.path.join(ICON_DIR, "icon.png" if ct == "image/png" else "icon.jpg")
    with open(dest, "wb") as f:
        f.write(data)


def user_row(u: User) -> dict:
    return {
        "address": u.address,
        "name": u.name,
        "description": u.description,
        "avatar_url": u.avatar_url,
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
    if user.avatar_url:
        matrix_set_avatar(token, matrix_id, user.avatar_url)
    for name, room_id in rooms.items():
        subnet_invite_user(room_id, matrix_id)
        print(f"[subnet] invited {matrix_id} → {name}")


# ────────────────────────── landing page ──────────────────────────
def landing_html() -> str:
    """A static, dependency-free explainer served at `/`. No user data is
    rendered, so it is safe to serve unauthenticated."""
    return (
        "<!doctype html><meta charset=\"utf-8\">"
        f"<title>{STEWARD_NAME}'s subnet directory</title>"
        "<p>This is the user directory for a subnet — a small, accountable network where "
        "humans and AI agents collaborate over Matrix chat. Membership is keyed to Ethereum "
        "addresses: an admin adds an address, the service provisions a matching Matrix account "
        f"on the homeserver ({MATRIX_SERVER_NAME}), and that address can then sign in and talk "
        "in the subnet's rooms. There is no self-service signup. Clients authenticate by signing "
        f"the fixed message \"{SIGN_MESSAGE}\" with their Ethereum private key and sending "
        "{address, signature} in the request body. The public directory is at "
        "<code>/api/users</code>, and the subnet's name and icon at "
        "<code>/api/metadata</code>.</p>"
    )


def admin_html() -> str:
    """Local-only branding form. Posts name + optional icon to /api/set_branding
    via fetch, so a human can rebrand the subnet without crafting curl."""
    current = subnet_name().replace('"', "&quot;")
    return (
        "<!doctype html><meta charset=\"utf-8\">"
        f"<title>{SUBNET_DOMAIN} · branding</title>"
        "<body style=\"font-family:system-ui;max-width:520px;margin:40px auto;padding:0 16px\">"
        "<h2>Subnet branding</h2>"
        f"<p>Set the name and icon for <code>{SUBNET_DOMAIN}</code>. Leave a field blank to keep it.</p>"
        "<form id=\"f\">"
        f"<p><label>Name<br><input name=\"name\" value=\"{current}\" style=\"width:100%;padding:8px\"></label></p>"
        "<p><label>Icon (png/jpg)<br><input type=\"file\" name=\"file\" accept=\"image/png,image/jpeg\"></label></p>"
        "<button type=\"submit\" style=\"padding:8px 16px\">Save</button> <span id=\"s\"></span>"
        "</form>"
        "<p>Current icon:</p>"
        "<img src=\"/api/icon\" alt=\"(none yet)\" style=\"max-width:96px;border:1px solid #ccc\" "
        "onerror=\"this.replaceWith('(none yet)')\">"
        "<script>"
        "const f=document.getElementById('f'),s=document.getElementById('s');"
        "f.onsubmit=async(e)=>{e.preventDefault();s.textContent=' saving…';"
        "const r=await fetch('/api/set_branding',{method:'POST',body:new FormData(f)});"
        "s.textContent=r.ok?' saved ✓':' error: '+await r.text();"
        "if(r.ok)setTimeout(()=>location.reload(),700);};"
        "</script></body>"
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    return landing_html()


# ────────────────────────── public / user endpoints ──────────────────────────
@app.get("/ping")
async def ping():
    return {"ok": True, "sign_message": SIGN_MESSAGE}


@app.get("/api/metadata")
async def metadata():
    """Public subnet metadata: its name, domain, and icon url (empty when no icon
    has been uploaded). The bottles frontend reads this to brand the left rail."""
    return {"name": subnet_name(), "domain": SUBNET_DOMAIN, "icon_url": icon_url()}


@app.get("/api/icon")
async def icon():
    p = icon_path()
    if not p:
        raise HTTPException(404, "no icon set")
    return FileResponse(p, media_type="image/png" if p.endswith(".png") else "image/jpeg")


@app.post("/api/set_branding")
async def set_branding(
    request: Request,
    name: str = Form(default=""),
    file: UploadFile | None = File(default=None),
):
    """Set the subnet's name and/or icon. Local-only — the operator runs it on the
    box, by hand (curl) or via the /admin page. Both fields are optional, so the
    name and icon can be updated independently."""
    require_local(request)
    if name.strip():
        os.makedirs(ICON_DIR, exist_ok=True)
        with open(os.path.join(ICON_DIR, "name.txt"), "w") as f:
            f.write(name.strip()[:200])
    if file is not None and file.filename:
        data = await file.read()
        if data:
            save_icon(data, file.filename, file.content_type or "")
    return {"name": subnet_name(), "domain": SUBNET_DOMAIN, "icon_url": icon_url()}


@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request):
    """A tiny local-only branding form so the operator can set the subnet's name
    and icon from a browser instead of crafting a multipart request by hand."""
    require_local(request)
    return admin_html()


@app.get("/api/users")
async def users():
    """Public subnet directory. The `subnet-client` SDK's `listSubnetUsers()`
    GETs this unauthenticated and keys off each row's `address` (it derives the
    Matrix user id from it). We return every member, newest registrations last,
    Steward included."""
    return {"users": [user_row(u) for u in User.select().order_by(User.created_at)]}


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
    if "avatar_url" in body:
        avatar_url = clean_avatar_link(body.get("avatar_url"))
        if avatar_url and user.matrix_synced and user.matrix_access_token:
            try:
                matrix_set_avatar(
                    user.matrix_access_token,
                    f"@{user.address}:{MATRIX_SERVER_NAME}",
                    avatar_url,
                )
            except Exception as e:
                raise HTTPException(502, f"matrix avatar update failed: {e}")
        user.avatar_url = avatar_url
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
    Returns the new user plus their Matrix credentials.

    The admin signs the request as themselves (`address` + `signature` in the
    body are the admin's, validated by `authed`); the new user's ETH address
    goes in `target_address`."""
    requester, body = await authed(request)
    require_admin(requester)

    target_address = (body.get("target_address") or "").strip().lower()
    if not (target_address.startswith("0x") and len(target_address) == 42):
        raise HTTPException(400, "target_address must be a 0x-prefixed ETH address")
    if User.get_or_none(User.address == target_address) is not None:
        raise HTTPException(409, "address already registered")

    role = body.get("role", "user")
    if role not in ("user", "admin"):
        raise HTTPException(400, "role must be 'user' or 'admin'")
    name = (body.get("name") or "").strip()[:120] or target_address
    avatar_url = clean_avatar_link(body.get("avatar_url"))

    user = User.create(
        address=target_address,
        matrix_password=secrets.token_urlsafe(24),
        name=name,
        avatar_url=avatar_url,
        role=role,
    )
    # provision_matrix_user shells out to the `subnet` CLI, which calls back
    # into this server's /api/credentials over HTTP. If we run it inline we
    # block the event loop and the CLI's callback can't be served — a self
    # deadlock that surfaces as `subprocess timed out`. Run it on a thread.
    try:
        await asyncio.to_thread(provision_matrix_user, user)
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
