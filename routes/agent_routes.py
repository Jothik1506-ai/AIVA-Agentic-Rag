"""
routes/agent_routes.py

Admin-only REST API for managing AIVA agents.

Endpoints:
  GET    /api/agents                    — list all agents + index status
  POST   /api/agents                    — create agent
  GET    /api/agents/<id>               — get single agent
  PUT    /api/agents/<id>               — update agent
  DELETE /api/agents/<id>               — delete agent
  POST   /api/agents/<id>/enable        — enable agent
  POST   /api/agents/<id>/disable       — disable agent
  POST   /api/agents/<id>/index         — (re)index agent now (sync)
  GET    /api/agents/<id>/status        — index + source status
  POST   /api/agents/debug/route        — show routing scores for a query
  POST   /api/agents/<id>/upload-source — upload a file as a new source
"""

import os
import logging
from pathlib import Path
from flask import Blueprint, jsonify, request

from modules.auth import require_login, get_current_user_id
from agents.agent_registry import (
    list_agents, get_agent, get_agent_for_owner, create_agent, update_agent,
    delete_agent, set_enabled
)
from agents.agent_indexer import agent_is_indexed, read_index_status
from app_state import get_agent_manager

logger = logging.getLogger(__name__)

agent_bp = Blueprint("agents", __name__, url_prefix="/api/agents")

UPLOADS_DIR = Path(__file__).parent.parent / "agent_uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _mgr():
    mgr = get_agent_manager()
    if mgr is None:
        return None, jsonify({"error": "Agent manager not ready"}), 503
    return mgr, None, None


def _owned_agent(agent_id):
    """Fetch an agent only if it belongs to the requesting user.

    Returns None for both nonexistent and foreign agents so responses don't
    reveal whether another user's notebook id exists.
    """
    return get_agent_for_owner(agent_id, get_current_user_id())


# ── List / Create ─────────────────────────────────────────────────────────────

@agent_bp.route("", methods=["GET"])
@require_login
def list_all_agents():
    agents = list_agents(include_disabled=True, owner_id=get_current_user_id())
    result = []
    for a in agents:
        idx = read_index_status(a["id"])
        result.append({
            **a,
            "is_indexed":   agent_is_indexed(a["id"]),
            "index_status": idx["status"],
            "index_error":  idx.get("error", ""),
        })
    return jsonify({"agents": result, "total": len(result)})


@agent_bp.route("", methods=["POST"])
@require_login
def create_new_agent():
    data = request.get_json(silent=True) or {}

    name        = data.get("name", "").strip()
    description = data.get("description", "").strip()
    keywords    = data.get("keywords", [])
    sources     = data.get("sources", [])          # [{type, value}]
    refresh_h   = int(data.get("refresh_hours", 24))
    enabled     = bool(data.get("enabled", True))

    if not name:
        return jsonify({"error": "Agent name is required"}), 400
    # Auto-generate keywords from name if none provided
    if not keywords:
        keywords = [w for w in name.lower().split() if len(w) > 2] or [name.lower()]

    agent = create_agent(
        name=name,
        description=description,
        keywords=keywords,
        sources=sources,
        refresh_hours=refresh_h,
        enabled=enabled,
        owner_id=get_current_user_id(),
    )

    # Kick off async indexing if sources provided
    if sources:
        mgr = get_agent_manager()
        if mgr:
            mgr.ensure_indexed(agent["id"], force=True)

    return jsonify({"agent": agent, "message": "Agent created. Indexing started in background."}), 201


# ── Single agent ──────────────────────────────────────────────────────────────

@agent_bp.route("/<agent_id>", methods=["GET"])
@require_login
def get_one_agent(agent_id):
    agent = _owned_agent(agent_id)
    if not agent:
        return jsonify({"error": "Agent not found"}), 404
    return jsonify({"agent": {**agent, "is_indexed": agent_is_indexed(agent_id)}})


@agent_bp.route("/<agent_id>", methods=["PUT"])
@require_login
def update_one_agent(agent_id):
    if not _owned_agent(agent_id):
        return jsonify({"error": "Agent not found"}), 404
    data = request.get_json(silent=True) or {}
    updated = update_agent(agent_id, data)
    if not updated:
        return jsonify({"error": "Agent not found"}), 404

    # If sources changed, re-index
    if "sources" in data:
        mgr = get_agent_manager()
        if mgr:
            mgr.ensure_indexed(agent_id, force=True)

    return jsonify({"agent": updated, "message": "Agent updated."})


@agent_bp.route("/<agent_id>", methods=["DELETE"])
@require_login
def delete_one_agent(agent_id):
    if not _owned_agent(agent_id):
        return jsonify({"error": "Agent not found"}), 404
    ok = delete_agent(agent_id)
    if not ok:
        return jsonify({"error": "Agent not found"}), 404
    return jsonify({"message": f"Agent '{agent_id}' deleted."})


# ── Enable / Disable ─────────────────────────────────────────────────────────

@agent_bp.route("/<agent_id>/enable", methods=["POST"])
@require_login
def enable_agent(agent_id):
    if not _owned_agent(agent_id):
        return jsonify({"error": "Agent not found"}), 404
    updated = set_enabled(agent_id, True)
    if not updated:
        return jsonify({"error": "Agent not found"}), 404
    return jsonify({"message": f"Agent '{agent_id}' enabled.", "agent": updated})


@agent_bp.route("/<agent_id>/disable", methods=["POST"])
@require_login
def disable_agent(agent_id):
    if not _owned_agent(agent_id):
        return jsonify({"error": "Agent not found"}), 404
    updated = set_enabled(agent_id, False)
    if not updated:
        return jsonify({"error": "Agent not found"}), 404
    return jsonify({"message": f"Agent '{agent_id}' disabled.", "agent": updated})


# ── Indexing ──────────────────────────────────────────────────────────────────

@agent_bp.route("/<agent_id>/index", methods=["POST"])
@require_login
def reindex_agent(agent_id):
    agent = _owned_agent(agent_id)
    if not agent:
        return jsonify({"error": "Agent not found"}), 404

    mgr = get_agent_manager()
    if not mgr:
        return jsonify({"error": "Agent manager not ready"}), 503

    ok = mgr.index_agent_sync(agent_id)
    if ok:
        return jsonify({"message": f"Agent '{agent_id}' indexed successfully."})
    else:
        return jsonify({"error": "Indexing failed. Check server logs."}), 500


@agent_bp.route("/<agent_id>/status", methods=["GET"])
@require_login
def agent_status(agent_id):
    agent = _owned_agent(agent_id)
    if not agent:
        return jsonify({"error": "Agent not found"}), 404
    idx_status = read_index_status(agent_id)
    return jsonify({
        "agent_id":     agent_id,
        "name":         agent["name"],
        "enabled":      agent["enabled"],
        "is_indexed":   agent_is_indexed(agent_id),
        "index_status": idx_status["status"],   # pending|indexing|indexed|failed
        "index_error":  idx_status.get("error", ""),
        "last_indexed": agent.get("last_indexed"),
        "sources":      agent.get("sources", []),
    })


@agent_bp.route("/<agent_id>/index-status", methods=["GET"])
@require_login
def index_status_fast(agent_id):
    """Lightweight endpoint polled by the UI to check indexing progress."""
    agent = _owned_agent(agent_id)
    if not agent:
        return jsonify({"error": "Agent not found"}), 404
    idx = read_index_status(agent_id)
    return jsonify({
        "agent_id":     agent_id,
        "index_status": idx["status"],
        "index_error":  idx.get("error", ""),
        "is_indexed":   agent_is_indexed(agent_id),
    })


# ── File upload as source ─────────────────────────────────────────────────────

@agent_bp.route("/<agent_id>/upload-source", methods=["POST"])
@require_login
def upload_source_file(agent_id):
    """
    Upload a file (PDF, TXT, CSV, MD) and add it as a source for this agent.
    The file is saved to agent_uploads/<agent_id>/ and the source entry is added.
    """
    agent = _owned_agent(agent_id)
    if not agent:
        return jsonify({"error": "Agent not found"}), 404

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    from werkzeug.utils import secure_filename
    from pathlib import Path as _Path

    _ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.csv', '.md', '.docx', '.xlsx', '.pptx'}
    _MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
    # Magic-byte prefixes for allowed binary types
    _MAGIC = {
        b'%PDF': '.pdf',
        b'PK\x03\x04': None,  # ZIP-based: docx/xlsx/pptx — validated by extension
    }

    filename = secure_filename(f.filename)
    if not filename:
        return jsonify({"error": "Invalid filename"}), 400
    ext = _Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        return jsonify({"error": f"File type not allowed. Permitted: {', '.join(_ALLOWED_EXTENSIONS)}"}), 400

    # Read once to enforce size limit before writing to disk
    data = f.read(_MAX_UPLOAD_BYTES + 1)
    if len(data) > _MAX_UPLOAD_BYTES:
        return jsonify({"error": "File exceeds 50 MB limit"}), 413

    # Magic-byte check for binary types
    if ext == '.pdf' and not data.startswith(b'%PDF'):
        return jsonify({"error": "File content does not match declared type"}), 400
    if ext in {'.docx', '.xlsx', '.pptx'} and not data.startswith(b'PK\x03\x04'):
        return jsonify({"error": "File content does not match declared type"}), 400

    save_dir = UPLOADS_DIR / agent_id
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    save_path.write_bytes(data)

    # Append source to agent
    sources = agent.get("sources", [])
    sources.append({"type": "file", "value": str(save_path)})
    update_agent(agent_id, {"sources": sources})

    # Trigger re-index
    mgr = get_agent_manager()
    if mgr:
        mgr.ensure_indexed(agent_id, force=True)

    return jsonify({
        "message": f"File '{filename}' uploaded and indexing started.",
        "file_path": str(save_path),
    })


# ── Source file read / save ───────────────────────────────────────────────────

@agent_bp.route("/<agent_id>/sources/<int:src_idx>/read", methods=["GET"])
@require_login
def read_source_file(agent_id, src_idx):
    """Read the content of an agent source file for in-place editing."""
    agent = _owned_agent(agent_id)
    if not agent:
        return jsonify({"error": "Agent not found"}), 404
    sources = agent.get("sources", [])
    if src_idx < 0 or src_idx >= len(sources):
        return jsonify({"error": "Source index out of range"}), 400
    src = sources[src_idx]
    if src.get("type") != "file":
        return jsonify({"error": "Source is not a file"}), 400

    from pathlib import Path as _Path
    import agents.file_editing_agent as fa

    abs_path = _Path(src["value"]).resolve()
    allowed  = fa._get_allowed_root()
    try:
        abs_path.relative_to(allowed)
    except ValueError:
        return jsonify({"error": "Path outside allowed root"}), 403

    rel_path = str(abs_path.relative_to(allowed))
    result   = fa.read_file_content(rel_path)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@agent_bp.route("/<agent_id>/sources/<int:src_idx>/save", methods=["POST"])
@require_login
def save_source_file(agent_id, src_idx):
    """Save edited content back to an agent source file and re-index."""
    agent = _owned_agent(agent_id)
    if not agent:
        return jsonify({"error": "Agent not found"}), 404
    sources = agent.get("sources", [])
    if src_idx < 0 or src_idx >= len(sources):
        return jsonify({"error": "Source index out of range"}), 400
    src = sources[src_idx]
    if src.get("type") != "file":
        return jsonify({"error": "Source is not a file"}), 400

    from pathlib import Path as _Path
    import agents.file_editing_agent as fa

    abs_path = _Path(src["value"]).resolve()
    allowed  = fa._get_allowed_root()
    try:
        abs_path.relative_to(allowed)
    except ValueError:
        return jsonify({"error": "Path outside allowed root"}), 403

    rel_path = str(abs_path.relative_to(allowed))
    data     = request.get_json() or {}
    suffix   = abs_path.suffix.lower()

    if suffix == ".docx":
        result = fa.save_docx_file(rel_path, data.get("paragraphs", []), confirmed=True)
    elif suffix in {".txt", ".md"}:
        result = fa.save_text_file(rel_path, data.get("content", ""), confirmed=True)
    elif suffix == ".csv":
        result = fa.save_csv_file(rel_path, data.get("rows", []), confirmed=True)
    else:
        return jsonify({"error": f"Editing not supported for {suffix}"}), 400

    if not result.get("success"):
        return jsonify(result), 400

    # Re-index this agent after saving
    mgr = get_agent_manager()
    if mgr:
        mgr.ensure_indexed(agent_id, force=True)

    return jsonify({"success": True, "message": "Saved and re-indexing started."})


# ── Debug routing ─────────────────────────────────────────────────────────────

@agent_bp.route("/debug/route", methods=["POST"])
@require_login
def debug_routing():
    """Show which agents would be selected for a given query and why."""
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "query field required"}), 400

    mgr = get_agent_manager()
    if not mgr:
        return jsonify({"error": "Agent manager not ready"}), 503

    explanation = mgr.get_routing_debug(query, owner_id=get_current_user_id())
    return jsonify({"query": query, "routing": explanation})
