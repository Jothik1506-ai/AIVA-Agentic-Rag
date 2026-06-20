"""
app/api/legacy.py  ─  Legacy Route Adapter
==========================================
Maps all your existing Flask API paths to FastAPI
so your existing frontend (chatbot_new.html, kb_popup.html)
continues to work without any changes.

Original routes preserved:
  POST /api/auth/login        → now handled by auth.py (same path)
  POST /api/auth/logout
  POST /api/auth/register
  POST /api/chat              (your original endpoint)
  GET  /api/chat/history
  POST /api/documents/upload  (your original upload path)
  GET  /api/documents
  DELETE /api/documents/{doc_id}
  GET  /api/health
  GET  /api/system/info
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.database import UserRecord

router = APIRouter(tags=["legacy"])


# ── /api/health (your existing endpoint) ─────────────────────────
@router.get("/api/health")
async def health_check():
    from app_state import system_initialized, initialization_error, doc_manager, llm
    docs_loaded = len(doc_manager.loaded_documents) if doc_manager else 0
    return {
        "status":      "ok" if system_initialized else "degraded",
        "initialized": system_initialized,
        "error":       initialization_error,
        "docs_loaded": docs_loaded,
        "llm_active":  llm is not None,
        "version":     "4.0.0",
    }


# ── /api/system/info (your existing endpoint) ────────────────────
@router.get("/api/system/info")
async def system_info():
    from app_state import system_initialized, doc_manager, llm, embedding_model
    from config import config
    cap = config.get_capability_status()
    return {
        "version":     "4.0.0",
        "initialized": system_initialized,
        "device":      config.DEVICE_NAME,
        "llm_model":   config.LLM_MODEL,
        "llm_active":  llm is not None,
        "docs_loaded": len(doc_manager.loaded_documents) if doc_manager else 0,
        "embed_model": config.EMBED_MODEL,
        "capabilities": cap,
        "supported_formats": config.SUPPORTED_DOCUMENT_FORMATS,
    }


# ── /api/documents (list) — your existing endpoint ───────────────
@router.get("/api/documents")
async def list_documents(
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user),
):
    from app.auth.rbac import get_allowed_doc_ids
    from app.core.database import DocumentRecord
    doc_ids = get_allowed_doc_ids(current_user, db)
    docs = db.query(DocumentRecord).filter(
        DocumentRecord.doc_id.in_(doc_ids)
    ).order_by(DocumentRecord.created_at.desc()).all()
    return [
        {
            "doc_id":    d.doc_id,
            "filename":  d.filename,
            "domain":    d.domain,
            "pages":     d.page_count,
            "uploaded":  str(d.created_at),
        }
        for d in docs
    ]


# ── DELETE /api/documents/{doc_id} — your existing endpoint ──────
@router.delete("/api/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user),
):
    from app.core.database import DocumentRecord, DocumentAccess, ChunkRecord
    # Verify access
    doc = db.query(DocumentRecord).filter_by(doc_id=doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    if doc.uploader_id != current_user.user_id:
        raise HTTPException(403, "Only the uploader can delete this document")

    # Delete chunks, access records, document
    db.query(ChunkRecord).filter_by(doc_id=doc_id).delete()
    db.query(DocumentAccess).filter_by(doc_id=doc_id).delete()
    db.delete(doc)
    db.commit()

    # Remove from loaded documents in your existing doc_manager
    try:
        from app_state import doc_manager
        if doc_manager and doc_id in doc_manager.loaded_documents:
            del doc_manager.loaded_documents[doc_id]
    except Exception:
        pass

    return {"message": f"Document {doc_id} deleted", "doc_id": doc_id}


# ── POST /api/documents/upload — your legacy upload path ─────────
# (The new path is /api/upload/ — this alias keeps old frontend working)
@router.post("/api/documents/upload")
async def legacy_upload(
    request: Request,
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user),
):
    """Alias for /api/upload/ — keeps your existing frontend working."""
    from fastapi import UploadFile, File, Form
    form = await request.form()
    file = form.get("file")
    access_key = form.get("access_key", current_user.access_key)

    if not file:
        raise HTTPException(400, "No file provided")

    file_bytes = await file.read()
    from app.ingestion.pipeline import ingest_document
    return await ingest_document(
        file_bytes=file_bytes,
        filename=file.filename,
        uploader_id=current_user.user_id,
        uploader_role=current_user.role,
        access_key=access_key or "",
        db=db,
    )


# ── POST /api/auth/logout ─────────────────────────────────────────
@router.post("/api/auth/logout")
async def logout(current_user: UserRecord = Depends(get_current_user)):
    """Your existing logout — JWT is stateless, client clears token."""
    return {"message": "Logged out successfully"}


# ── POST /api/auth/register ───────────────────────────────────────
# (Full implementation is in auth.py — this is the alias)
class RegisterLegacy(BaseModel):
    username:   str
    password:   str
    role:       str
    access_key: str

@router.post("/api/auth/register")
async def legacy_register(req: RegisterLegacy, db: Session = Depends(get_db)):
    from app.api.auth import register as _register
    from app.api.auth import RegisterRequest
    return _register(RegisterRequest(**req.dict()), db)
