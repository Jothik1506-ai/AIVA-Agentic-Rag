"""
app.py  ─  V4 GraphRAG FastAPI Application Entry Point
=======================================================
Run with:  python app.py
"""

import os
import uuid
import warnings
warnings.filterwarnings("ignore")

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.api.auth     import router as auth_router
from app.api.upload   import router as upload_router
from app.api.chat     import router as chat_router
from app.api.system   import router as system_router
from app.api.legacy   import router as legacy_router

BASE_DIR      = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR    = BASE_DIR / "static"


def _seed_demo_user():
    """Create a demo account when the database is empty."""
    try:
        from app.core.database import SessionLocal, UserRecord
        from app.core.security import hash_password
        from app.core.config import get_settings
        cfg = get_settings()
        db = SessionLocal()
        try:
            if db.query(UserRecord).first():
                return
            db.add(UserRecord(
                user_id=str(uuid.uuid4()),
                username="demo",
                hashed_pw=hash_password("demo123"),
                role="agent",
                access_key="12345",
                llm_choice=cfg.llm_models["default"],
            ))
            db.commit()
            print("  Demo user created: demo / demo123 (access key 12345)")
        finally:
            db.close()
    except Exception as e:
        print(f"  Demo user seed skipped: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown."""
    print("\nStarting V4 GraphRAG System...")
    try:
        from app_state import initialize_system
        initialize_system()
    except Exception as e:
        print(f"System init warning: {e}")
        print("   App will start in degraded mode")
        try:
            from app.core.database import init_db
            init_db()
        except Exception:
            pass
    _seed_demo_user()
    yield
    try:
        from app.core.database import Neo4jDB
        Neo4jDB.close()
    except Exception:
        pass
    print("V4 GraphRAG shut down cleanly")


app = FastAPI(
    title="V4 GraphRAG API",
    description="Multi-Agent GraphRAG with RBAC, Knowledge Graph, and MoE embeddings",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(upload_router)
app.include_router(chat_router)
app.include_router(system_router)
app.include_router(legacy_router)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _serve_template(name: str):
    path = TEMPLATES_DIR / name
    if path.exists():
        return FileResponse(str(path))
    return {"error": f"{name} not found in templates/"}


@app.get("/")
async def serve_login():
    return _serve_template("login.html")


@app.get("/login.html")
async def serve_login_alias():
    return _serve_template("login.html")


@app.get("/chatbot")
async def serve_chatbot():
    return _serve_template("chatbot.html")


@app.get("/chatbot.html")
async def serve_chatbot_alias():
    return _serve_template("chatbot.html")


@app.get("/health")
async def health():
    from app_state import system_initialized, initialization_error
    return {
        "status":  "ok" if system_initialized else "degraded",
        "version": "4.0.0",
        "error":   initialization_error,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"\nStarting server on http://localhost:{port}")
    print(f"API docs: http://localhost:{port}/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
