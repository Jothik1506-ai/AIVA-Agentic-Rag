"""
Application state management.
This module holds the global application state to avoid circular imports.
"""
import os
import logging
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from modules.document_manager import AdvancedDocumentManager
from modules.conversation import EnhancedConversationContext
from config import config as app_config

logger = logging.getLogger(__name__)

# Initialize system state
system_initialized = False
initialization_error = None
embedding_model = None
llm = None          # primary: LM Studio
_groq_llm = None   # fallback: Groq cloud
doc_manager = None
context_manager = None
streaming_callback = None
agent_manager = None          # ← Stage 2: AIVA Agent Manager

# ── LM Studio health check ────────────────────────────────────────────────────

def _lm_studio_healthy() -> bool:
    """Ping LM Studio with a 2-second timeout. Returns True if reachable."""
    try:
        import urllib.request
        url = f"{app_config.LM_STUDIO_BASE_URL.rstrip('/')}/v1/models"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=2):
            return True
    except Exception:
        return False


def get_active_llm():
    """
    Return the appropriate LLM client:
      - LM Studio if it is reachable (PC on, LM Studio running)
      - Groq cloud fallback otherwise
    Falls back to the last known working client if both checks fail.
    """
    global llm, _groq_llm

    if _lm_studio_healthy():
        if llm is not None:
            return llm, "lm_studio"
        # LM Studio is up but llm wasn't initialised — build it now
        try:
            llm = ChatOpenAI(
                model=app_config.LLM_MODEL,
                temperature=0.1,
                base_url=f"{app_config.LM_STUDIO_BASE_URL.rstrip('/')}/v1",
                api_key=os.getenv("LLM_API_KEY", "lm-studio"),
                max_tokens=app_config.LM_STUDIO_MAX_TOKENS,
            )
            return llm, "lm_studio"
        except Exception as e:
            logger.warning("[LLM] LM Studio re-init failed: %s", e)

    # LM Studio unreachable — use Groq
    if _groq_llm is not None:
        return _groq_llm, "groq"

    if app_config.GROQ_API_KEY and app_config.GROQ_API_KEY != "your_groq_api_key_here":
        try:
            _groq_llm = ChatOpenAI(
                model=app_config.GROQ_MODEL,
                temperature=0.1,
                base_url=app_config.GROQ_BASE_URL,
                api_key=app_config.GROQ_API_KEY,
                max_tokens=app_config.LM_STUDIO_MAX_TOKENS,
            )
            logger.info("[LLM] Switched to Groq fallback (%s)", app_config.GROQ_MODEL)
            return _groq_llm, "groq"
        except Exception as e:
            logger.warning("[LLM] Groq init failed: %s", e)

    # Last resort: return whatever we have (may be None)
    return llm, "lm_studio"


def get_system_state():
    """Get the current system state"""
    return system_initialized, initialization_error, doc_manager, context_manager, app_config, embedding_model, llm

def get_agent_manager():
    """Return the global AgentManager instance (or None if not ready)."""
    return agent_manager

def initialize_system():
    """Initialize the RAG system on startup"""
    global system_initialized, initialization_error, doc_manager, context_manager, embedding_model, llm, streaming_callback, agent_manager
    
    try:
        print("Initializing RAG system...")
        
        # Initialize embedding model
        # Cloud (Render): uses HuggingFace Inference API — no torch/local model needed
        # Local (PC):     uses HuggingFaceEmbeddings with local sentence-transformers
        hf_api_key = getattr(app_config, "HF_API_KEY", "") or os.getenv("HF_API_KEY", "")
        if hf_api_key:
            print("🌐 Using HuggingFace Inference API for embeddings (cloud mode)...")
            try:
                from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
                embed_model_name = app_config.EMBED_MODEL_OPTIONS[0]
                embedding_model = HuggingFaceInferenceAPIEmbeddings(
                    api_key=hf_api_key,
                    model_name=embed_model_name,
                )
                print(f"✅ Embedding model (API): {embed_model_name}")
            except Exception as e:
                raise Exception(f"HuggingFace Inference API embedding failed: {e}")
        else:
            print(f"💻 Loading local embedding model on {app_config.DEVICE}...")
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                embedding_model = HuggingFaceEmbeddings(
                    model_name=app_config.EMBED_MODEL_OPTIONS[0],
                    model_kwargs={'device': app_config.DEVICE},
                    encode_kwargs={'normalize_embeddings': True}
                )
                print(f"✅ Embedding model loaded: {app_config.EMBED_MODEL_OPTIONS[0]} on {app_config.DEVICE}")
            except Exception as e1:
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                    embedding_model = HuggingFaceEmbeddings(
                        model_name=app_config.EMBED_MODEL,
                        model_kwargs={'device': app_config.DEVICE},
                        encode_kwargs={'normalize_embeddings': True}
                    )
                    print(f"✅ Fallback embedding model: {app_config.EMBED_MODEL}")
                except Exception as e2:
                    raise Exception(f"Failed to load any embedding model: {e2}")
        
        # Initialize LLM via LM Studio's OpenAI-compatible endpoint
        print("Initializing language model...")
        try:
            lm_studio_base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
            llm_base_url = f"{lm_studio_base_url.rstrip('/')}/v1"

            llm = ChatOpenAI(
                model=app_config.LLM_MODEL,
                temperature=0.1,
                base_url=llm_base_url,
                api_key=os.getenv("LLM_API_KEY", "lm-studio"),
                max_tokens=getattr(app_config, "LM_STUDIO_MAX_TOKENS", 2048),
            )

            print(f"✅ Language model initialized: {app_config.LLM_MODEL} @ {llm_base_url}")

        except Exception as e:
            print(f"⚠️  Warning: LLM initialization failed: {e}")
            print("   The system will continue with document search only (no LLM responses)")
            llm = None  # Continue without LLM
        
        # Initialize managers
        os.makedirs(app_config.EMBEDDINGS_DIR, exist_ok=True)
        doc_manager = AdvancedDocumentManager(app_config.EMBEDDINGS_DIR, embedding_model=embedding_model)
        context_manager = EnhancedConversationContext()
        
        # Auto-load documents if available (with timeout to prevent blocking)
        print("Loading documents...")
        from threading import Thread
        import time
        
        def load_docs_with_timeout(timeout_seconds=60):
            """Load documents with a timeout to prevent blocking"""
            try:
                doc_manager.load_all_documents()
            except Exception as e:
                print(f"⚠️  Warning: Document loading failed: {e}")
        
        # Try to load documents with a timeout
        doc_thread = Thread(target=load_docs_with_timeout, daemon=True)
        doc_thread.start()
        doc_thread.join(timeout=60)  # Wait max 60 seconds for document loading
        
        if doc_thread.is_alive():
            print("⚠️  Warning: Document loading timed out (60s), continuing without full documents")
        
        system_initialized = True
        print("✅ System initialization complete")

        # ── Stage 2: Start AIVA Agent Manager & scheduler ──────────────────
        try:
            from agents.agent_manager import AgentManager
            from agents.scheduler import start_scheduler
            agent_manager = AgentManager(embedding_model)
            start_scheduler(agent_manager)
            print("✅ Agent Manager started (Stage 2 active)")
        except Exception as e:
            print(f"⚠️  Agent Manager failed to start: {e}")
            agent_manager = None
        
    except Exception as e:
        system_initialized = False
        initialization_error = str(e)
        print(f"❌ Error initializing system: {e}")
        raise