"""
Application state management.
This module holds the global application state to avoid circular imports.
"""
import os
import time
import logging
import functools
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings as _LCEmbeddings
from langchain_openai import ChatOpenAI
from modules.document_manager import AdvancedDocumentManager
from modules.conversation import EnhancedConversationContext
from config import config as app_config

# Imported at module level (not inside the init thread) — importing heavy
# packages from a daemon thread during worker startup can wedge on the
# import lock and leave the system stuck in "initializing" forever.
try:
    from huggingface_hub import InferenceClient as _IClient
except ImportError:
    _IClient = None
import numpy as _np

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
    # On Render (cloud) LM Studio can never be reachable at 127.0.0.1 —
    # skip the probe entirely and go straight to the Groq fallback.
    if os.environ.get('RENDER'):
        return False
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
        init_t0 = time.time()

        def _step_log(msg: str):
            print(f"[init +{time.time() - init_t0:.1f}s] {msg}", flush=True)

        _step_log("Initializing RAG system...")

        # Initialize embedding model
        # Cloud (Render): uses HuggingFace Inference API — no torch/local model needed
        # Local (PC):     uses HuggingFaceEmbeddings with local sentence-transformers
        hf_api_key = getattr(app_config, "HF_API_KEY", "") or os.getenv("HF_API_KEY", "")
        if hf_api_key:
            _step_log("Step 1/4: HuggingFace Inference API embeddings (cloud mode)...")
            if _IClient is None:
                raise Exception("huggingface_hub is not installed — cannot use Inference API embeddings")
            try:
                # Must subclass langchain's Embeddings ABC: FAISS checks
                # isinstance(self.embeddings, Embeddings) at query time and
                # otherwise treats the object as a legacy callable — plain
                # duck-typing crashes every similarity_search with
                # "'_HFInferenceEmbeddings' object is not callable".
                # Client is created lazily on first use to avoid blocking during startup.
                class _HFInferenceEmbeddings(_LCEmbeddings):
                    def __init__(self, api_key: str, model: str):
                        self._api_key = api_key
                        self._model = model
                        self._client = None  # created on first call
                        # Each embed is a remote HTTP call; the same query is embedded
                        # once per document per agentic iteration, so cache by text.
                        self._encode = functools.lru_cache(maxsize=512)(self._encode_uncached)

                    def _get_client(self):
                        if self._client is None:
                            self._client = _IClient(token=self._api_key)
                        return self._client

                    def _encode_uncached(self, text: str) -> tuple:
                        result = self._get_client().feature_extraction(
                            text, model=self._model
                        )
                        arr = _np.array(result)
                        if arr.ndim == 2:
                            arr = arr.mean(axis=0)
                        return tuple(arr.tolist())

                    def embed_documents(self, texts):
                        return [list(self._encode(t)) for t in texts]

                    def embed_query(self, text: str):
                        return list(self._encode(text))

                # HF's serverless inference (hf-inference provider) no longer
                # serves sentence-transformers/all-MiniLM-L6-v2 — requests fail
                # with "Model not supported by provider hf-inference".
                # BAAI/bge-small-en-v1.5 is served, and is a drop-in 384-dim
                # replacement. Note: indexes are only searchable with the same
                # model that built them, so cloud indexes must be built in the
                # cloud (uploads re-index there anyway).
                embed_model_name = os.getenv("CLOUD_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
                embedding_model = _HFInferenceEmbeddings(
                    api_key=hf_api_key,
                    model=embed_model_name,
                )
                _step_log(f"✅ Embedding model (API): {embed_model_name}")
            except Exception as e:
                raise Exception(f"HuggingFace Inference API embedding failed: {e}")
        else:
            _step_log(f"Step 1/4: loading local embedding model on {app_config.DEVICE}...")
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
        _step_log("Step 2/4: Initializing language model...")
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

            _step_log(f"✅ LLM client created: {app_config.LLM_MODEL}")

        except Exception as e:
            _step_log(f"⚠️  LLM init skipped (will use Groq fallback): {e}")
            llm = None

        # Initialize managers
        _step_log("Step 3/4: Creating document manager...")
        os.makedirs(app_config.EMBEDDINGS_DIR, exist_ok=True)
        doc_manager = AdvancedDocumentManager(app_config.EMBEDDINGS_DIR, embedding_model=embedding_model)
        context_manager = EnhancedConversationContext()

        # Load documents — short timeout so init doesn't hang on empty dir
        _step_log("Step 4/4: Loading documents...")
        from threading import Thread

        def _load_docs():
            try:
                doc_manager.load_all_documents()
            except Exception as e:
                print(f"⚠️  Document loading failed: {e}")

        doc_thread = Thread(target=_load_docs, daemon=True)
        doc_thread.start()
        doc_thread.join(timeout=10)  # 10 s max — empty dir returns instantly

        if doc_thread.is_alive():
            _step_log("⚠️  Document loading taking >10s, continuing in background")

        system_initialized = True
        _step_log("✅ System initialization complete")

        # ── Stage 2: Start AIVA Agent Manager & scheduler ──────────────────
        try:
            from agents.agent_manager import AgentManager
            from agents.scheduler import start_scheduler
            agent_manager = AgentManager(embedding_model)
            start_scheduler(agent_manager)
            print("✅ Agent Manager started (Stage 2 active)", flush=True)
        except Exception as e:
            print(f"⚠️  Agent Manager failed to start: {e}", flush=True)
            agent_manager = None

    except Exception as e:
        system_initialized = False
        initialization_error = str(e)
        print(f"❌ Error initializing system: {e}", flush=True)
        raise