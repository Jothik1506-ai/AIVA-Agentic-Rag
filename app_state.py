"""
app_state.py  ─  V4 GraphRAG Application State
================================================
Upgraded from your existing app_state.py.
Original initialization logic is preserved and extended.
V4 additions: MoE embedder, Neo4j, FAISS multi-index,
RBAC DB, snapshot fabric, NLI verifier.
"""

import os
from pathlib import Path
from config import config as app_config

# ── State flags (your existing pattern) ──────────────────────────
system_initialized   = False
initialization_error = None

# ── Your existing components ──────────────────────────────────────
embedding_model  = None    # primary HuggingFace model (your existing)
llm              = None    # Ollama LLM (your existing)
doc_manager      = None    # AdvancedDocumentManager (your existing)
context_manager  = None    # EnhancedConversationContext (your existing)
streaming_callback = None  # your existing streaming callback

# ── NEW V4 components ─────────────────────────────────────────────
moe_embedder     = None    # MoE multi-model embedder
faiss_manager    = None    # multi-index FAISS manager
neo4j_db         = None    # Neo4j graph DB connection
sql_engine       = None    # SQLAlchemy engine
knowledge_fabric = None    # Knowledge Fabric Layer (snapshot isolation)
nli_verifier     = None    # NLI fact verifier
reranker         = None    # Cross-encoder re-ranker


def get_system_state():
    """
    Returns all state objects.
    Extended from your original 7-tuple to include V4 components.
    """
    return (
        system_initialized,
        initialization_error,
        doc_manager,
        context_manager,
        app_config,
        embedding_model,
        llm,
        # V4 additions:
        moe_embedder,
        faiss_manager,
        neo4j_db,
        knowledge_fabric,
    )


def initialize_system():
    """
    Initialize the full V4 GraphRAG system.
    Follows your existing initialization pattern exactly.
    V4 components initialize AFTER your existing components
    so existing functionality is never broken.
    """
    global system_initialized, initialization_error
    global embedding_model, llm, doc_manager, context_manager, streaming_callback
    global moe_embedder, faiss_manager, neo4j_db, sql_engine, knowledge_fabric
    global nli_verifier, reranker

    try:
        print("=" * 60)
        print("[START] Initializing V4 GraphRAG System")
        print("=" * 60)

        # ── STEP 1: Your existing embedding model init (unchanged) ──
        print(f"\n[1/8] Loading embedding model on {app_config.DEVICE}...")
        from langchain_huggingface import HuggingFaceEmbeddings

        loaded_model_name = None
        for model_name in app_config.EMBED_MODEL_OPTIONS:
            try:
                print(f"  Trying {model_name}...")
                embedding_model = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': app_config.DEVICE},
                    encode_kwargs={'normalize_embeddings': True}
                )
                loaded_model_name = model_name
                print(f"  [OK] Loaded: {model_name} on {app_config.DEVICE}")
                break
            except Exception as e:
                print(f"  [WARN] Failed {model_name}: {e}")
        if embedding_model is None:
            raise Exception("Could not load any embedding model")

        # ── STEP 2: Your existing LLM init (unchanged) ──────────────
        print(f"\n[2/8] Initializing language model ({app_config.LLM_MODEL})...")
        try:
            from langchain_community.llms import Ollama
            llm = Ollama(
                model=app_config.LLM_MODEL,
                temperature=0.1,
                base_url=app_config.OLLAMA_BASE_URL,
                num_ctx=app_config.OLLAMA_NUM_CTX,
            )
            print(f"  [OK] LLM ready: {app_config.LLM_MODEL} @ {app_config.OLLAMA_BASE_URL}")
        except Exception as e:
            print(f"  [WARN] LLM init failed: {e} — continuing in search-only mode")
            llm = None

        # ── STEP 3: Your existing document manager ──────────────────
        print("\n[3/8] Initializing document manager...")
        from modules.document_manager import AdvancedDocumentManager
        from modules.conversation import EnhancedConversationContext
        os.makedirs(app_config.EMBEDDINGS_DIR, exist_ok=True)
        doc_manager     = AdvancedDocumentManager(
            str(app_config.EMBEDDINGS_DIR),
            embedding_model=embedding_model
        )
        context_manager = EnhancedConversationContext()
        print("  [OK] Document manager ready")

        # ── STEP 4: Your existing document auto-load ─────────────────
        print("\n[4/8] Loading documents (60s timeout)...")
        from threading import Thread
        def _load():
            try:
                doc_manager.load_all_documents()
            except Exception as e:
                print(f"  [WARN] Doc loading warning: {e}")

        t = Thread(target=_load, daemon=True)
        t.start()
        t.join(timeout=60)
        if t.is_alive():
            print("  [WARN] Document loading timed out — continuing")
        else:
            print(f"  [OK] Documents loaded: {len(doc_manager.loaded_documents)} available")

        # ── STEP 5: NEW V4 — MoE multi-model embedder ────────────────
        print("\n[5/8] Initializing MoE embedder (V4)...")
        try:
            from app.ingestion.embedder import MoEEmbedder
            moe_embedder = MoEEmbedder(fallback_model=embedding_model)
            print("  [OK] MoE embedder ready")
        except Exception as e:
            print(f"  [WARN] MoE embedder failed: {e} — using primary model only")
            moe_embedder = None

        # ── STEP 6: NEW V4 — FAISS multi-index manager ───────────────
        print("\n[6/8] Initializing FAISS multi-index (V4)...")
        try:
            from app.core.database import FAISSManager
            faiss_manager = FAISSManager()
            print("  [OK] FAISS multi-index ready")
        except Exception as e:
            print(f"  [WARN] FAISS multi-index failed: {e}")
            faiss_manager = None

        # ── STEP 7: NEW V4 — SQL DB + Neo4j (non-blocking) ───────────
        print("\n[7/8] Initializing databases (V4)...")
        try:
            from app.core.database import init_db, engine as _engine
            sql_engine = _engine
            init_db()
            print("  [OK] SQL database ready")
        except Exception as e:
            print(f"  [WARN] SQL DB failed: {e}")

        try:
            from app.core.database import Neo4jDB
            neo4j_db = Neo4jDB
            # Verify connection
            with neo4j_db.session() as s:
                s.run("RETURN 1")
            print("  [OK] Neo4j connected")
        except Exception as e:
            print(f"  [WARN] Neo4j not available: {e} — graph features disabled")
            neo4j_db = None

        # ── STEP 8: NEW V4 — Knowledge Fabric + lazy NLI/reranker ────
        print("\n[8/8] Initializing Knowledge Fabric (V4)...")
        try:
            from app.graph.knowledge_fabric import KnowledgeFabric
            knowledge_fabric = KnowledgeFabric()
            print("  [OK] Knowledge Fabric ready")
        except Exception as e:
            print(f"  [WARN] Knowledge Fabric failed: {e}")
            knowledge_fabric = None

        # NLI and re-ranker load lazily (first query that needs them)
        print("  [INFO] NLI verifier + re-ranker: lazy load on first query")

        system_initialized = True
        print("\n" + "=" * 60)
        print("[OK] V4 GraphRAG System Ready")
        print("=" * 60)
        _print_startup_summary()

    except Exception as e:
        system_initialized = False
        initialization_error = str(e)
        print(f"\n[FAIL] Initialization failed: {e}")
        raise


def _print_startup_summary():
    """Print a clean summary of what is active."""
    status = app_config.get_capability_status()
    print("\n[INFO] Active Capabilities:")
    for key, val in status.items():
        print(f"   {key:<12} {val}")
    print(f"\n   docs loaded  {len(doc_manager.loaded_documents) if doc_manager else 0}")
    print(f"   llm          {'[OK] ' + app_config.LLM_MODEL if llm else '[OFFLINE]'}")
    print(f"   neo4j        {'[OK] connected' if neo4j_db else '[OFFLINE] not available'}")
    print(f"   moe embed    {'[OK] active' if moe_embedder else '[WARN] primary only'}")
    print()
