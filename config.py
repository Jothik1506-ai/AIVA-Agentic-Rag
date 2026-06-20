"""
config.py  ─  V4 GraphRAG Unified Configuration
================================================
Inherits and extends your existing RAGConfig from KB-Bot.
All your existing settings are preserved. V4 additions are
clearly marked with  # ── NEW V4 ──
"""

import os
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Neither CUDA nor MPS.*", category=UserWarning)


# ── Device detection (your existing logic, unchanged) ─────────────
def get_optimal_device():
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    except ImportError:
        return 'cpu'

OPTIMAL_DEVICE = get_optimal_device()


# ── Optional dependency flags (your existing pattern) ─────────────
def _try_import(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False

HAS_PANDAS    = _try_import("pandas")
HAS_TABULA    = _try_import("tabula")
HAS_CAMELOT   = _try_import("camelot")
HAS_PYMUPDF   = _try_import("fitz")
HAS_OCR       = _try_import("easyocr")
HAS_PIL       = _try_import("PIL")
HAS_OPENCV    = _try_import("cv2")
HAS_MATPLOTLIB= _try_import("matplotlib")
HAS_NETWORKX  = _try_import("networkx")
HAS_KEYBERT   = _try_import("keybert")
HAS_SPACY     = _try_import("spacy")


# ══════════════════════════════════════════════════════════════════
class RAGConfig:
    """
    Unified config.
    Your original KB-Bot settings are in Section A.
    V4 additions are in Section B.
    """

    # ─────────────────────────────────────────────────────────────
    # SECTION A  ─  YOUR ORIGINAL KB-BOT SETTINGS (unchanged)
    # ─────────────────────────────────────────────────────────────

    PROJECT_ROOT   = Path(__file__).parent.absolute()
    EMBEDDINGS_DIR = PROJECT_ROOT / "embedding"     # keeps your existing folder
    DOCS_DIR       = PROJECT_ROOT / "docs"
    IMAGES_DIR     = PROJECT_ROOT / "images"
    DATA_DIR       = PROJECT_ROOT / "data"          # NEW V4 sqlite lives here

    @classmethod
    def get_embeddings_path(cls) -> str:
        return str(cls.EMBEDDINGS_DIR)

    # Chunking (your existing values)
    CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1200"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Embedding models — your existing MoE list
    EMBED_MODEL_OPTIONS: List[str] = [
        "sentence-transformers/all-MiniLM-L6-v2",       # fast fallback
        "sentence-transformers/all-mpnet-base-v2",       # primary general
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ]
    EMBED_MODEL = EMBED_MODEL_OPTIONS[1]

    # LLM (your existing Ollama config)
    LLM_MODEL      = os.getenv("LLM_MODEL", "gemma2:9b")
    OLLAMA_BASE_URL= os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))

    # Device
    DEVICE      = OPTIMAL_DEVICE
    DEVICE_NAME = {
        'mps':  'Apple Silicon GPU (MPS)',
        'cuda': 'NVIDIA GPU (CUDA)',
        'cpu':  'CPU',
    }.get(OPTIMAL_DEVICE, 'CPU')

    # Search (your existing values)
    FAST_QUERY_K          = 12
    FAST_QUERY_MAX_CHARS  = 3000
    MAX_CONTEXT_CHARS     = 10000
    SEARCH_CACHE_SIZE     = 200
    PARALLEL_SEARCH_THREADS = 4
    MIN_RELEVANCE_THRESHOLD = 0.3

    # Advanced NLP flags (your existing)
    ENABLE_KEYWORD_SEARCH  = True
    ENABLE_ENTITY_SEARCH   = True
    ENABLE_SEMANTIC_EXPANSION = True
    KEYWORD_BOOST_FACTOR   = 0.3
    ENTITY_BOOST_FACTOR    = 0.4
    CONTEXT_BOOST_FACTOR   = 0.2
    MAX_QUERY_EXPANSIONS   = 3
    SIMILARITY_THRESHOLD   = 0.7
    EXPANSION_DIVERSITY    = 0.8
    CONTEXT_MEMORY_SIZE    = 20
    DOCUMENT_CONTEXT_WEIGHT = 0.7
    ENTITY_CONTEXT_WEIGHT  = 0.3

    # Document processing (your existing)
    MAX_FILE_SIZE_MB         = 100
    ENABLE_OCR_IN_DOCUMENTS  = True
    ENABLE_TABLE_EXTRACTION  = True
    DEFAULT_ENCODING         = 'utf-8'
    FALLBACK_ENCODINGS       = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    IMAGE_MIN_SIZE           = 75
    IMAGE_MAX_SIZE           = 3000
    OCR_CONFIDENCE_THRESHOLD = 0.3
    TABLE_MIN_ROWS           = 2
    SUPPORTED_DOCUMENT_FORMATS = ['.pdf', '.pptx', '.ppt', '.docx', '.txt', '.html', '.md']

    # Conversation behavior (your existing)
    USE_CONTEXT_AFTER_FIRST = False

    # Company variants (your existing)
    COMPANY_VARIANTS = [
        "intouch", "intouchcx", "intouchx", "intouchc", "in touch",
        "intouch cx", "intouch x", "intouch c", "intouche",
    ]

    # Capabilities (your existing pattern)
    CAPABILITIES = {
        'advanced_tables':    HAS_TABULA or HAS_CAMELOT,
        'image_processing':   HAS_PYMUPDF and HAS_OCR,
        'chart_analysis':     HAS_MATPLOTLIB,
        'enhanced_search':    True,
        'conversation_context': True,
        'multi_format':       True,
        'keyword_extraction': HAS_KEYBERT,
        'entity_recognition': HAS_SPACY,
        'semantic_expansion': HAS_KEYBERT and HAS_SPACY,
        'advanced_metadata':  HAS_KEYBERT and HAS_SPACY,
    }

    # ─────────────────────────────────────────────────────────────
    # SECTION B  ─  V4 GraphRAG NEW ADDITIONS
    # ─────────────────────────────────────────────────────────────

    # ── NEW V4: Authentication & Security ────────────────────────
    SECRET_KEY          = os.getenv("SECRET_KEY", "CHANGE_ME_IN_PRODUCTION_V4_GRAPHRAG")
    TOKEN_EXPIRE_MINUTES= int(os.getenv("TOKEN_EXPIRE_MINUTES", "480"))
    ALGORITHM           = "HS256"

    # ── NEW V4: Database ─────────────────────────────────────────
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/graphrag.db")

    # ── NEW V4: Neo4j Knowledge Graph ────────────────────────────
    NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    # ── NEW V4: FAISS multi-model indexes ────────────────────────
    FAISS_INDEX_DIR = PROJECT_ROOT / "embedding" / "faiss_indexes"
    FAISS_INDEXES   = {
        "bge":   "faiss_bge.index",
        "e5":    "faiss_e5.index",
        "code":  "faiss_code.index",
        "legal": "faiss_legal.index",
    }

    # ── NEW V4: Extended MoE embedding models ────────────────────
    MOE_EMBED_MODELS: Dict[str, Dict] = {
        "bge":   {"model": "BAAI/bge-large-en-v1.5",
                  "domains": ["general", "research", "analytical"], "dim": 1024},
        "e5":    {"model": "intfloat/e5-large-v2",
                  "domains": ["research", "technical"], "dim": 1024},
        "code":  {"model": "microsoft/codebert-base",
                  "domains": ["code", "engineering"], "dim": 768},
        "legal": {"model": "nlpaueb/legal-bert-base-uncased",
                  "domains": ["legal", "compliance"], "dim": 768},
        # Fallback: your existing models are used when new ones unavailable
        "minilm":{"model": "sentence-transformers/all-MiniLM-L6-v2",
                  "domains": ["general"], "dim": 384},
        "mpnet": {"model": "sentence-transformers/all-mpnet-base-v2",
                  "domains": ["general", "research"], "dim": 768},
    }

    # ── NEW V4: LLM choices (multi-model selection) ───────────────
    LLM_MODELS_AVAILABLE = [
        {"id": "gemma2",      "name": "Gemma 2 (9B)",           "provider": "ollama",    "model_name": "gemma2:9b"},
        {"id": "llama3",      "name": "Llama 3.1 (8B)",         "provider": "ollama",    "model_name": "llama3.1:8b"},
        {"id": "mistral",     "name": "Mistral (7B)",            "provider": "ollama",    "model_name": "mistral:7b"},
        {"id": "tinyllama",   "name": "TinyLlama (1.1B)",        "provider": "ollama",    "model_name": "tinyllama"},  # your existing default
        {"id": "openai_gpt4", "name": "GPT-4o",                  "provider": "openai",    "model_name": "gpt-4o"},
        {"id": "claude",      "name": "Claude 3.5 Sonnet",       "provider": "anthropic", "model_name": "claude-sonnet-4-5"},
    ]
    LLM_DEFAULT = os.getenv("LLM_DEFAULT", "tinyllama")   # tinyllama = your existing default

    # ── NEW V4: RBAC role definitions ─────────────────────────────
    ROLES: Dict[str, Dict] = {
        "agent":          {"level": 1, "max_documents": 100,  "can_upload": True},
        "manager":        {"level": 2, "max_documents": 300,  "can_upload": True},
        "vice_president": {"level": 3, "max_documents": None, "can_upload": True},
    }

    # ── NEW V4: Re-ranker ─────────────────────────────────────────
    RERANKER_MODEL    = "BAAI/bge-reranker-large"
    RERANKER_TOP_K_IN = 15
    RERANKER_TOP_K_OUT= 5

    # ── NEW V4: NLI verifier ─────────────────────────────────────
    NLI_MODEL          = "cross-encoder/nli-deberta-v3-large"
    NLI_AMBIGUITY_LOW  = 0.40
    NLI_AMBIGUITY_HIGH = 0.75

    # ── NEW V4: Graph traversal ───────────────────────────────────
    GRAPH_BEAM_WIDTH          = 3
    GRAPH_MAX_DEPTH           = 3
    GRAPH_RELEVANCE_THRESHOLD = 0.65
    GRAPH_HOP_DECAY           = 0.95

    # ── NEW V4: Adaptive context budget ──────────────────────────
    CONTEXT_BUDGET = {
        "factual":     3,
        "analytical":  7,
        "multi_hop":   15,
        "comparative": 10,
    }
    PRUNE_EXPANSION_THRESHOLD = 0.40   # if >40% pruned, expand 1.5x

    # ── NEW V4: Bayesian confidence ───────────────────────────────
    CONFIDENCE_HOP_PENALTY = 0.95     # conf × 0.95^hop
    CONFIDENCE_MIN_CRAG    = 0.60     # CRAG rejects below this

    # ── NEW V4: Knowledge Fabric schema version ───────────────────
    FABRIC_SCHEMA_VERSION = "2.1"

    @classmethod
    def ensure_dirs(cls):
        """Create all required directories"""
        for d in [cls.EMBEDDINGS_DIR, cls.DOCS_DIR, cls.IMAGES_DIR,
                  cls.DATA_DIR, cls.FAISS_INDEX_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_capability_status(cls) -> Dict[str, str]:
        """Your existing capability status method — unchanged"""
        s = {}
        s['device']   = f"[OK] ({cls.DEVICE_NAME})"
        s['tables']   = f"{'[OK]' if cls.CAPABILITIES['advanced_tables'] else '[OFFLINE]'} ({'Tabula' if HAS_TABULA else ''}{' + Camelot' if HAS_CAMELOT else ' pattern-based'})"
        s['images']   = f"{'[OK]' if cls.CAPABILITIES['image_processing'] else '[OFFLINE]'} ({'PyMuPDF + EasyOCR' if HAS_PYMUPDF and HAS_OCR else 'Limited'})"
        s['charts']   = f"{'[OK]' if cls.CAPABILITIES['chart_analysis'] else '[OFFLINE]'} ({'Advanced' if HAS_MATPLOTLIB else 'Basic OCR only'})"
        s['search']   = "[OK] (Intelligent context-aware)"
        s['context']  = "[OK] (Entity and topic tracking)"
        s['formats']  = "[OK] (PDF, PPT, DOCX, TXT, HTML, MD)"
        s['keywords'] = f"{'[OK]' if HAS_KEYBERT else '[OFFLINE]'} ({'KeyBERT' if HAS_KEYBERT else 'Basic'})"
        s['entities'] = f"{'[OK]' if HAS_SPACY else '[OFFLINE]'} ({'spaCy NER' if HAS_SPACY else 'Basic'})"
        s['graph']    = "[OK] Neo4j Knowledge Graph"   # NEW V4
        s['rbac']     = "[OK] Role-Based Access Control"  # NEW V4
        return s


# ══════════════════════════════════════════════════════════════════
# Prompt Templates  ─  your full set preserved + V4 additions
# ══════════════════════════════════════════════════════════════════

class PromptTemplates:
    """All your original templates preserved. V4 adds graph-aware template."""

    # ── Your existing templates (unchanged) ──────────────────────

    ADVANCED_CONTEXT_QA_TEMPLATE = """You are an expert document analyst with advanced reasoning capabilities.
Provide intelligent, context-aware answers using both the document content and conversation context.
Try keeping the answers a little short, not to be too lengthy, and usually to the point; depending on the question asked.
If asked for document names or roles or specific person names, keep it very concise and brief.

CONVERSATION CONTEXT:
{conversation_context}

DOCUMENT CONTENT:
{context}

QUESTION: {question}

ANALYSIS INSTRUCTIONS:
1. REFERENCE RESOLUTION: Resolve vague references using conversation context.
2. CONTEXTUAL UNDERSTANDING: Adjust depth to match question type.
3. DATA HANDLING: Extract structured data accurately, highlight patterns.
4. INSIGHT GENERATION: Explain why information matters.
5. SOURCE ATTRIBUTION: Cite document name/section/page.
6. GENDER NEUTRALITY: Use gender-neutral language.

OUTPUT STYLE:
- Start with a short direct answer
- Use bullet points (*) for lists, bold (**text**) for emphasis
- End with a short source note

ANSWER:"""

    TABLE_ANALYSIS_TEMPLATE = """You are a data analysis expert. Provide intelligent, structured analysis of the table.

TABLE CONTENT:
{table_content}

CONVERSATION CONTEXT:
{conversation_context}

QUESTION: {question}

INSTRUCTIONS:
1. Extract all relevant entries accurately from the table.
2. Avoid assumptions beyond what is shown.
3. Identify patterns, anomalies, groupings.
4. Structure: summary → breakdown → insights.
5. Use gender-neutral language.

ANSWER:"""

    CHART_ANALYSIS_TEMPLATE = """You are a data visualization expert. Analyze this chart or graph.

CHART/VISUAL CONTENT:
{chart_content}

CONVERSATION CONTEXT:
{conversation_context}

QUESTION: {question}

INSTRUCTIONS:
1. Identify chart type and purpose.
2. Extract labels, axes, values, trends.
3. Analyze patterns and anomalies.
4. Generate insights from the data.
5. Connect to conversation context.

ANSWER:"""

    DOCUMENT_SUMMARY_TEMPLATE = """You are an expert document summarizer.

CONTEXT:
{context}

CONVERSATION CONTEXT:
{conversation_context}

QUESTION: {question}

INSTRUCTIONS:
1. Analyze ALL provided content comprehensively.
2. Structure: Overview → Key Concepts → Details → Technical Info → Conclusions.
3. Include specific details, numbers, names.
4. Include source attribution.

Provide a detailed comprehensive summary:"""

    PAGE_SPECIFIC_TEMPLATE = """You are an expert document analyst for page-specific content.

CONVERSATION CONTEXT:
{conversation_context}

PAGE-SPECIFIC CONTENT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Confine answer to PAGE-SPECIFIC CONTENT only.
2. Describe all content including text, data, images.
3. If content is missing, state it clearly.
4. Attribute your answer to the specific page.

ANSWER:"""

    UI_ELEMENT_TEMPLATE = """You are a UI/UX expert analyzing user interface elements from screenshots.

CONVERSATION CONTEXT:
{conversation_context}

UI ELEMENT CONTENT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Identify all UI elements: popups, dialogs, buttons, tabs, screens.
2. Extract element details systematically.
3. Explain purpose, context, relationships.
4. Use exact names as shown in screenshots.

ANSWER:"""

    # ── NEW V4: Graph-aware synthesis template ────────────────────
    GRAPH_SYNTHESIS_TEMPLATE = """You are an expert analyst combining knowledge graph facts with document evidence.
Every claim must cite its source. Use ONLY the provided sources.
{domain_instruction}

QUESTION: {question}

GRAPH KNOWLEDGE (high-confidence facts from knowledge graph):
{graph_facts}

CONVERSATION CONTEXT:
{memory_context}

NUMBERED SOURCES (cite as [1], [2] etc.):
{anchors}

{conflict_note}
DERIVED CONCLUSIONS:
{derived}

QUESTION (restated): {question}

Answer with inline citations formatted as [source_num, conf=X.XX].
Start with a direct answer. Use bullet points (*) for lists.

ANSWER:"""


# ══════════════════════════════════════════════════════════════════
# Global singletons
# ══════════════════════════════════════════════════════════════════

config    = RAGConfig()
templates = PromptTemplates()

# Ensure directories exist
config.ensure_dirs()

# Module-level exports (your existing pattern)
EMBED_MODEL_OPTIONS = RAGConfig.EMBED_MODEL_OPTIONS
EMBED_MODEL         = RAGConfig.EMBED_MODEL
LLM_MODEL           = RAGConfig.LLM_MODEL

TABLE_ANALYSIS_TEMPLATE    = PromptTemplates.TABLE_ANALYSIS_TEMPLATE
CHART_ANALYSIS_TEMPLATE    = PromptTemplates.CHART_ANALYSIS_TEMPLATE
ADVANCED_CONTEXT_QA_TEMPLATE = PromptTemplates.ADVANCED_CONTEXT_QA_TEMPLATE
DOCUMENT_SUMMARY_TEMPLATE  = PromptTemplates.DOCUMENT_SUMMARY_TEMPLATE
PAGE_SPECIFIC_TEMPLATE     = PromptTemplates.PAGE_SPECIFIC_TEMPLATE
UI_ELEMENT_TEMPLATE        = PromptTemplates.UI_ELEMENT_TEMPLATE
GRAPH_SYNTHESIS_TEMPLATE   = PromptTemplates.GRAPH_SYNTHESIS_TEMPLATE  # NEW V4
