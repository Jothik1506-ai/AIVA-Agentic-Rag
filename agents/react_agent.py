"""
agents/react_agent.py

LLM-controlled Agentic RAG (ReAct) loop.

Unlike _agentic_retrieve in chat_routes (a Python-controlled
retrieve → evaluate → rewrite loop), here the LLM itself drives retrieval:

    User Query ─> Agent (LLM) ─> Tool Call ─> Observation ─> reflect
                     ▲                                          │
                     └──────────── (repeat if needed) ──────────┘
                                        │
                                  Final Answer

The LLM replies in a strict JSON protocol (works with models that lack
native OpenAI function calling — e.g. LM Studio local models):

    {"thought": "...", "action": {"tool": "...", "args": {...}}}
    {"thought": "...", "final_answer": "..."}

Tools exposed to the model:
    search_knowledge_base(query, k)      — semantic search across documents
    get_document_pages(doc_name, page_number) — full chunks of one page
    list_available_documents()           — indexed docs + metadata
"""

import json
import re
import logging
from typing import Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

MAX_ITERATIONS_DEFAULT = 5
_OBSERVATION_CHAR_LIMIT = 6000   # keep observations inside the context window
_CHUNK_PREVIEW_CHARS = 500

SYSTEM_PROMPT = """You are a highly capable Agentic RAG Assistant. Your goal is to answer the user's questions accurately using only verified information retrieved from the knowledge base.

To do this, you have access to tools that search and retrieve document content. You must think step-by-step, plan your searches, evaluate the quality of retrieved information, and reformulate your queries if you do not find the answer immediately.

### AVAILABLE TOOLS

1. search_knowledge_base(query: str, k: int = 5)
   - Performs a semantic search across all uploaded documents.
   - Example: {"tool": "search_knowledge_base", "args": {"query": "revenue trends 2023", "k": 5}}

2. get_document_pages(doc_name: str, page_number: int)
   - Retrieves the raw text, tables, and figures for a specific page of a document.
   - Example: {"tool": "get_document_pages", "args": {"doc_name": "annual_report.pdf", "page_number": 12}}

3. list_available_documents()
   - Lists all indexed documents, their metadata, total pages, and types.
   - Example: {"tool": "list_available_documents", "args": {}}

### CONSTRAINTS & BEHAVIOR
1. Self-Correction & Iteration: If the search results do not contain the answer, do not guess. Analyze why, reformulate your search query (synonyms, broader terms, or specific entities), and search again. You can call tools multiple times.
2. Grounding: Answer ONLY using facts retrieved in the observations. If the facts are missing, and repeated searches yield nothing, state clearly that the information is not available in the database.
3. Citations: Always cite the source document name and page number when stating a fact (e.g., "[Source: annual_report.pdf, Page 12]").

### PROTOCOL
For every turn, you must respond with EXACTLY ONE JSON object inside a ```json code block, in one of two forms:

Option A — call a tool:
```json
{
  "thought": "What you are looking for, why you chose this tool, and what you expect to find.",
  "action": {
    "tool": "tool_name",
    "args": { }
  }
}
```

Option B — deliver the final answer:
```json
{
  "thought": "Review the observations and confirm the answer is fully grounded in the retrieved sources.",
  "final_answer": "Your detailed answer here with proper source citations."
}
```

Never output anything outside the single JSON code block."""


# ── Tools ─────────────────────────────────────────────────────────────────────

def _tool_search_knowledge_base(doc_manager, context_manager, query: str, k: int = 5) -> Tuple[str, List[Dict]]:
    """Semantic search. Returns (observation_text, source_dicts)."""
    from modules.search import intelligent_document_search
    k = max(1, min(int(k or 5), 15))
    results, _ = intelligent_document_search(
        query, doc_manager, k=k, fast=True, context_manager=context_manager,
    )
    if not results:
        return "No results found for this query. Try different or broader terms.", []

    lines, sources = [], []
    for i, doc in enumerate(results[:k], 1):
        meta = getattr(doc, "metadata", {}) or {}
        name = meta.get("source_document", "unknown")
        page = meta.get("display_page", meta.get("page", "?"))
        text = (doc.page_content or "").strip().replace("\n\n", "\n")
        lines.append(f"[{i}] Source: {name}, Page {page}\n{text[:_CHUNK_PREVIEW_CHARS]}")
        sources.append({"document": name, "page": page})
    return "\n\n".join(lines), sources


def _tool_get_document_pages(doc_manager, doc_name: str, page_number: int) -> Tuple[str, List[Dict]]:
    """Full chunks for one page of one document."""
    loaded = getattr(doc_manager, "loaded_documents", {}) or {}
    if doc_name not in loaded:
        names = ", ".join(sorted(loaded)) or "(none loaded)"
        return f"Document '{doc_name}' not found. Available documents: {names}", []

    try:
        page_number = int(page_number)
    except (TypeError, ValueError):
        return "page_number must be an integer.", []

    db = loaded[doc_name].get("db")
    store = getattr(getattr(db, "docstore", None), "_dict", {}) or {}
    chunks = []
    for d in store.values():
        meta = getattr(d, "metadata", {}) or {}
        page = meta.get("display_page", meta.get("page"))
        if page == page_number:
            chunks.append(d.page_content or "")
    if not chunks:
        total = loaded[doc_name].get("metadata", {}).get("total_pages", "?")
        return (f"No content stored for page {page_number} of '{doc_name}' "
                f"(document has {total} pages).", [])

    text = "\n\n".join(chunks)[:_OBSERVATION_CHAR_LIMIT]
    return (f"Content of {doc_name}, page {page_number}:\n{text}",
            [{"document": doc_name, "page": page_number}])


def _tool_list_available_documents(doc_manager) -> Tuple[str, List[Dict]]:
    loaded = getattr(doc_manager, "loaded_documents", {}) or {}
    if not loaded:
        return "No documents are currently loaded in the knowledge base.", []
    lines = []
    for name, data in loaded.items():
        meta = data.get("metadata", {}) or {}
        stats = data.get("stats", {}) or {}
        lines.append(
            f"- {name}: title={meta.get('title', 'Unknown')!r}, "
            f"author={meta.get('author', 'Unknown')!r}, "
            f"pages={meta.get('total_pages', '?')}, chunks={stats.get('chunks', '?')}"
        )
    return "Indexed documents:\n" + "\n".join(lines), []


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_agent_response(content: str) -> dict:
    """Extract the JSON object from the model output. Raises ValueError."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    candidate = m.group(1) if m else None
    if candidate is None:
        # Fall back to the outermost braces in the raw text
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in model output")
        candidate = content[start:end + 1]
    return json.loads(candidate)


def _recover_native_tool_call(exc: Exception):
    """
    Some hosted models (e.g. Llama on Groq) emit their NATIVE tool-call tokens
    when they see tool descriptions in the prompt. Since we declare no
    API-side tools, the provider rejects the generation with a 400
    'tool_use_failed' error whose body contains the intended call as
    failed_generation: '{"name": ..., "arguments": {...}}'.
    Convert that into our JSON protocol so the loop can continue.
    """
    s = str(exc)
    if "tool_use_failed" not in s:
        return None
    m = re.search(r"['\"]failed_generation['\"]:\s*'(.+?)'\s*\}\s*\}?\s*$", s, re.DOTALL)
    if not m:
        m = re.search(r"['\"]failed_generation['\"]:\s*\"(.+?)\"\s*\}\s*\}?\s*$", s, re.DOTALL)
    if not m:
        return None
    try:
        call = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    name = call.get("name") or call.get("tool")
    args = call.get("arguments") or call.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not name:
        return None
    return {"thought": "(recovered from native tool-call output)",
            "action": {"tool": name, "args": args}}


# ── Agent loop ────────────────────────────────────────────────────────────────

def agent_query_loop(
    query: str,
    chat_history: str,
    doc_manager,
    context_manager,
    llm_invoke: Callable,
    max_iterations: int = MAX_ITERATIONS_DEFAULT,
) -> Tuple[str, List[Dict], List[Dict]]:
    """
    Run the ReAct loop.

    llm_invoke(messages) must accept a list of {"role", "content"} dicts and
    return an object with a .content string (LangChain chat model interface).

    Returns:
        (final_answer, trace, sources)
        trace   — per-iteration dicts for the agentic debug panel.
        sources — deduplicated [{document, page}] cited by tool observations.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"User Query: {query}\nConversation History: {chat_history or '(none)'}"},
    ]
    trace: List[Dict] = []
    sources: List[Dict] = []
    seen_sources = set()

    def _add_sources(new):
        for s in new:
            key = (s.get("document"), s.get("page"))
            if key not in seen_sources:
                seen_sources.add(key)
                sources.append(s)

    for iteration in range(1, max_iterations + 1):
        try:
            response = llm_invoke(messages)
            content = getattr(response, "content", str(response))
            recovered = None
        except Exception as e:
            recovered = _recover_native_tool_call(e)
            if recovered is None:
                raise
            content = "```json\n" + json.dumps(recovered) + "\n```"
            logger.info("[ReAct] recovered native tool call: %s", recovered["action"])

        try:
            action_data = recovered or _parse_agent_response(content)
        except (ValueError, json.JSONDecodeError) as e:
            # One recovery attempt: remind the model of the protocol
            trace.append({"iteration": iteration, "error": f"unparseable response: {e}",
                          "raw": content[:400]})
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content":
                             "Your last reply was not a single valid JSON object in a ```json block. "
                             "Reply again following the PROTOCOL exactly."})
            continue

        thought = str(action_data.get("thought", ""))

        # ── Final answer ──────────────────────────────────────────────────────
        if "final_answer" in action_data:
            trace.append({"iteration": iteration, "thought": thought, "final": True})
            return str(action_data["final_answer"]), trace, sources

        # ── Tool call ─────────────────────────────────────────────────────────
        action = action_data.get("action") or {}
        tool = action.get("tool", "")
        args = action.get("args") or {}

        if tool == "search_knowledge_base":
            observation, found = _tool_search_knowledge_base(
                doc_manager, context_manager,
                str(args.get("query") or query), args.get("k", 5))
            _add_sources(found)
        elif tool == "get_document_pages":
            observation, found = _tool_get_document_pages(
                doc_manager, str(args.get("doc_name", "")), args.get("page_number", 0))
            _add_sources(found)
        elif tool == "list_available_documents":
            observation, _ = _tool_list_available_documents(doc_manager)
        else:
            observation = (f"Error: tool '{tool}' does not exist. Available tools: "
                           "search_knowledge_base, get_document_pages, list_available_documents.")

        observation = observation[:_OBSERVATION_CHAR_LIMIT]
        trace.append({
            "iteration": iteration,
            "thought": thought,
            "tool": tool,
            "args": args,
            "observation_preview": observation[:300],
        })
        logger.info("[ReAct] iter=%d tool=%s args=%s", iteration, tool, args)

        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    # Iterations exhausted — ask for a best-effort grounded summary
    messages.append({"role": "user", "content":
                     "You have used all reasoning steps. Based ONLY on the observations above, "
                     "give your best final answer now, citing sources, or state what is missing. "
                     "Reply as plain text."})
    try:
        response = llm_invoke(messages)
        answer = getattr(response, "content", str(response))
    except Exception as e:
        if _recover_native_tool_call(e) is None:
            raise
        answer = ("I could not finish reasoning within the step limit. "
                  "Please try rephrasing your question.")
    trace.append({"iteration": max_iterations + 1, "final": True, "forced": True})
    return answer, trace, sources
