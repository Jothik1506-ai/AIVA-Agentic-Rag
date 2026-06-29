# modules/response_generator.py
"""
Response generation and streaming module.
Handles prompt generation and response streaming.
"""

import json
import time
import re


class EnhancedStreamingCallback:
    """Simple streaming callback for Flask"""
    def __init__(self):
        self.text = ""
        self.is_streaming = False

    def reset(self):
        self.text = ""
        self.is_streaming = False


def generate_prompt_based_on_mode(search_mode, bot_name, user_message, combined_context, 
                                  conversation_context, followup_force_context):
    """Generate prompt based on search mode"""
    
    if search_mode == 'general_plus_docs':
        if followup_force_context:
            return f"""You are {bot_name}, a concise AI assistant.

CONVERSATION CONTEXT:
{conversation_context}

DOCUMENT CONTENT (if available):
{combined_context if combined_context else "No relevant document content found."}

QUESTION: {user_message}

INSTRUCTIONS:
- Give a short, direct follow-up answer (2-4 sentences max unless a list is clearly needed).
- Use document content if relevant, otherwise use general knowledge.
- No padding, no repetition, no lengthy intros.

ANSWER:"""
        else:
            return f"""You are {bot_name}, a concise AI assistant.

CONVERSATION CONTEXT:
{conversation_context}

DOCUMENT CONTENT (if available):
{combined_context if combined_context else "No relevant document content found."}

QUESTION: {user_message}

INSTRUCTIONS:
- Answer directly and briefly (2-4 sentences unless a list/table is clearly needed).
- Prioritize document content; cite source if used.
- If documents have no relevant info, answer from general knowledge.
- No filler phrases, no lengthy explanations.

ANSWER:"""

    else:  # documents_only mode (default)
        if followup_force_context:
            return f"""You are {bot_name}. Answer from the documents below only.

CONVERSATION CONTEXT:
{conversation_context}

DOCUMENT CONTENT:
{combined_context}

QUESTION: {user_message}

INSTRUCTIONS:
- Give a short, focused follow-up answer (2-4 sentences max unless a list is clearly needed).
- Base answer strictly on the document content above.
- If not in the documents, say: "This information is not in the loaded documents."

ANSWER:"""
        else:
            return f"""You are {bot_name}. Answer from the documents below only.

CONVERSATION CONTEXT:
{conversation_context}

DOCUMENT CONTENT:
{combined_context}

QUESTION: {user_message}

INSTRUCTIONS:
- Be brief and direct (2-4 sentences max unless a list/table is clearly needed).
- Use only the document content above — no external knowledge.
- If not in the documents, say: "This information is not in the loaded documents."

ANSWER:"""


def format_sources_for_response(search_results, max_sources=6):
    """Format search results into source information"""
    formatted_sources = []
    for source in search_results[:max_sources]:
        if hasattr(source, 'metadata'):
            formatted_sources.append({
                'document': source.metadata.get('source_document', 'Unknown'),
                'title': source.metadata.get('document_title', 'Unknown'),
                'page': source.metadata.get('display_page', 'Unknown'),
                'type': source.metadata.get('type', 'text'),
                'has_table': source.metadata.get('has_table', False),
                'image_type': source.metadata.get('image_type', None),
                'relevance_score': source.metadata.get('relevance_score', 0.0),
                'keywords': source.metadata.get('contextual_keywords', [])[:3],
                'entities': source.metadata.get('named_entities', [])[:3],
                'citations': source.metadata.get('citations', [])[:2]
            })
    return formatted_sources


def build_context_from_results(search_results, max_results=8, is_fast=False):
    """Build context string from search results"""
    context_parts = []
    for doc in search_results[:max_results]:
        doc_name = doc.metadata.get('source_document', 'Unknown')
        page_num = doc.metadata.get('display_page', 'Unknown')
        doc_type = doc.metadata.get('type', 'text')
        snippet = doc.page_content
        
        if is_fast and len(snippet) > 2000 // 4:
            snippet = snippet[:2000 // 4]
        elif (not is_fast) and len(snippet) > 8000 // 6:
            snippet = snippet[:8000 // 6]
        
        context_parts.append(f"[Source: {doc_name} | Page: {page_num} | Type: {doc_type}]\n{snippet}")
    
    return "\n\n".join(context_parts)


def generate_stream_response(llm, prompt, search_results, start_time, max_sources=6):
    """Generate streaming response tokens"""
    try:
        # Stream tokens from model
        for chunk in llm.stream(prompt):
            yield f"data: {json.dumps({'type': 'token', 'content': str(chunk)})}\n\n"
        
        # Format and send sources
        formatted_sources = format_sources_for_response(search_results, max_sources)
        total_time = round(time.time() - start_time, 2)
        yield f"data: {json.dumps({'type': 'done', 'sources': formatted_sources, 'processing_time': total_time})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


def generate_casual_stream_response(response_text, start_time):
    """Generate streaming response for casual conversation"""
    try:
        yield ": ping\n\n"
        yield f"data: {json.dumps({'type': 'status', 'message': 'responding'})}\n\n"
        
        words = response_text.split()
        for i, word in enumerate(words):
            yield f"data: {json.dumps({'type': 'token', 'content': word + (' ' if i < len(words) - 1 else '')})}\n\n"
        
        yield f"data: {json.dumps({'type': 'done', 'sources': [], 'processing_time': round(time.time() - start_time, 2)})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


def detect_follow_up_query(query: str) -> bool:
    """Detect if query is a follow-up question"""
    follow_up_patterns = [
        r'\b(?:explain\s+more|tell\s+me\s+more|elaborate|expand|go\s+deeper|more\s+details?)\b',
        r'\b(?:what\s+about|how\s+about|can\s+you\s+explain|can\s+you\s+tell)\b',
        r'\b(?:this|that|it|they|them|these|those)\b',
        r'\b(?:previous|earlier|above|before|mentioned|discussed|said)\b',
        r'\b(?:same|similar|related|also|additionally|furthermore)\b'
    ]
    
    query_lower = query.lower()
    return any(re.search(pattern, query_lower) for pattern in follow_up_patterns)