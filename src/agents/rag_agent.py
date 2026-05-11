"""
layer4_agent.py — Layer 4: Local LLM Generation + ReAct Agent Loop

Responsibility:
    Wire the Ollama local LLM (via LangChain ChatOllama) to the three RAG tools
    (keyword_search, semantic_search, chunk_read) and run a ReAct-style agent
    loop that iterates tool calls until the LLM produces a final answer.

Key components:
    SYSTEM_PROMPT        — Instructs the LLM on tool strategy and citation behaviour.
    build_llm()          — Instantiates ChatOllama and binds the three tools.
    run_agent(query)     — The ReAct loop: think → tool → observe → ... → answer.

Usage:
    from layer4_agent import run_agent
    answer = run_agent("What is the bypass procedure for the UPS system?")
"""

import json
import logging
from typing import Any

from ollama import Client

from src.core.config import LLM_PROVIDER, GROQ_MODEL, GROQ_API_KEY, OLLAMA_MODEL, LLM_TEMPERATURE
from src.tools.rag_tools import (
    chunk_read,
    get_document_chunks,
    keyword_search,
    list_documents,
    reset_c_read,
    semantic_search,
    triple_lookup,
    parse_document,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT: str = """You are an expert research analyst. Your task is to provide detailed, accurate, and relevant answers by synthesizing information from government contracts.

CRITICAL GUIDELINES:
1. Multi-Hop Reasoning: Link facts across sections. If fact A points to entity B, search for entity B to complete the logic.
2. Synthesis over Extraction: Synthesize a coherent narrative that directly answers the user. Explain "why" and "how" only if relevant to the query.
3. Iterative Exploration: Use tools until you have a COMPLETE answer. If a tool call reveals a new lead, follow it.
4. Snippet vs. Full Text: Search results are previews. You MUST use chunk_read for the full context before concluding.
5. Contextual Awareness: Respect dates and amendments. Ensure you are looking at the most recent information.

*** ACCURACY & RELEVANCE RULES (STRICT) ***:
1. Stay Focused: Answer the SPECIFIC question asked. Do NOT include unrelated document data (like prices, dates, or contact info) if it was not requested.
2. Be Comprehensive but Concise: Provide all necessary details for the query, but avoid "info-dumping" the entire document.
3. Cross-Reference: Mention which sections/documents the information came from.
4. Factual Integrity: Use ONLY retrieved info. NEVER invent facts or assume details.
5. Entity Disambiguation: Pay extreme attention to EXACT name/ID matches. "John D" is NOT the same entity as "John". If the user asks for a specific name with an initial/surname (e.g., "Patel Vijaykumar N") and you only find a partial match (e.g., "Patel Vijaykumar"), you MUST state that the exact person was not found. DO NOT mistakenly attribute data of a partial match to the user's explicit query.

PROCEDURE — Fact-Finding & Reasoning:
  Step 1: Break the query into required data points.
  Step 2: Use triple_lookup or search tools to find anchors.
  Step 3: Read relevant chunks in full via chunk_read.
  Step 4: Synthesize the answer, ensuring all logic is explained.

OUTPUT FORMAT:
**Reasoning:** <Short, 1-2 sentence explanation of how you found the specific answer.>

**Sources:** <List of chunk_ids used.>

**Final Answer:** <Detailed response focused ONLY on the user's query.>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Hallucination guard helper
# ─────────────────────────────────────────────────────────────────────────────

def _has_chunk_errors(messages: list[dict]) -> bool:
    """Return True if any ToolMessage in the history contains chunk-not-found errors."""
    for msg in messages:
        if msg.get("role") == "tool":
            content = str(msg.get("content", ""))
            if "not found in the database" in content or ("chunk_id" in content and "not found" in content):
                return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
# Tool Mappings
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_MAP: dict[str, Any] = {
    "list_documents":      list_documents,
    "get_document_chunks": get_document_chunks,
    "keyword_search":      keyword_search,
    "semantic_search":     semantic_search,
    "chunk_read":          chunk_read,
    "triple_lookup":       triple_lookup,
    "parse_document":      parse_document,
}

OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List all documents that have been ingested into the RAG system.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_chunks",
            "description": "Retrieve the first N chunk IDs and their snippets for a specific document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "integer", "description": "The integer doc_id"},
                    "max_chunks": {"type": "integer", "description": "Maximum number of chunks to retrieve"}
                },
                "required": ["doc_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "keyword_search",
            "description": "Exact lexical matching across documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {"type": "array", "items": {"type": "string"}, "description": "List of search terms"},
                    "top_k": {"type": "integer", "description": "Number of top results"}
                },
                "required": ["keywords"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Semantic similarity search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query"},
                    "top_k": {"type": "integer", "description": "Number of top semantic results"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "chunk_read",
            "description": "Read the complete content of document segments by their IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_ids": {"type": "array", "items": {"type": "string"}, "description": "List of chunk IDs to read"}
                },
                "required": ["chunk_ids"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "triple_lookup",
            "description": "Direct structured lookup for factual questions about contract data. Use this for: prices, quantities, dates, names, IDs, addresses, specifications.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "The section or subject (e.g. 'Buyer', 'Product', 'Seller')"},
                    "attribute": {"type": "string", "description": "The field name (e.g. 'Unit Price', 'Organisation Name', 'GSTIN')"}
                },
                "required": ["entity", "attribute"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "parse_document",
            "description": "Parse a PDF document into structured sections and embedding-ready chunks. Use this to inspect raw document structure, extract tables, handle bilingual text, or diagnose extraction issues.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pdf_path": {"type": "string", "description": "Absolute or relative path to the PDF file (e.g. 'data/raw/contract.pdf')"}
                },
                "required": ["pdf_path"]
            }
        }
    }
]

# ─────────────────────────────────────────────────────────────────────────────
# LLM Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_llm() -> Any:
    """Instantiate the LLM Client."""
    if LLM_PROVIDER == "groq":
        import groq
        logger.info("Building Groq client: model=%s, temperature=%s", GROQ_MODEL, LLM_TEMPERATURE)
        client = groq.Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client ready with tools.")
        return client
    else:
        logger.info("Building Ollama client: model=%s, temperature=%s", OLLAMA_MODEL, LLM_TEMPERATURE)
        client = Client() # connects to localhost:11434
        logger.info("Ollama client ready with tools.")
        return client

# ─────────────────────────────────────────────────────────────────────────────
# ReAct Agent Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(
    query: str,
    max_iterations: int = 10,
    llm_with_tools: Any | None = None,
) -> str:
    """Run the ReAct agent loop using native Ollama library."""

    reset_c_read()

    client = llm_with_tools if llm_with_tools is not None else build_llm()

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    logger.info("=== Agent loop started: query='%s' ===", query)

    for iteration in range(max_iterations):
        logger.debug("Iteration %d/%d — invoking LLM ...", iteration + 1, max_iterations)

        if LLM_PROVIDER == "groq":
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=OLLAMA_TOOLS,
                temperature=LLM_TEMPERATURE
            )
            reply_msg_obj = response.choices[0].message
            reply_msg = {"role": "assistant"}
            if reply_msg_obj.content:
                reply_msg["content"] = reply_msg_obj.content
            
            tool_calls = []
            if reply_msg_obj.tool_calls:
                for tc in reply_msg_obj.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
                reply_msg["tool_calls"] = tool_calls
        else:
            response = client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                tools=OLLAMA_TOOLS,
                options={"raw": True, "temperature": LLM_TEMPERATURE}
            )
            reply_msg = response["message"]

        messages.append(reply_msg)

        if not reply_msg.get("tool_calls"):
            logger.info("Agent produced final answer after %d iteration(s).", iteration + 1)
            return reply_msg.get("content", "")

        for tool_call in reply_msg["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            tool_args_raw = tool_call["function"]["arguments"]
            tool_args = json.loads(tool_args_raw) if isinstance(tool_args_raw, str) else tool_args_raw

            logger.info("Tool call: %s | args=%s", tool_name, tool_args)

            if tool_name in _TOOL_MAP:
                try:
                    tool_result = _TOOL_MAP[tool_name].invoke(tool_args)
                    tool_result_str = str(tool_result)

                    # Increased truncation limit to allow for more context in multi-hop reasoning
                    if len(tool_result_str) > 8000:
                        tool_result_str = tool_result_str[:8000] + "\n... [CONTENT TRUNCATED TO SAVE TOKENS. NOT ALL DATA SHOWN.] ..."
                except Exception as exc:  # noqa: BLE001
                    tool_result_str = f"Tool '{tool_name}' raised an error: {exc}"
                    logger.error("Tool '%s' error: %s", tool_name, exc)
            else:
                tool_result_str = f"Unknown tool: {tool_name!r}"
                logger.warning("Unknown tool requested by LLM: %s", tool_name)

            tool_result_msg = {
                "role": "tool",
                "content": tool_result_str,
                "name": tool_name
            }
            if LLM_PROVIDER == "groq" and "id" in tool_call:
                tool_result_msg["tool_call_id"] = tool_call["id"]
                
            messages.append(tool_result_msg)
            logger.debug("Tool '%s' result appended to messages.", tool_name)

    logger.warning("max_iterations=%d reached without final answer. Requesting wrap-up.", max_iterations)

    chunk_error_warning = (
        "\n\n*** WARNING: Some chunk_read calls returned 'not found' errors. "
        "This means those chunk_ids do not exist in the database. "
        "Do NOT invent or assume content for them. ***"
        if _has_chunk_errors(messages) else ""
    )

    messages.append({
        "role": "user",
        "content": (
            "You have completed your document search. Now write your final answer."
            "\n\nSTRICT RULES FOR THIS FINAL ANSWER:"
            "\n1. Use ONLY text that appeared in the tool results above."
            "\n2. Do NOT use training data, prior knowledge, or assumptions."
            "\n3. Do NOT invent any names, companies, amounts, dates, or terms."
            "\n4. If the retrieved chunks did not contain enough information, "
            "output EXACTLY: \"Not found in the provided documents.\""
            + chunk_error_warning
        )
    })

    if LLM_PROVIDER == "groq":
        wrap_up = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE
        )
        return wrap_up.choices[0].message.content or ""
    else:
        wrap_up = client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"raw": True, "temperature": LLM_TEMPERATURE}
        )
        return wrap_up["message"].get("content", "")


# ─────────────────────────────────────────────────────────────────────────────
# Streaming ReAct Agent
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_stream(
    query: str,
    max_iterations: int = 5,
    llm_with_tools: Any | None = None,
):
    """Streaming variant of run_agent.

    Runs the full ReAct tool-calling loop synchronously, then streams the
    final answer back to the caller token-by-token as a Python generator.
    """
    reset_c_read()

    client = llm_with_tools if llm_with_tools is not None else build_llm()

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    logger.info("=== Streaming agent loop started: query='%s' ===", query)

    # ── Phase 1: Tool-calling loop (non-streaming, same as run_agent) ──────────
    for iteration in range(max_iterations):
        logger.debug("Stream iteration %d/%d", iteration + 1, max_iterations)

        if LLM_PROVIDER == "groq":
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=OLLAMA_TOOLS,
                temperature=LLM_TEMPERATURE
            )
            reply_msg_obj = response.choices[0].message
            reply_msg = {"role": "assistant"}
            if reply_msg_obj.content:
                reply_msg["content"] = reply_msg_obj.content
                
            tool_calls = []
            if reply_msg_obj.tool_calls:
                for tc in reply_msg_obj.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
                reply_msg["tool_calls"] = tool_calls
        else:
            response = client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                tools=OLLAMA_TOOLS,
                options={"raw": True, "temperature": LLM_TEMPERATURE}
            )
            reply_msg = response["message"]

        messages.append(reply_msg)

        # If no tool calls, the LLM has produced its final answer
        if not reply_msg.get("tool_calls"):
            content = reply_msg.get("content", "")
            logger.info("Streaming agent: final answer received (len=%d).", len(content))
            
            if content:
                chunk_size = 30
                for i in range(0, len(content), chunk_size):
                    yield content[i:i+chunk_size]
            return

        # ── Execute tool calls ────────────────────────────────────────────────
        for tool_call in reply_msg["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            tool_args_raw = tool_call["function"]["arguments"]
            tool_args = json.loads(tool_args_raw) if isinstance(tool_args_raw, str) else tool_args_raw
            logger.info("[Stream] Tool call: %s | args=%s", tool_name, tool_args)

            if tool_name in _TOOL_MAP:
                try:
                    tool_result = _TOOL_MAP[tool_name].invoke(tool_args)
                    tool_result_str = str(tool_result)
                    if len(tool_result_str) > 8000:
                        tool_result_str = tool_result_str[:8000] + "\n... [TRUNCATED] ..."
                except Exception as exc:
                    tool_result_str = f"Tool '{tool_name}' raised an error: {exc}"
            else:
                tool_result_str = f"Unknown tool: {tool_name!r}"

            tool_result_msg = {
                "role": "tool",
                "content": tool_result_str,
                "name": tool_name
            }
            if LLM_PROVIDER == "groq" and "id" in tool_call:
                tool_result_msg["tool_call_id"] = tool_call["id"]
                
            messages.append(tool_result_msg)

    # ── Phase 2: Max iterations reached — force wrap-up with streaming ─────────
    logger.warning("[Stream] max_iterations=%d reached. Requesting streamed wrap-up.", max_iterations)

    chunk_error_warning = (
        "\n\n*** WARNING: Some chunk_read calls returned 'not found' errors. Do NOT invent content. ***"
        if _has_chunk_errors(messages) else ""
    )

    messages.append({
        "role": "user",
        "content": (
            "You have completed your document search. Now write your final answer."
            "\n\nSTRICT RULES FOR THIS FINAL ANSWER:"
            "\n1. Use ONLY text that appeared in the tool results above."
            "\n2. Do NOT use training data, prior knowledge, or assumptions."
            "\n3. Do NOT invent any names, companies, amounts, dates, or terms."
            "\n4. If the retrieved chunks did not contain enough information, "
            "output EXACTLY: \"Not found in the provided documents.\""
            + chunk_error_warning
        )
    })

    if LLM_PROVIDER == "groq":
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    else:
        for chunk in client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"raw": True, "temperature": LLM_TEMPERATURE},
            stream=True,
        ):
            token_text = chunk["message"].get("content", "")
            if token_text:
                yield token_text
