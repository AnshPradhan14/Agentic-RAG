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

from src.core.config import OLLAMA_MODEL, LLM_TEMPERATURE
from src.tools.rag_tools import (
    chunk_read,
    get_document_chunks,
    keyword_search,
    list_documents,
    reset_c_read,
    semantic_search,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT: str = """You are an intelligent research assistant equipped with hierarchical retrieval tools to answer complex questions from a large corpus. Your goal is to gather sufficient evidence through iterative exploration.

CRITICAL GUIDELINES:
1. Iterative Reasoning: After each tool call, analyze the results. If the information is incomplete, decide which tool will best help you find the missing piece.
2. Snippet vs. Full Text: Initial search results (keyword_search and semantic_search) return only abbreviated snippets. These are for identification purposes only and are NOT sufficient for answering questions.
3. The 'Read' Requirement: You MUST use the chunk_read tool to access the full content of any chunk you find promising before you attempt to formulate a final answer.
4. Avoid Redundancy: Use the context history to ensure you are not repeatedly reading the same chunks. If a tool tells you a chunk has been read, pivot your search strategy.
TOOLS AVAILABLE:
- list_documents       : Returns doc_id + filename for every ingested PDF. Use this FIRST for multi-doc tasks.
- get_document_chunks  : Returns real chunk_ids for a specific doc_id. Use this for summarisation.
- keyword_search       : Finds chunks by exact word/phrase match across all documents.
- semantic_search      : Finds chunks by meaning across all documents.
- chunk_read           : Reads the FULL text of chunks. You MUST use this before answering.

*** ANTI-HALLUCINATION RULES (ABSOLUTE, NO EXCEPTIONS) ***:
1. NEVER invent names, companies, amounts, dates, or terms not present in retrieved text.
2. NEVER guess or fabricate chunk_ids. chunk_ids come ONLY from tool results.
3. If chunk_read returns an error for a chunk_id, that chunk does NOT exist — do NOT invent its contents.
4. If retrieved text is insufficient to answer, output EXACTLY: "Not found in the provided documents."
5. Do NOT answer from training data or prior knowledge under any circumstances.

PROCEDURE — Q&A queries (specific facts, buyer/seller, dates, prices):
  Step 1: Call keyword_search with 2-3 key terms.
  Step 2: Call semantic_search with the full rephrased question.
  Step 3: Combine unique chunk_ids from both results (use only IDs returned by the tools).
  Step 4: Call chunk_read with the combined chunk_ids.
  Step 5: Write your answer ONLY from the retrieved text.

PROCEDURE — Summarisation / comparison queries (summarise, compare, list all, overview):
  Step 1: Call list_documents to discover all doc_ids and filenames.
  Step 2: For EACH doc_id, call get_document_chunks(doc_id) to get that document's real chunk_ids.
  Step 3: Call chunk_read with the chunk_ids returned by get_document_chunks (never invent IDs).
  Step 4: Write a separate summary section for EACH document, labelled by its filename.
  Step 5: Only include facts that appear word-for-word in the retrieved chunks.

RULES:
- For tables, also read chunk_id + 1 (tables often span multiple chunks).
- Always label which document each fact came from.
- Do NOT repeat sentences.

OUTPUT FORMAT:
**Transformed Query:** <rewrite the user's question into clear search terms>
**Retrieved Context Summary:** <1 sentence: what the most relevant chunks contained>
**Final Answer:** <direct answer, clearly attributed per document; "Not found" if text was absent>
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
}

OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List all documents that have been ingested into the RAG system.",
            "parameters": {"type": "object", "properties": {}}
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
                    "max_chunks": {"type": "integer"}
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
                    "top_k": {"type": "integer"}
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
                    "top_k": {"type": "integer"}
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
    }
]

# ─────────────────────────────────────────────────────────────────────────────
# LLM Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_llm() -> Client:
    """Instantiate the Ollama Client.
    """
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
    llm_with_tools: Client | None = None,
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

        response = client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=OLLAMA_TOOLS,
            options={"temperature": LLM_TEMPERATURE}
        )

        reply_msg = response["message"]
        messages.append(reply_msg)

        if not reply_msg.get("tool_calls"):
            logger.info("Agent produced final answer after %d iteration(s).", iteration + 1)
            return reply_msg.get("content", "")

        for tool_call in reply_msg["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"]["arguments"]

            logger.info("Tool call: %s | args=%s", tool_name, tool_args)

            if tool_name in _TOOL_MAP:
                try:
                    tool_result = _TOOL_MAP[tool_name].invoke(tool_args)
                    tool_result_str = str(tool_result)
                    
                    # Truncate to prevent TPM explosion from huge chunks / tables
                    if len(tool_result_str) > 6000:
                        tool_result_str = tool_result_str[:6000] + "\n... [CONTENT TRUNCATED TO SAVE TOKENS. NOT ALL DATA SHOWN.] ..."
                except Exception as exc:  # noqa: BLE001
                    tool_result_str = f"Tool '{tool_name}' raised an error: {exc}"
                    logger.error("Tool '%s' error: %s", tool_name, exc)
            else:
                tool_result_str = f"Unknown tool: {tool_name!r}"
                logger.warning("Unknown tool requested by LLM: %s", tool_name)

            messages.append(
                {
                    "role": "tool",
                    "content": tool_result_str,
                    "name": tool_name
                }
            )
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

    wrap_up = client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={"temperature": LLM_TEMPERATURE}
    )
    return wrap_up["message"].get("content", "")
