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

import os

from src.core.config import OLLAMA_MODEL, LLM_TEMPERATURE
from src.tools.rag_tools import (
    chunk_read,
    get_document_chunks,
    keyword_search,
    list_documents,
    reset_c_read,
    semantic_search,
    triple_lookup,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT: str = """You are an expert research assistant with access to a document corpus through retrieval tools.

Your goal is to answer questions accurately using ONLY retrieved document evidence.

### Retrieval Strategy

1. Analyze the user's question and identify all required information.
2. For simple factual questions, retrieve only the necessary evidence.
3. For multi-hop questions, decompose the query into sub-questions and retrieve evidence for each.
4. Connect related facts across chunks when needed.
5. Continue retrieval only until all required evidence is found.

### Tool Usage

* Use keyword search for IDs, names, contract numbers, clauses, and exact terms.
* Use semantic search for concepts, obligations, requirements, summaries, and relationships.
* Read full chunks only when snippets are incomplete or additional context is required.
* Prefer parallel searches when multiple independent facts are needed.

### Reasoning Rules

* Synthesize information; do not simply extract text.
* Combine evidence from multiple chunks when necessary.
* Follow references, entities, and clauses that lead to additional required information.
* Answer only the question asked.

### Grounding Rules

* Use only retrieved information.
* Never invent facts or fill gaps with assumptions.
* If information is unavailable, explicitly state that it was not found in the retrieved documents.
* Pay close attention to exact entity names, IDs, dates, quantities, and clause references.

### Efficiency Rules

* Minimize tool calls.
* Stop searching as soon as sufficient evidence is gathered.
* Do not retrieve redundant information.
* Do not read chunks unnecessarily.
* IF a specific document name is provided in the prompt context (e.g., "[Context: focus on documents: ...]"), use `get_document_chunks` with the `doc_name` parameter to retrieve its chunks directly.

### Output Format

Reasoning:
Brief explanation of how the answer was derived.

Sources:
List of supporting chunk IDs.

Final Answer:
Clear, concise, evidence-based answer.

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
}

OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_document_chunks",
            "description": "Retrieve the first N chunk IDs and their snippets for a specific document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "integer", "description": "The integer doc_id"},
                    "doc_name": {"type": "string", "description": "The exact document name (e.g. 'Contract - GEMC-1.pdf')"},
                    "max_chunks": {"type": "integer", "description": "Maximum number of chunks to retrieve"}
                },
                "required": []
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
    }
]

# ─────────────────────────────────────────────────────────────────────────────
# LLM Factory
# ─────────────────────────────────────────────────────────────────────────────

def _unified_chat(
    messages: list[dict],
    tools: list[dict] | None = None,
) -> dict:
    """Unified chat completion interface supporting both Groq and Ollama."""
    provider = os.environ.get("LLM_PROVIDER", "groq").strip().lower()

    if provider == "groq":
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        client = Groq(api_key=api_key)
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        # Groq strict validation: ensure tool messages have string content
        safe_messages = []
        for m in messages:
            safe_m = dict(m)
            if safe_m.get("role") == "tool" and not isinstance(safe_m.get("content"), str):
                safe_m["content"] = str(safe_m.get("content"))
            safe_messages.append(safe_m)

        response = client.chat.completions.create(
            model=model,
            messages=safe_messages,
            tools=tools if tools else None,
            temperature=LLM_TEMPERATURE,
        )
        
        msg = response.choices[0].message
        result_msg = {"role": msg.role, "content": msg.content or ""}
        
        if msg.tool_calls:
            result_msg["tool_calls"] = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                result_msg["tool_calls"].append({
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": args
                    }
                })
        return result_msg

    elif provider == "ollama":
        from ollama import Client
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        client = Client(host=host)
        model = OLLAMA_MODEL
        response = client.chat(
            model=model,
            messages=messages,
            tools=tools if tools else None,
            options={"raw": True, "temperature": LLM_TEMPERATURE, "num_think": 0}
        )
        return response["message"]
    
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def _unified_chat_stream(
    messages: list[dict]
):
    """Unified streaming chat interface supporting both Groq and Ollama."""
    provider = os.environ.get("LLM_PROVIDER", "groq").strip().lower()

    if provider == "groq":
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        client = Groq(api_key=api_key)
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        safe_messages = []
        for m in messages:
            safe_m = dict(m)
            if safe_m.get("role") == "tool" and not isinstance(safe_m.get("content"), str):
                safe_m["content"] = str(safe_m.get("content"))
            safe_messages.append(safe_m)

        response = client.chat.completions.create(
            model=model,
            messages=safe_messages,
            temperature=LLM_TEMPERATURE,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    elif provider == "ollama":
        from ollama import Client
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        client = Client(host=host)
        model = OLLAMA_MODEL
        
        response = client.chat(
            model=model,
            messages=messages,
            options={"temperature": LLM_TEMPERATURE, "num_think": 0},
            stream=True
        )
        
        for chunk in response:
            if chunk.get("message", {}).get("content"):
                yield chunk["message"]["content"]
    
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

def _unified_chat_stream_tools(
    messages: list[dict],
    tools: list[dict] | None = None,
):
    """
    Unified chat streaming that also supports tools.
    Yields tuples of ("content", text) or ("done", reply_msg_dict)
    """
    provider = os.environ.get("LLM_PROVIDER", "groq").strip().lower()

    if provider == "groq":
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        client = Groq(api_key=api_key)
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        safe_messages = []
        for m in messages:
            safe_m = dict(m)
            if safe_m.get("role") == "tool" and not isinstance(safe_m.get("content"), str):
                safe_m["content"] = str(safe_m.get("content"))
            safe_messages.append(safe_m)

        response = client.chat.completions.create(
            model=model,
            messages=safe_messages,
            tools=tools if tools else None,
            temperature=LLM_TEMPERATURE,
            stream=True
        )
        
        tool_calls_dict = {}
        full_content = ""
        
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                full_content += delta.content
                yield ("content", delta.content)
            
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_dict:
                        tool_calls_dict[idx] = {
                            "id": tc.id,
                            "function": {"name": tc.function.name or "", "arguments": tc.function.arguments or ""}
                        }
                    else:
                        if tc.function.arguments:
                            tool_calls_dict[idx]["function"]["arguments"] += tc.function.arguments

        final_tool_calls = []
        for idx in sorted(tool_calls_dict.keys()):
            tc = tool_calls_dict[idx]
            try:
                tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
            except Exception:
                tc["function"]["arguments"] = {}
            final_tool_calls.append(tc)
                
        reply_msg = {"role": "assistant", "content": full_content}
        if final_tool_calls:
            reply_msg["tool_calls"] = final_tool_calls
        
        yield ("done", reply_msg)

    elif provider == "ollama":
        from ollama import Client
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        client = Client(host=host)
        model = OLLAMA_MODEL
        
        response = client.chat(
            model=model,
            messages=messages,
            tools=tools if tools else None,
            options={"temperature": LLM_TEMPERATURE, "num_think": 0},
            stream=True
        )
        
        full_content = ""
        final_tool_calls = []
        
        for chunk in response:
            msg = chunk.get("message", {})
            if msg.get("content"):
                full_content += msg["content"]
                yield ("content", msg["content"])
            if msg.get("tool_calls"):
                final_tool_calls = msg["tool_calls"]
        
        reply_msg = {"role": "assistant", "content": full_content}
        if final_tool_calls:
            reply_msg["tool_calls"] = final_tool_calls
            
        yield ("done", reply_msg)
    
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

# ─────────────────────────────────────────────────────────────────────────────
# ReAct Agent Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(
    query: str,
    max_iterations: int = 10,
) -> str:
    """Run the ReAct agent loop using unified LLM client."""

    reset_c_read()

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    logger.info("=== Agent loop started: query='%s' ===", query)

    for iteration in range(max_iterations):
        logger.debug("Iteration %d/%d — invoking LLM ...", iteration + 1, max_iterations)

        reply_msg = _unified_chat(
            messages=messages,
            tools=OLLAMA_TOOLS
        )
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

                    # Increased truncation limit to allow for more context in multi-hop reasoning
                    if len(tool_result_str) > 8000:
                        tool_result_str = tool_result_str[:8000] + "\n... [CONTENT TRUNCATED TO SAVE TOKENS. NOT ALL DATA SHOWN.] ..."
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
                    "name": tool_name,
                    "tool_call_id": tool_call.get("id", f"call_{tool_name}")
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

    wrap_up = _unified_chat(
        messages=messages,
        stream=False
    )
    return wrap_up.get("content", "")


# ─────────────────────────────────────────────────────────────────────────────
# Streaming ReAct Agent
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_stream(
    query: str,
    max_iterations: int = 10,
):
    """Streaming variant of run_agent.

    Runs the full ReAct tool-calling loop synchronously, then streams the
    final answer back to the caller token-by-token as a Python generator.

    Yields:
        str: Individual token text chunks from the LLM as they are generated.

    Usage (server.py):
        for token in run_agent_stream(query, max_iterations):
            yield f"data: {json.dumps(token)}\\n\\n"
    """
    reset_c_read()

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    logger.info("=== Streaming agent loop started: query='%s' ===", query)

    # ── Phase 1: Tool-calling loop ──────────
    for iteration in range(max_iterations):
        logger.debug("Stream iteration %d/%d", iteration + 1, max_iterations)

        reply_msg = None
        for chunk_type, data in _unified_chat_stream_tools(messages, tools=OLLAMA_TOOLS):
            if chunk_type == "content":
                yield data
            elif chunk_type == "done":
                reply_msg = data

        messages.append(reply_msg)

        # If no tool calls, the LLM has produced its final answer
        if not reply_msg.get("tool_calls"):
            logger.info("Streaming agent: final answer received (len=%d).", len(reply_msg.get("content", "")))
            return

        # ── Execute tool calls ────────────────────────────────────────────────
        for tool_call in reply_msg["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"]["arguments"]
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

            messages.append({
                "role": "tool", 
                "content": tool_result_str, 
                "name": tool_name,
                "tool_call_id": tool_call.get("id", f"call_{tool_name}")
            })

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

    # Final wrap-up — this one we DO want to stream fresh
    for token_text in _unified_chat_stream(messages=messages):
        if token_text:
            yield token_text

