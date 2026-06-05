"""
llm_client.py
────────────────────────────────────────────────────────────────────────────
Unified LLM client factory for the ingestion pipeline.

Reads LLM_PROVIDER from the environment (default: ollama) and returns a
callable that has the same signature regardless of which backend is active:

    complete(system_prompt, user_prompt) -> str

Supported providers
───────────────────
  groq   — groq-cloud via the official `groq` SDK
             Env vars: GROQ_API_KEY, GROQ_MODEL
  ollama — local Ollama daemon
             Env vars: OLLAMA_HOST, OLLAMA_MODEL

Temperature is ALWAYS 0 — the ingestion pipeline requires fully deterministic
output for restructuring and JSON extraction.

Retry / error handling
──────────────────────
  - APITimeoutError / connection error → retry 3× with exponential backoff (2s, 4s, 8s)
  - All other API errors → logged at ERROR, re-raised immediately
  - Empty response → logged at ERROR, returned as "" so caller decides fallback

Usage
─────
    from ingestion.llm_client import get_llm_client

    llm = get_llm_client()
    output = llm.complete(system_prompt="You are ...", user_prompt="Do this: ...")
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ── Retry constants (Phase 14 error handling) ─────────────────────────────────
_MAX_RETRIES    = 3
_BACKOFF_SECS   = [2, 4, 8]
_TEMPERATURE    = 0          # MUST always be 0 — non-negotiable


# ── Abstract base ─────────────────────────────────────────────────────────────

class LLMClient(ABC):
    """Abstract LLM client with a single `complete` method."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a (system, user) message pair to the LLM and return the
        text response.

        Parameters
        ──────────
        system_prompt : str
            The system-level instruction for the model.
        user_prompt : str
            The user message / content to process.

        Returns
        ───────
        str
            The model's text output. May be empty string on failure.
        """


# ── Groq backend ──────────────────────────────────────────────────────────────

class _GroqClient(LLMClient):
    """
    Thin wrapper around the `groq` SDK.
    Reads GROQ_API_KEY and GROQ_MODEL from environment.
    """

    def __init__(self) -> None:
        try:
            from groq import Groq, APITimeoutError, APIError  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'groq' package is not installed. "
                "Run: pip install groq"
            ) from exc

        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Export it before running the pipeline."
            )

        self._client      = Groq(api_key=api_key)
        self._model       = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        self._APITimeout  = APITimeoutError
        self._APIError    = APIError
        logger.info(f"[llm_client] Groq client ready — model={self._model}")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        for attempt in range(_MAX_RETRIES):
            try:
                logger.debug(
                    f"[llm_client:groq] Calling model={self._model} "
                    f"(attempt {attempt + 1}/{_MAX_RETRIES})"
                )
                response = self._client.chat.completions.create(
                    model=self._model,
                    temperature=_TEMPERATURE,
                    max_tokens=32768,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                )
                text: str = response.choices[0].message.content or ""
                if not text.strip():
                    logger.error(
                        f"[llm_client:groq] Empty response from model={self._model}."
                    )
                return text

            except self._APITimeout:
                if attempt < _MAX_RETRIES - 1:
                    wait = _BACKOFF_SECS[attempt]
                    logger.warning(
                        f"[llm_client:groq] Timeout on attempt {attempt + 1}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        f"[llm_client:groq] All {_MAX_RETRIES} attempts timed out."
                    )
                    raise

            except self._APIError as exc:
                logger.error(f"[llm_client:groq] API error: {exc}")
                raise

        return ""  # unreachable, but satisfies type checkers


# ── Ollama backend ────────────────────────────────────────────────────────────

class _OllamaClient(LLMClient):
    """
    Thin wrapper around the `ollama` Python SDK.
    Reads OLLAMA_HOST and OLLAMA_MODEL from environment.
    """

    def __init__(self) -> None:
        try:
            from ollama import Client as OllamaSDKClient  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'ollama' package is not installed. "
                "Run: pip install ollama"
            ) from exc

        host       = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._model  = os.environ.get("OLLAMA_MODEL", "llama3")
        self._client = OllamaSDKClient(host=host)
        logger.info(
            f"[llm_client] Ollama client ready — host={host}, model={self._model}"
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        for attempt in range(_MAX_RETRIES):
            try:
                logger.debug(
                    f"[llm_client:ollama] Calling model={self._model} "
                    f"(attempt {attempt + 1}/{_MAX_RETRIES})"
                )
                response = self._client.chat(
                    model=self._model,
                    options={"temperature": _TEMPERATURE, "num_think": 0},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                )
                text: str = response["message"].get("content", "")
                if text.strip():
                    import re
                    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                if not text.strip():
                    logger.error(
                        f"[llm_client:ollama] Empty response from model={self._model}."
                    )
                return text

            except Exception as exc:
                # Ollama SDK raises generic exceptions on connection errors
                err_name = type(exc).__name__
                is_retryable = any(
                    kw in err_name.lower() or kw in str(exc).lower()
                    for kw in ("timeout", "connection", "connrefused")
                )
                if is_retryable and attempt < _MAX_RETRIES - 1:
                    wait = _BACKOFF_SECS[attempt]
                    logger.warning(
                        f"[llm_client:ollama] Retryable error on attempt "
                        f"{attempt + 1} ({err_name}). Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"[llm_client:ollama] Error: {exc}")
                    raise

        return ""


# ── Public factory ────────────────────────────────────────────────────────────

def get_llm_client() -> LLMClient:
    """
    Read LLM_PROVIDER from the environment and return the appropriate client.

    Values:
        "groq"   → GroqClient  (uses GROQ_API_KEY + GROQ_MODEL)
        "ollama" → OllamaClient (uses OLLAMA_HOST + OLLAMA_MODEL)

    Raises
    ──────
    ValueError
        If LLM_PROVIDER is set to an unrecognised value.
    EnvironmentError
        If required API keys are missing.
    """
    provider = os.environ.get("LLM_PROVIDER", "groq").strip().lower()

    if provider == "groq":
        return _GroqClient()
    elif provider == "ollama":
        return _OllamaClient()
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER='{provider}'. "
            "Set LLM_PROVIDER to 'groq' or 'ollama' in your .env file."
        )
