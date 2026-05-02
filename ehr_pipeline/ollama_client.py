"""Ollama HTTP client for the EHR pipeline.

The ``OllamaClient`` class talks directly to the Ollama REST API using only
the stdlib ``urllib`` — no third-party ``ollama`` package required.

Authentication logic:
- Cloud (OLLAMA_HOST=https://ollama.com): always attaches ``Authorization: Bearer``.
- Local daemon (http://localhost:…): header is suppressed unless
  ``OLLAMA_SEND_BEARER_TO_LOCAL=1`` is set, because the stock daemon returns
  401 when it receives an unexpected Bearer token.

Module-level ``chat_text`` / ``chat_json`` helpers maintain the same interface
the pipeline stages already use, so no stage file needed to change.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from typing import Any

from . import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _backoff_delay(attempt: int, base: float) -> float:
    """Exponential backoff with full jitter: U(0, base * 2^(attempt-1))."""
    cap = base * (2 ** (attempt - 1))
    return random.uniform(0.5, cap)


def _is_retryable_status(code: int) -> bool:
    """Retry transient errors; do not retry definitive client errors (4xx except 429)."""
    return code == 0 or code == 429 or code >= 500
    # 0   = network-level error (URLError / timeout)
    # 429 = rate-limited — back off and retry
    # 5xx = server-side transient error


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class OllamaError(Exception):
    """Raised when the Ollama API returns an HTTP error or non-JSON content."""

    def __init__(self, message: str, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code

    def __str__(self) -> str:
        prefix = f"[HTTP {self.status_code}] " if self.status_code else ""
        return f"{prefix}{super().__str__()}"


# ---------------------------------------------------------------------------
# Client class
# ---------------------------------------------------------------------------

class OllamaClient:
    """Stateless HTTP client for the Ollama ``/api/chat`` endpoint.

    Args:
        host: Base URL, e.g. ``https://ollama.com`` or ``http://localhost:11434``.
        api_key: Bearer token for Ollama Cloud. Pass ``None`` for local daemons.
        timeout: Per-request timeout in seconds (default 600 to allow large models).
    """

    def __init__(
        self,
        host: str,
        api_key: str | None = None,
        timeout: int = 600,
    ) -> None:
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post_once(self, payload: dict[str, Any]) -> str:
        """Single attempt: POST to /api/chat and return the assistant content."""
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
                return body.get("message", {}).get("content", "")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except Exception:
                pass
            raise OllamaError(f"Ollama API failed: {detail}", exc.code) from exc
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self.host}: {exc.reason}", status_code=0
            ) from exc

    def _post(self, payload: dict[str, Any]) -> str:
        """POST with automatic retry on transient network / 5xx errors."""
        max_retries = config.LLM_MAX_RETRIES
        base_delay = config.LLM_RETRY_BASE_DELAY
        last_exc: OllamaError | None = None

        for attempt in range(1, max_retries + 2):  # +1 for the initial try
            try:
                return self._post_once(payload)
            except OllamaError as exc:
                if not _is_retryable_status(exc.status_code):
                    raise  # 4xx — don't retry, it won't help
                last_exc = exc
                if attempt <= max_retries:
                    delay = _backoff_delay(attempt, base_delay)
                    log.warning(
                        "LLM request failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt, max_retries + 1, delay, exc,
                    )
                    time.sleep(delay)

        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def build_messages(
        user: str, system: str | None = None
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return messages

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        format: str | dict[str, Any] | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> str:
        """Raw chat call. Returns the assistant message as a string."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if extra_options:
            options.update(extra_options)
        if options:
            payload["options"] = options

        if format is not None:
            payload["format"] = format

        log.debug("chat model=%s format=%s", model, type(format).__name__)
        return self._post(payload)

    def chat_text(
        self,
        model: str,
        user: str,
        *,
        system: str | None = None,
        temperature: float = config.DEFAULT_TEMPERATURE,
        extra_options: dict[str, Any] | None = None,
    ) -> str:
        """Free-text chat; returns the raw assistant reply."""
        return self.chat(
            model,
            self.build_messages(user, system),
            temperature=temperature,
            extra_options=extra_options,
        )

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences that some models wrap around JSON.

        Handles variants like:
          ```json\\n{...}\\n```
          ```\\n{...}\\n```
          {... bare JSON ...}
        """
        stripped = text.strip()
        # Fast-path: already looks like raw JSON
        if stripped.startswith("{") or stripped.startswith("["):
            return stripped
        # Drop opening fence (```json or ``` or ~~~)
        import re
        stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
        stripped = re.sub(r"^~~~[a-zA-Z]*\s*", "", stripped)
        # Drop closing fence
        stripped = re.sub(r"\s*```\s*$", "", stripped)
        stripped = re.sub(r"\s*~~~\s*$", "", stripped)
        return stripped.strip()

    def _parse_json(self, raw: str, model: str) -> Any:
        """Try to extract a valid JSON object/array from a raw LLM response."""
        cleaned = self._strip_fences(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # Last resort: scan for the first { or [ and try from there
        for start_char, end_char in (("{", "}"), ("[", "]")):
            idx = cleaned.find(start_char)
            if idx != -1:
                fragment = cleaned[idx:]
                ridx = fragment.rfind(end_char)
                if ridx != -1:
                    try:
                        return json.loads(fragment[: ridx + 1])
                    except json.JSONDecodeError:
                        pass
        log.error(
            "Model %s returned non-JSON even after fence stripping: %.500s", model, raw
        )
        raise OllamaError(f"Model {model} did not return valid JSON: {raw[:200]}")

    def chat_json(
        self,
        model: str,
        user: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        extra_options: dict[str, Any] | None = None,
    ) -> Any:
        """JSON-mode chat with retry on bad JSON; returns parsed Python object.

        Passes ``schema`` as Ollama's structured-output ``format`` when given,
        otherwise falls back to ``format="json"`` for loose JSON mode.
        Automatically strips markdown code fences that some models add.
        Retries up to ``config.LLM_MAX_RETRIES`` times when the model returns
        non-JSON or a transient network error occurs.
        """
        fmt: Any = schema if schema is not None else "json"
        messages = self.build_messages(user, system)
        max_retries = config.LLM_MAX_RETRIES
        base_delay = config.LLM_RETRY_BASE_DELAY
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 2):
            try:
                raw = self.chat(
                    model, messages,
                    temperature=temperature,
                    format=fmt,
                    extra_options=extra_options,
                )
                return self._parse_json(raw, model)
            except OllamaError as exc:
                last_exc = exc
                # 4xx errors won't be fixed by retrying
                if not _is_retryable_status(exc.status_code):
                    raise
            if attempt <= max_retries:
                delay = _backoff_delay(attempt, base_delay)
                log.warning(
                    "LLM JSON request failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt, max_retries + 1, delay, last_exc,
                )
                time.sleep(delay)

        raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Module-level singleton and convenience functions
# (stages call these directly without constructing a client)
# ---------------------------------------------------------------------------

_client: OllamaClient | None = None
_client_fingerprint: tuple[str, str | None] | None = None


def _use_bearer() -> bool:
    """True when the Bearer header should be sent to the configured host."""
    host = config.OLLAMA_HOST.rstrip("/")
    is_cloud = host == config.CLOUD_HOST.rstrip("/")
    force_local = os.getenv("OLLAMA_SEND_BEARER_TO_LOCAL", "").strip().lower() in (
        "1", "true", "yes"
    )
    return is_cloud or force_local


def get_client() -> OllamaClient:
    """Return a cached ``OllamaClient`` for the current host + auth configuration."""
    global _client, _client_fingerprint

    host = config.OLLAMA_HOST.rstrip("/")
    api_key = config.OLLAMA_API_KEY if _use_bearer() else None
    fingerprint = (host, api_key)

    if _client is not None and _client_fingerprint == fingerprint:
        return _client

    if _use_bearer() and not config.OLLAMA_API_KEY:
        raise RuntimeError(
            "Bearer auth required but OLLAMA_API_KEY is not set. "
            "For Ollama Cloud set OLLAMA_HOST=https://ollama.com and add the key; "
            "for local Ollama unset OLLAMA_API_KEY or set OLLAMA_SEND_BEARER_TO_LOCAL=0."
        )

    log.debug("Building OllamaClient for host=%s bearer=%s", host, api_key is not None)
    _client = OllamaClient(host=host, api_key=api_key, timeout=config.REQUEST_TIMEOUT_SECONDS)
    _client_fingerprint = fingerprint
    return _client


def chat_text(
    *,
    model: str,
    user: str,
    system: str | None = None,
    temperature: float = config.DEFAULT_TEMPERATURE,
    extra_options: dict[str, Any] | None = None,
) -> str:
    """Module-level convenience wrapper for ``OllamaClient.chat_text``."""
    return get_client().chat_text(
        model, user,
        system=system,
        temperature=temperature,
        extra_options=extra_options,
    )


def chat_json(
    *,
    model: str,
    user: str,
    system: str | None = None,
    schema: dict[str, Any] | None = None,
    temperature: float = 0.0,
    extra_options: dict[str, Any] | None = None,
) -> Any:
    """Module-level convenience wrapper for ``OllamaClient.chat_json``."""
    return get_client().chat_json(
        model, user,
        system=system,
        schema=schema,
        temperature=temperature,
        extra_options=extra_options,
    )
