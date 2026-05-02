#!/usr/bin/env python3
"""
Run inference against a local Ollama server or Ollama Cloud.

The inference logic lives in ``ehr_pipeline.ollama_client.OllamaClient``.
This script is a thin CLI wrapper around it.

Examples:
  python infer_ollama.py --local --model llama3.2 "Write a haiku about GPUs"
  python infer_ollama.py --cloud --model gemma4:31b "Explain attention in one paragraph"
  OLLAMA_HOST=https://ollama.com OLLAMA_API_KEY=... python infer_ollama.py --model deepseek-v4-pro "Hello"
  cat note.txt | python infer_ollama.py --model qwen3.5:397b --system "Summarize briefly"

Environment:
  OLLAMA_HOST       Defaults to http://localhost:11434, or https://ollama.com with --cloud.
  OLLAMA_MODEL      Default model if --model is omitted.
  OLLAMA_API_KEY    Required for Ollama Cloud.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

LOCAL_HOST = "http://localhost:11434"
CLOUD_HOST = "https://ollama.com"


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader — no third-party dependencies."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _get_client(host: str, api_key: str | None):
    """Import OllamaClient from the package; fall back to inline urllib impl."""
    # Add project root to sys.path so the package is importable when running
    # this script directly (python infer_ollama.py …).
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from ehr_pipeline.ollama_client import OllamaClient  # noqa: PLC0415
        return OllamaClient(host=host, api_key=api_key)
    except ImportError:
        # Inline fallback so the script stays usable without the ehr_pipeline package.
        import json
        import urllib.error
        import urllib.request

        class _FallbackClient:
            def __init__(self, host, api_key):
                self._host = host.rstrip("/")
                self._api_key = api_key

            def chat_text(self, model, user, *, system=None, temperature=0.1, extra_options=None):
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": user})
                payload = {"model": model, "messages": messages, "stream": False,
                           "options": {"temperature": temperature}}
                headers = {"Content-Type": "application/json"}
                if self._api_key:
                    headers["Authorization"] = f"Bearer {self._api_key}"
                req = urllib.request.Request(
                    f"{self._host}/api/chat",
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers, method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=600) as resp:
                        body = json.loads(resp.read().decode("utf-8"))
                        return body.get("message", {}).get("content", "")
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode("utf-8", errors="replace")
                    raise SystemExit(f"Ollama returned HTTP {exc.code}: {detail}") from exc
                except urllib.error.URLError as exc:
                    raise SystemExit(f"Could not reach Ollama at {self._host}: {exc.reason}") from exc

        return _FallbackClient(host, api_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call an Ollama model locally or through Ollama Cloud."
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--local", action="store_true",
                        help=f"Use local Ollama at {LOCAL_HOST}.")
    target.add_argument("--cloud", action="store_true",
                        help=f"Use Ollama Cloud at {CLOUD_HOST}. Requires OLLAMA_API_KEY.")
    target.add_argument("--host",
                        help="Custom Ollama host, e.g. http://localhost:11434 or https://ollama.com.")

    parser.add_argument("prompt", nargs="?", help="Prompt to send to the model.")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL"), help="Model name.")
    parser.add_argument("--system", help="Optional system prompt.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    return parser.parse_args()


def main() -> None:
    _load_dotenv()
    args = parse_args()

    prompt = args.prompt or sys.stdin.read().strip()
    if not prompt:
        raise SystemExit("Pass a prompt argument or pipe one through stdin.")
    if not args.model:
        raise SystemExit("Set --model or the OLLAMA_MODEL environment variable.")

    if args.cloud:
        host = CLOUD_HOST
    elif args.local:
        host = LOCAL_HOST
    elif args.host:
        host = args.host
    else:
        host = os.getenv("OLLAMA_HOST", LOCAL_HOST)

    api_key = os.getenv("OLLAMA_API_KEY")
    if host.rstrip("/") == CLOUD_HOST.rstrip("/") and not api_key:
        raise SystemExit("Ollama Cloud requires OLLAMA_API_KEY to be set.")

    client = _get_client(host, api_key)
    result = client.chat_text(
        args.model, prompt,
        system=args.system,
        temperature=args.temperature,
    )
    print(result)


if __name__ == "__main__":
    main()
