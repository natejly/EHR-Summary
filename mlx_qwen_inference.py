"""Terminal chat interface and reusable MLX-LM wrapper for Qwen3.5-9B 4-bit.

Install:
    pip install -U mlx-lm

Default model:
    mlx-community/Qwen3.5-9B-4bit
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Sequence


Message = Mapping[str, str]


@dataclass(slots=True)
class MlxQwenConfig:
    model_path: str = "mlx-community/Qwen3.5-9B-4bit"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    min_p: float = 0.0
    trust_remote_code: bool = False
    enable_thinking: bool = False
    verbose: bool = False


@dataclass(slots=True, frozen=True)
class CompletionResult:
    """Generated text plus the *reason* generation stopped.

    finish_reason values:
        "stop_string"  - one of the caller's stop strings was emitted; the
                         output is structurally complete from our point of view.
        "eos"          - the model emitted an EOS token; complete.
        "length"       - the max_tokens cap was hit before the model finished
                         (mlx-lm's "length"); the output is likely truncated.
        None           - reason couldn't be determined (older mlx-lm or no
                         streaming path); fall back to text-based heuristics.
    """

    text: str
    finish_reason: str | None
    generation_tokens: int


class Qwen35MlxInference:
    """Small wrapper around `mlx-lm` for single-turn and chat inference."""

    def __init__(self, config: MlxQwenConfig | None = None) -> None:
        self.config = config or MlxQwenConfig()
        self.model = None
        self.tokenizer = None

    def load_model(self, reload: bool = False) -> None:
        if self.model is not None and self.tokenizer is not None and not reload:
            return

        _, load_fn, _, _ = _import_mlx_lm()
        load_kwargs = {}
        if self.config.trust_remote_code:
            load_kwargs["tokenizer_config"] = {"trust_remote_code": True}

        self.model, self.tokenizer = load_fn(self.config.model_path, **load_kwargs)

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        history: Sequence[Message] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        stop_strings: Sequence[str] | None = None,
        verbose: bool | None = None,
    ) -> str:
        return self.complete_with_meta(
            prompt,
            system_prompt=system_prompt,
            history=history,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            stop_strings=stop_strings,
            verbose=verbose,
        ).text

    def complete_with_meta(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        history: Sequence[Message] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        stop_strings: Sequence[str] | None = None,
        verbose: bool | None = None,
    ) -> CompletionResult:
        """Same as `complete` but also returns *why* generation stopped.

        Use this when the caller needs to distinguish a clean stop ("stop"
        / "stop_string" / "eos") from a `max_tokens` truncation ("length").
        """

        self._ensure_loaded()
        formatted_prompt = self._format_single_turn_prompt(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
        )
        sampler = self._build_sampler(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
        return self._generate_formatted_prompt_with_meta(
            formatted_prompt,
            sampler=sampler,
            max_tokens=max_tokens,
            stop_strings=stop_strings,
            verbose=verbose,
        )

    def chat(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        stop_strings: Sequence[str] | None = None,
        verbose: bool | None = None,
    ) -> str:
        self._ensure_loaded()
        formatted_prompt = self._format_messages(messages)
        sampler = self._build_sampler(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
        return self._generate_formatted_prompt(
            formatted_prompt,
            sampler=sampler,
            max_tokens=max_tokens,
            stop_strings=stop_strings,
            verbose=verbose,
        )

    def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        stop_strings: Sequence[str] | None = None,
    ) -> Iterator[str]:
        self._ensure_loaded()
        formatted_prompt = self._format_messages(messages)
        sampler = self._build_sampler(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
        yield from self._stream_formatted_prompt(
            formatted_prompt,
            sampler=sampler,
            max_tokens=max_tokens,
            stop_strings=stop_strings,
        )

    def stream(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        history: Sequence[Message] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        stop_strings: Sequence[str] | None = None,
    ) -> Iterator[str]:
        self._ensure_loaded()
        messages = self._build_messages(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
        )
        yield from self.stream_chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            stop_strings=stop_strings,
        )

    def run_terminal_chat(
        self,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        render_markdown: bool = False,
    ) -> None:
        self._ensure_loaded()
        if render_markdown:
            ensure_terminal_markdown_support()
        active_system_prompt = system_prompt
        messages = self._initial_messages(active_system_prompt)

        print(f"Loaded model: {self.config.model_path}")
        print("Commands: /help, /clear, /system <prompt>, /exit")

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break

            if not user_input:
                continue

            if user_input in {"/exit", "/quit"}:
                print("Exiting chat.")
                break

            if user_input == "/help":
                print("Commands:")
                print("  /help              Show this help message")
                print("  /clear             Clear conversation history")
                print("  /system <prompt>   Set or replace the system prompt and reset chat")
                print("  /system            Clear the current system prompt and reset chat")
                print("  /exit              Quit the chat session")
                continue

            if user_input == "/clear":
                messages = self._initial_messages(active_system_prompt)
                print("Conversation cleared.")
                continue

            if user_input.startswith("/system"):
                new_system_prompt = user_input[len("/system") :].strip() or None
                active_system_prompt = new_system_prompt
                messages = self._initial_messages(active_system_prompt)
                if active_system_prompt:
                    print("System prompt updated. Conversation reset.")
                else:
                    print("System prompt cleared. Conversation reset.")
                continue

            messages.append({"role": "user", "content": user_input})
            chunks: list[str] = []

            if render_markdown:
                print("Assistant:")
            else:
                print("Assistant: ", end="", flush=True)
            try:
                for chunk in self.stream_chat(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                ):
                    chunks.append(chunk)
                    if not render_markdown:
                        print(chunk, end="", flush=True)
            except KeyboardInterrupt:
                print("\nGeneration interrupted." if not render_markdown else "Generation interrupted.")

            assistant_response = "".join(chunks).strip()
            assistant_response = self._clean_response_text(assistant_response)
            if render_markdown:
                if assistant_response:
                    print_terminal_output(assistant_response, render_markdown=True)
                else:
                    print()
            else:
                print()

            if assistant_response:
                messages.append({"role": "assistant", "content": assistant_response})
            else:
                messages.pop()

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None

    def _ensure_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            self.load_model()

    def _resolve_max_tokens(self, max_tokens: int | None) -> int:
        return self.config.max_tokens if max_tokens is None else max_tokens

    def _generate_formatted_prompt(
        self,
        formatted_prompt: str,
        *,
        sampler,
        max_tokens: int | None,
        stop_strings: Sequence[str] | None,
        verbose: bool | None,
    ) -> str:
        return self._generate_formatted_prompt_with_meta(
            formatted_prompt,
            sampler=sampler,
            max_tokens=max_tokens,
            stop_strings=stop_strings,
            verbose=verbose,
        ).text

    def _generate_formatted_prompt_with_meta(
        self,
        formatted_prompt: str,
        *,
        sampler,
        max_tokens: int | None,
        stop_strings: Sequence[str] | None,
        verbose: bool | None,
    ) -> CompletionResult:
        # Streaming path is required to know whether one of OUR stop strings
        # was hit; mlx-lm's non-streaming `generate` only knows about EOS /
        # max_tokens. Use the streaming generator unconditionally and read
        # both the model's `finish_reason` and our stop-string flag from it.
        verbose_output = self.config.verbose if verbose is None else verbose
        chunks: list[str] = []
        last_meta: dict[str, object] = {}

        for chunk, meta in self._stream_formatted_prompt_with_meta(
            formatted_prompt,
            sampler=sampler,
            max_tokens=max_tokens,
            stop_strings=stop_strings,
        ):
            if chunk and verbose_output:
                print(chunk, end="", flush=True)
            if chunk:
                chunks.append(chunk)
            if meta:
                last_meta = meta
        if verbose_output:
            print()

        cleaned = self._clean_response_text("".join(chunks))
        return CompletionResult(
            text=cleaned,
            finish_reason=str(last_meta.get("finish_reason"))
                if last_meta.get("finish_reason") is not None
                else None,
            generation_tokens=int(last_meta.get("generation_tokens", 0) or 0),
        )

    def _stream_formatted_prompt(
        self,
        formatted_prompt: str,
        *,
        sampler,
        max_tokens: int | None,
        stop_strings: Sequence[str] | None,
    ) -> Iterator[str]:
        for chunk, _meta in self._stream_formatted_prompt_with_meta(
            formatted_prompt,
            sampler=sampler,
            max_tokens=max_tokens,
            stop_strings=stop_strings,
        ):
            if chunk:
                yield chunk

    def _stream_formatted_prompt_with_meta(
        self,
        formatted_prompt: str,
        *,
        sampler,
        max_tokens: int | None,
        stop_strings: Sequence[str] | None,
    ) -> Iterator[tuple[str, dict[str, object]]]:
        """Stream chunks and final-response metadata.

        Yields `(text_chunk, meta)` tuples. `meta` is empty for intermediate
        chunks and populated only on the final tuple. The final tuple's text
        is "" so callers can simply append non-empty chunks and read meta
        from whichever tuple has it.

        meta keys: `finish_reason` ("stop" | "length" | "stop_string"),
        `generation_tokens` (int).
        """

        _, _, stream_generate_fn, _ = _import_mlx_lm()

        responses = stream_generate_fn(
            self.model,
            self.tokenizer,
            formatted_prompt,
            max_tokens=self._resolve_max_tokens(max_tokens),
            sampler=sampler,
        )

        active_stops = [s for s in (stop_strings or []) if s]
        holdback = (max(len(s) for s in active_stops) - 1) if active_stops else 0
        buffer = ""
        last_finish_reason: str | None = None
        last_generation_tokens: int = 0
        stop_string_hit = False

        for response in responses:
            last_finish_reason = getattr(response, "finish_reason", None)
            last_generation_tokens = int(
                getattr(response, "generation_tokens", last_generation_tokens) or 0
            )
            chunk = getattr(response, "text", "") or ""
            if not active_stops:
                if chunk:
                    yield (chunk, {})
                continue

            buffer += chunk
            stop_index = self._find_first_stop_index(buffer, active_stops)
            if stop_index is not None:
                if stop_index > 0:
                    yield (buffer[:stop_index], {})
                buffer = ""
                stop_string_hit = True
                break

            if holdback == 0:
                if buffer:
                    yield (buffer, {})
                buffer = ""
                continue

            flush_upto = len(buffer) - holdback
            if flush_upto > 0:
                yield (buffer[:flush_upto], {})
                buffer = buffer[flush_upto:]

        if buffer:
            yield (buffer, {})

        # Resolve the final reason: stop-string > model EOS / length signal.
        if stop_string_hit:
            final_reason: str | None = "stop_string"
        else:
            final_reason = last_finish_reason
        yield (
            "",
            {
                "finish_reason": final_reason,
                "generation_tokens": last_generation_tokens,
            },
        )

    def _build_sampler(
        self,
        *,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        min_p: float | None,
    ):
        _, _, _, make_sampler_fn = _import_mlx_lm()
        return make_sampler_fn(
            temperature if temperature is not None else self.config.temperature,
            top_p=self.config.top_p if top_p is None else top_p,
            top_k=self.config.top_k if top_k is None else top_k,
            min_p=self.config.min_p if min_p is None else min_p,
        )

    def _format_single_turn_prompt(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
        history: Sequence[Message] | None,
    ) -> str:
        return self._format_messages(
            self._build_messages(
                prompt=prompt,
                system_prompt=system_prompt,
                history=history,
            )
        )

    def _build_messages(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
        history: Sequence[Message] | None,
    ) -> list[dict[str, str]]:
        messages = self._initial_messages(system_prompt)
        if history:
            messages.extend(self._normalize_messages(history))
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _initial_messages(system_prompt: str | None) -> list[dict[str, str]]:
        if not system_prompt:
            return []
        return [{"role": "system", "content": system_prompt}]

    def _format_messages(self, messages: Sequence[Message]) -> str:
        normalized_messages = self._normalize_messages(messages)
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.config.enable_thinking,
            )

        return "\n".join(
            f"{message['role'].upper()}: {message['content']}"
            for message in normalized_messages
        )

    @staticmethod
    def _normalize_messages(messages: Sequence[Message]) -> list[dict[str, str]]:
        normalized = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if not role or content is None:
                raise ValueError(
                    "Each message must include string values for 'role' and 'content'."
                )
            normalized.append({"role": str(role), "content": str(content)})
        return normalized

    @staticmethod
    def _clean_response_text(text: str) -> str:
        text = text.strip()
        if text.startswith("<think>"):
            text = re.sub(r"^<think>\s*.*?\s*</think>\s*", "", text, flags=re.DOTALL)
        return text.strip()

    @staticmethod
    def _stop_stream_at_strings(
        chunks: Iterable[str],
        stop_strings: Sequence[str] | None,
    ) -> Iterator[str]:
        active_stops = [stop for stop in (stop_strings or []) if stop]
        if not active_stops:
            yield from chunks
            return

        holdback = max(max(len(stop) for stop in active_stops) - 1, 0)
        buffer = ""

        for chunk in chunks:
            if not chunk:
                continue
            buffer += chunk
            stop_index = Qwen35MlxInference._find_first_stop_index(buffer, active_stops)
            if stop_index is not None:
                if stop_index > 0:
                    yield buffer[:stop_index]
                return

            if holdback == 0:
                yield buffer
                buffer = ""
                continue

            flush_upto = len(buffer) - holdback
            if flush_upto > 0:
                yield buffer[:flush_upto]
                buffer = buffer[flush_upto:]

        if buffer:
            yield buffer

    @staticmethod
    def _find_first_stop_index(text: str, stop_strings: Sequence[str]) -> int | None:
        indexes = [text.find(stop) for stop in stop_strings]
        valid_indexes = [index for index in indexes if index >= 0]
        if not valid_indexes:
            return None
        return min(valid_indexes)


def _import_mlx_lm():
    try:
        from mlx_lm import generate, load, stream_generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise RuntimeError(
            "mlx-lm is not installed. Install it with `pip install -r requirements.txt`."
        ) from exc
    return generate, load, stream_generate, make_sampler


def _import_rich_markdown():
    try:
        from rich.console import Console
        from rich.markdown import Markdown
    except ImportError as exc:
        raise RuntimeError(
            "Markdown rendering requires `rich`. Install it with `pip install -r requirements.txt`."
        ) from exc
    return Console, Markdown


def ensure_terminal_markdown_support() -> None:
    _import_rich_markdown()


def print_terminal_output(text: str, *, render_markdown: bool = False) -> None:
    if not render_markdown:
        print(text)
        return

    console_cls, markdown_cls = _import_rich_markdown()
    console_cls().print(markdown_cls(text))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with Qwen3.5-9B 4-bit in your terminal.")
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3.5-9B-4bit",
        help="Model repo or local path to an MLX-compatible model.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt for the session.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Run a single prompt instead of the interactive terminal chat.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate per assistant turn.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling value.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling value. Use 0 to disable.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Min-p sampling value. Use 0 to disable.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow models that require trusted remote tokenizer code.",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Qwen3.5 thinking mode and show reasoning output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose generation output for non-streamed calls.",
    )
    parser.add_argument(
        "--render-markdown",
        action="store_true",
        help="Render assistant output as markdown in the terminal after each response completes.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.render_markdown:
        ensure_terminal_markdown_support()
    client = Qwen35MlxInference(
        MlxQwenConfig(
            model_path=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            trust_remote_code=args.trust_remote_code,
            enable_thinking=args.thinking,
            verbose=args.verbose,
        )
    )

    if args.prompt:
        reply = client.complete(args.prompt, system_prompt=args.system_prompt)
        print_terminal_output(reply, render_markdown=args.render_markdown)
        return

    client.run_terminal_chat(
        system_prompt=args.system_prompt,
        render_markdown=args.render_markdown,
    )


if __name__ == "__main__":
    main()
