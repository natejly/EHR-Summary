"""Clinical note summarization helpers built on top of local MLX Qwen inference."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Sequence


_LOG = logging.getLogger(__name__)

from mlx_qwen_inference import (
    CompletionResult,
    MlxQwenConfig,
    Qwen35MlxInference,
    ensure_terminal_markdown_support,
    print_terminal_output,
)
from prompts import (
    CLINICAL_SUMMARY_CLAIM_DECOMPOSITION_END_MARKER,
    CLINICAL_SUMMARY_END_MARKER,
    CLINICAL_SUMMARY_FACT_CHECK_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    build_clinical_summary_claim_decomposition_prompt,
    build_clinical_summary_prompt as build_clinical_summary_prompt_from_blocks,
    build_clinical_summary_revision_prompt,
)
from verification import (
    ClaimVerificationResult,
    ClaimVerifier,
    LLMClaimVerifier,
    VerificationPassResult,
    fallback_claim_split,
    parse_atomic_claims,
    score_claims_via_adapter,
)


DEFAULT_MODEL_PATH = "mlx-community/Qwen3.5-9B-4bit"
DEFAULT_DIRECTORY_SUFFIXES = (".txt", ".md")


@dataclass(slots=True)
class ClinicalNote:
    """A single clinical note or document passed to the summarizer."""

    content: str
    source: str | None = None
    note_type: str | None = None
    note_id: str | None = None

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        note_type: str | None = None,
        encoding: str = "utf-8",
    ) -> ClinicalNote:
        resolved_path = Path(path)
        return cls(
            content=resolved_path.read_text(encoding=encoding),
            source=str(resolved_path),
            note_type=note_type or infer_note_type(resolved_path),
            note_id=resolved_path.stem,
        )

    def render(self, index: int) -> str:
        metadata = [f"[Note {index}]"]
        if self.note_id:
            metadata.append(f"Note ID: {self.note_id}")
        if self.note_type:
            metadata.append(f"Note Type: {self.note_type}")
        if self.source:
            metadata.append(f"Source: {self.source}")
        metadata.append("Content:")
        metadata.append(self.content.strip())
        return "\n".join(metadata)


# Stop strings: anything that signals the model is done emitting the structured
# summary or is starting an unwanted postscript / disclaimer / chain-of-thought tail.
DEFAULT_CLINICAL_STOP_STRINGS: tuple[str, ...] = (
    CLINICAL_SUMMARY_END_MARKER,
    "\n\nNote:",
    "\n\nDisclaimer:",
    "\n\nImportant:",
    "\n\nPlease note",
    "\n\nThis summary",
    "\n\nIn summary",
    "\n\nLet me know",
    "\n\nIs there anything else",
    "\n\n---",
)


# Heuristic first-pass compression targets keyed on input token count. Each
# tier is (max_input_tokens_inclusive, target_compression_ratio,
# min_target_words). Callers can override the ratio directly via
# `ClinicalSummarizerConfig.target_compression_ratio`.
DEFAULT_LENGTH_BUDGET_TIERS: tuple[tuple[int, float, int], ...] = (
    (500, 0.25, 60),
    (2000, 0.18, 90),
    (8000, 0.12, 170),
)
# Anything above the last tier uses these numbers.
DEFAULT_LONG_CHART_RATIO: float = 0.10
DEFAULT_LONG_CHART_MIN_WORDS: int = 220
DEFAULT_MIN_TARGET_WORDS: int = 35
# Absolute upper bound: a discharge summary humans actually read tops out at
# ~1.5 pages regardless of how long the source chart is.
DEFAULT_LENGTH_HARD_CAP_WORDS: int = 800
# Structured discharge summaries tokenize more densely than plain prose
# because headings, abbreviations, labs, and punctuation all cost extra.
_TOKENS_PER_WORD: float = 1.5
DEFAULT_LENGTH_BUDGET_HEADROOM: float = 1.20
DEFAULT_MIN_MAX_NEW_TOKENS: int = 160
DEFAULT_SUMMARY_TRUNCATION_RETRY_ATTEMPTS: int = 2
DEFAULT_SUMMARY_TRUNCATION_RETRY_SCALE: float = 1.5
DEFAULT_SUMMARY_TRUNCATION_RETRY_MIN_STEP_TOKENS: int = 64


@dataclass(slots=True, frozen=True)
class LengthBudget:
    """Computed length target for a single summarization call."""

    input_tokens: int
    input_words: int
    target_ratio: float
    target_words: int
    hard_cap_words: int
    max_new_tokens: int


def compute_length_budget(
    input_tokens: int,
    input_words: int,
    *,
    tiers: Sequence[tuple[int, float, int]] = DEFAULT_LENGTH_BUDGET_TIERS,
    long_chart_ratio: float = DEFAULT_LONG_CHART_RATIO,
    long_chart_min_words: int = DEFAULT_LONG_CHART_MIN_WORDS,
    target_compression_ratio: float | None = None,
    min_target_words: int = DEFAULT_MIN_TARGET_WORDS,
    hard_cap_words: int = DEFAULT_LENGTH_HARD_CAP_WORDS,
    headroom: float = DEFAULT_LENGTH_BUDGET_HEADROOM,
    min_max_new_tokens: int = DEFAULT_MIN_MAX_NEW_TOKENS,
) -> LengthBudget:
    """Derive a first-pass length target for a clinical summary.

    A single target ratio drives both prompting and token budgeting. The
    per-call hard cap is kept only slightly above the target so the decoder
    has room to finish the final bullet without inviting a large overshoot.
    """

    if input_tokens <= 0:
        raise ValueError(f"input_tokens must be positive, got {input_tokens}")
    if input_words <= 0:
        raise ValueError(f"input_words must be positive, got {input_words}")
    if hard_cap_words <= 0:
        raise ValueError("hard_cap_words must be positive")
    if min_target_words <= 0:
        raise ValueError("min_target_words must be positive")
    if target_compression_ratio is not None and not (0 < target_compression_ratio <= 1):
        raise ValueError("target_compression_ratio must be in (0, 1]")

    default_ratio: float
    tier_floor_words: int
    for max_in, tier_ratio, tier_floor in tiers:
        if input_tokens <= max_in:
            default_ratio = tier_ratio
            tier_floor_words = tier_floor
            break
    else:
        default_ratio = long_chart_ratio
        tier_floor_words = long_chart_min_words

    ratio = target_compression_ratio or default_ratio
    if target_compression_ratio is None:
        target_words = max(tier_floor_words, int(round(input_words * ratio)))
    else:
        target_words = max(min_target_words, int(round(input_words * ratio)))
    target_words = min(target_words, hard_cap_words)

    # Keep the cap close to target. We want enough room to finish cleanly,
    # not enough room to turn a concise prompt into a long-form summary.
    cap_words = min(
        hard_cap_words,
        max(target_words + 12, int(round(target_words * 1.12))),
    )
    cap_words = max(target_words, cap_words)

    max_new_tokens = max(
        min_max_new_tokens,
        int(round(cap_words * _TOKENS_PER_WORD * headroom)),
    )

    return LengthBudget(
        input_tokens=input_tokens,
        input_words=input_words,
        target_ratio=ratio,
        target_words=target_words,
        hard_cap_words=cap_words,
        max_new_tokens=max_new_tokens,
    )


@dataclass(slots=True)
class ClinicalSummarizerConfig:
    """Inference and prompting defaults for clinical summarization."""

    model_path: str = DEFAULT_MODEL_PATH
    # Generous ceiling so a complete structured summary rarely truncates; the
    # prompt + sampling settings are what enforce brevity. When
    # `enforce_length_budget` is True this is treated as an upper safety
    # cap on top of the per-call computed budget.
    max_tokens: int = 1536
    # Keep sampling nearly deterministic so the first pass stays terse and
    # schema-following under tight length budgets.
    temperature: float = 0.05
    # Tighter nucleus + a small min_p floor to suppress meandering low-prob tails
    # that tend to inflate length on Qwen.
    top_p: float = 0.7
    top_k: int = 0
    min_p: float = 0.05
    trust_remote_code: bool = False
    enable_thinking: bool = False
    verbose: bool = False
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    stop_strings: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_CLINICAL_STOP_STRINGS
    )
    # Optional explicit compression target (summary words / source-note words).
    # When unset, the summarizer uses a source-length-based heuristic.
    target_compression_ratio: float | None = None
    # When True, compute a per-call length budget from the source-note token
    # count and (a) inject a target word-count into the prompt, (b) cap
    # `max_new_tokens` accordingly. Disable to fall back to flat `max_tokens`.
    enforce_length_budget: bool = True
    # Multiplier on the computed cap when computing `max_new_tokens`. Slightly
    # >1 lets the model finish its final bullet + end marker. Keep this
    # tight so a permissive decoder doesn't blow the word budget.
    length_budget_headroom: float = DEFAULT_LENGTH_BUDGET_HEADROOM
    # If the model explicitly reports `finish_reason="length"`, retry with a
    # larger decode budget instead of silently accepting a cut-off summary.
    summary_truncation_retry_attempts: int = (
        DEFAULT_SUMMARY_TRUNCATION_RETRY_ATTEMPTS
    )
    summary_truncation_retry_scale: float = DEFAULT_SUMMARY_TRUNCATION_RETRY_SCALE
    summary_truncation_retry_min_step_tokens: int = (
        DEFAULT_SUMMARY_TRUNCATION_RETRY_MIN_STEP_TOKENS
    )
    # --- Agentic fact-check loop ----------------------------------------
    # When True, `summarize_text_with_verification` runs claim decomposition,
    # LLM claim judging for hallucination detection, and up to
    # `verification_max_passes` rounds of revision before returning.
    enable_verification: bool = False
    verification_max_passes: int = 3
    # Claims with judge confidence below this threshold are flagged as
    # unsupported and fed back into the revision prompt.
    claim_support_threshold: float = 0.5
    # Fact-check/revision system prompt used for decomposition and revision.
    fact_check_system_prompt: str = CLINICAL_SUMMARY_FACT_CHECK_SYSTEM_PROMPT
    claim_decomposition_stop_strings: tuple[str, ...] = (
        CLINICAL_SUMMARY_CLAIM_DECOMPOSITION_END_MARKER,
    )


@dataclass(slots=True)
class VerifiedClinicalSummary:
    """Final clinical summary plus its full agentic verification trace."""

    summary: str
    initial_summary: str
    verified: bool
    passes: tuple[VerificationPassResult, ...]


class ClinicalSummarizer:
    """Thin orchestration layer for source-grounded clinical summarization."""

    def __init__(
        self,
        config: ClinicalSummarizerConfig | None = None,
        inference: Qwen35MlxInference | None = None,
        claim_verifier: ClaimVerifier | object | None = None,
        entity_extractor: Callable[[str], set[tuple[str, str]]] | object | None = None,
    ) -> None:
        self.config = config or ClinicalSummarizerConfig()
        self.inference = inference or Qwen35MlxInference(
            MlxQwenConfig(
                model_path=self.config.model_path,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                min_p=self.config.min_p,
                trust_remote_code=self.config.trust_remote_code,
                enable_thinking=self.config.enable_thinking,
                verbose=self.config.verbose,
            )
        )
        self.claim_verifier = claim_verifier
        # Reserved for future non-LLM gates; unused in the current pipeline.
        self.entity_extractor = entity_extractor
        # Memo cache for claim decomposition: identical summaries produced
        # across passes (e.g. when the LLM declines to revise) can re-use the
        # prior decomposition output and skip a full Qwen generation step.
        self._claim_decomposition_cache: dict[tuple[str, str | None], list[str]] = {}

    def summarize_text(
        self,
        note_text: str,
        *,
        patient_id: str | None = None,
        note_type: str | None = None,
        source: str | None = None,
    ) -> str:
        note = ClinicalNote(
            content=note_text,
            source=source,
            note_type=note_type or "clinical note",
        )
        return self.summarize_notes([note], patient_id=patient_id)

    def summarize_notes(
        self,
        notes: Sequence[ClinicalNote],
        *,
        patient_id: str | None = None,
    ) -> str:
        normalized_notes = _normalize_notes(notes)
        budget = self._compute_budget_for(normalized_notes)
        return self._generate_summary(
            normalized_notes,
            patient_id=patient_id,
            budget=budget,
        )

    def _generate_summary(
        self,
        notes: Sequence[ClinicalNote],
        *,
        patient_id: str | None,
        budget: LengthBudget | None,
    ) -> str:
        prompt = build_clinical_summary_prompt(
            notes,
            patient_id=patient_id,
            length_budget=budget,
        )

        return self._complete_structured_summary(
            prompt,
            system_prompt=self.config.system_prompt,
            max_tokens=self._first_pass_max_tokens(budget),
            stop_strings=self.config.stop_strings,
            purpose="Clinical summary generation",
        )

    def _complete_summary(self, prompt: str, *, max_tokens: int) -> CompletionResult:
        return self._complete_text_with_meta(
            prompt,
            system_prompt=self.config.system_prompt,
            max_tokens=max_tokens,
            stop_strings=self.config.stop_strings,
        )

    def _complete_structured_summary(
        self,
        prompt: str,
        *,
        system_prompt: str,
        max_tokens: int,
        stop_strings: Sequence[str],
        purpose: str,
    ) -> str:
        result = self._complete_structured_summary_with_retries(
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            stop_strings=stop_strings,
            purpose=purpose,
        )
        return result.text

    def _complete_structured_summary_with_retries(
        self,
        prompt: str,
        *,
        system_prompt: str,
        max_tokens: int,
        stop_strings: Sequence[str],
        purpose: str,
    ) -> CompletionResult:
        current_max_tokens = max_tokens
        attempts = max(0, self.config.summary_truncation_retry_attempts)
        result = self._complete_text_with_meta(
            prompt,
            system_prompt=system_prompt,
            max_tokens=current_max_tokens,
            stop_strings=stop_strings,
        )

        for retry_index in range(1, attempts + 1):
            if result.finish_reason != "length":
                break
            next_max_tokens = self._next_summary_retry_max_tokens(current_max_tokens)
            if next_max_tokens <= current_max_tokens:
                break
            _LOG.info(
                "%s hit max_tokens=%d; retrying with max_tokens=%d (%d/%d).",
                purpose,
                current_max_tokens,
                next_max_tokens,
                retry_index,
                attempts,
            )
            current_max_tokens = next_max_tokens
            result = self._complete_text_with_meta(
                prompt,
                system_prompt=system_prompt,
                max_tokens=current_max_tokens,
                stop_strings=stop_strings,
            )

        if result.finish_reason == "length":
            _LOG.warning(
                "%s remained truncated after retries at max_tokens=%d.",
                purpose,
                current_max_tokens,
            )
        return result

    def _complete_text_with_meta(
        self,
        prompt: str,
        *,
        system_prompt: str,
        max_tokens: int,
        stop_strings: Sequence[str],
    ) -> CompletionResult:
        # Prefer the meta-aware path so downstream callers keep generation
        # metadata when the inference client exposes it. Test doubles that only
        # implement `complete()` still work via a synthetic CompletionResult.
        kwargs = self._completion_kwargs(
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            stop_strings=stop_strings,
        )
        if hasattr(self.inference, "complete_with_meta"):
            return self.inference.complete_with_meta(prompt, **kwargs)
        return CompletionResult(
            text=self.inference.complete(prompt, **kwargs),
            finish_reason=None,
            generation_tokens=0,
        )

    def _complete_text(
        self,
        prompt: str,
        *,
        system_prompt: str,
        max_tokens: int,
        stop_strings: Sequence[str],
    ) -> str:
        return self._complete_text_with_meta(
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            stop_strings=stop_strings,
        ).text

    def _completion_kwargs(
        self,
        *,
        system_prompt: str,
        max_tokens: int,
        stop_strings: Sequence[str],
    ) -> dict[str, object]:
        return {
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "min_p": self.config.min_p,
            "stop_strings": stop_strings,
            "verbose": self.config.verbose,
        }

    def stream_summarize_notes(
        self,
        notes: Sequence[ClinicalNote],
        *,
        patient_id: str | None = None,
    ) -> Iterator[str]:
        normalized_notes = _normalize_notes(notes)
        budget = self._compute_budget_for(normalized_notes)
        prompt = build_clinical_summary_prompt(
            normalized_notes,
            patient_id=patient_id,
            length_budget=budget,
        )
        return self.inference.stream(
            prompt,
            system_prompt=self.config.system_prompt,
            max_tokens=self._first_pass_max_tokens(budget),
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            min_p=self.config.min_p,
            stop_strings=self.config.stop_strings,
        )

    def summarize_path(
        self,
        path: str | Path,
        *,
        patient_id: str | None = None,
        recursive: bool = False,
        suffixes: Sequence[str] | None = None,
    ) -> str:
        notes = load_clinical_notes(path, recursive=recursive, suffixes=suffixes)
        return self.summarize_notes(notes, patient_id=patient_id)

    # --- Agentic fact-check loop ----------------------------------------

    def summarize_text_with_verification(
        self,
        note_text: str,
        *,
        patient_id: str | None = None,
        note_type: str | None = None,
        source: str | None = None,
    ) -> VerifiedClinicalSummary:
        """Generate a summary, then run up to N LLM-judge revision passes.

        The initial generation reuses the existing length-budget path. Each
        revision call is given the same `LengthBudget` so the prompt keeps the
        same compression target across passes. Returns the final summary plus
        the full per-pass trace (including the unrevised baseline at index 0 of
        `passes`).
        """

        if self.config.verification_max_passes < 1:
            raise ValueError("verification_max_passes must be at least 1.")

        note = ClinicalNote(
            content=note_text,
            source=source,
            note_type=note_type or "clinical note",
        )
        normalized_notes = _normalize_notes([note])
        budget = self._compute_budget_for(normalized_notes)
        cleaned_source_note = "\n\n".join(n.content for n in normalized_notes).strip()

        initial_summary = self._generate_summary(
            normalized_notes,
            patient_id=patient_id,
            budget=budget,
        )
        current_summary = initial_summary
        passes: list[VerificationPassResult] = []
        previous_unsupported: frozenset[str] | None = None

        for pass_index in range(1, self.config.verification_max_passes + 1):
            pass_result = self.verify_generated_summary(
                current_summary,
                source_note_text=cleaned_source_note,
                pass_index=pass_index,
                patient_id=patient_id,
            )
            passes.append(pass_result)
            if pass_result.passed:
                return VerifiedClinicalSummary(
                    summary=current_summary,
                    initial_summary=initial_summary,
                    verified=True,
                    passes=tuple(passes),
                )
            # Short-circuit: if revision didn't change which claims fail
            # verification, additional passes will keep producing the same
            # output. Stop early instead of burning more inference time.
            current_unsupported = frozenset(pass_result.unsupported_claims)
            if (
                previous_unsupported is not None
                and current_unsupported == previous_unsupported
            ):
                _LOG.info(
                    "Verification stalled at pass %d (unsupported set unchanged); "
                    "stopping early.",
                    pass_index,
                )
                break
            previous_unsupported = current_unsupported

            if pass_index == self.config.verification_max_passes:
                break

            current_summary = self._revise_clinical_summary(
                source_note_text=cleaned_source_note,
                current_summary=current_summary,
                unsupported_claims=pass_result.unsupported_claims,
                patient_id=patient_id,
                budget=budget,
            )

        return VerifiedClinicalSummary(
            summary=current_summary,
            initial_summary=initial_summary,
            verified=bool(passes and passes[-1].passed),
            passes=tuple(passes),
        )

    def decompose_summary_into_claims(
        self,
        summary: str,
        *,
        patient_id: str | None = None,
    ) -> list[str]:
        cache_key = (summary, patient_id)
        cached = self._claim_decomposition_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        prompt = build_clinical_summary_claim_decomposition_prompt(
            summary,
            patient_id=patient_id,
        )
        response = self._complete_fact_check_prompt(
            prompt,
            stop_strings=self.config.claim_decomposition_stop_strings,
        )
        claims = parse_atomic_claims(response)
        if not claims:
            claims = fallback_claim_split(summary)
        self._claim_decomposition_cache[cache_key] = list(claims)
        return claims

    def verify_generated_summary(
        self,
        summary: str,
        *,
        source_note_text: str,
        pass_index: int = 1,
        patient_id: str | None = None,
    ) -> VerificationPassResult:
        claims = tuple(
            self.decompose_summary_into_claims(summary, patient_id=patient_id)
        )
        claim_results = tuple(self._score_claims(source_note_text, claims))
        unsupported_claims = tuple(
            result.claim
            for result in claim_results
            if (not result.supported)
            or result.probability < self.config.claim_support_threshold
        )
        passed = not unsupported_claims
        return VerificationPassResult(
            pass_index=pass_index,
            summary=summary,
            claims=claims,
            claim_results=claim_results,
            unsupported_claims=unsupported_claims,
            entity_metrics=None,
            passed=passed,
        )

    def _revise_clinical_summary(
        self,
        *,
        source_note_text: str,
        current_summary: str,
        unsupported_claims: Sequence[str],
        patient_id: str | None,
        budget: LengthBudget | None,
    ) -> str:
        prompt = build_clinical_summary_revision_prompt(
            source_note_text,
            current_summary,
            unsupported_claims,
            patient_id=patient_id,
            length_target_words=budget.target_words if budget else None,
            length_hard_cap_words=budget.hard_cap_words if budget else None,
            length_target_ratio=budget.target_ratio if budget else None,
            length_target_tokens=budget.max_new_tokens if budget else None,
        )
        # Cap revision length to the same per-call budget so revisions can
        # never grow back past the original target.
        max_tokens = self._first_pass_max_tokens(budget)
        return self._complete_text(
            prompt,
            system_prompt=self.config.fact_check_system_prompt,
            max_tokens=max_tokens,
            stop_strings=self.config.stop_strings,
        )

    def _complete_fact_check_prompt(
        self, prompt: str, *, stop_strings: Sequence[str]
    ) -> str:
        # Decomposition output is short; cap at half the configured ceiling
        # so a runaway decomposer can't dominate runtime.
        max_tokens = max(256, self.config.max_tokens // 2)
        return self._complete_text(
            prompt,
            system_prompt=self.config.fact_check_system_prompt,
            max_tokens=max_tokens,
            stop_strings=stop_strings,
        )

    def _complete_claim_judge_prompt(
        self,
        prompt: str,
        *,
        system_prompt: str,
        stop_strings: Sequence[str],
    ) -> str:
        # Claim-judge responses are compact JSON lines; a smaller cap keeps
        # the verifier deterministic and inexpensive.
        max_tokens = max(256, self.config.max_tokens // 2)
        return self._complete_text(
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            stop_strings=stop_strings,
        )

    def _score_claims(
        self,
        source_note_text: str,
        claims: Sequence[str],
    ) -> list[ClaimVerificationResult]:
        return score_claims_via_adapter(
            self._get_claim_verifier(), source_note_text, claims
        )

    def _get_claim_verifier(self) -> ClaimVerifier | object:
        if self.claim_verifier is None:
            self.claim_verifier = LLMClaimVerifier(
                self._complete_claim_judge_prompt
            )
        return self.claim_verifier

    def _compute_budget_for(
        self, notes: Sequence[ClinicalNote]
    ) -> LengthBudget | None:
        """Tokenize the source notes and derive a per-call length budget.

        Returns None when the budget is disabled in config or when token
        counting fails for any reason; callers fall back to flat
        `config.max_tokens`.
        """

        if not self.config.enforce_length_budget:
            return None

        joined = "\n\n".join(note.content for note in notes if note.content)
        if not joined.strip():
            return None

        try:
            input_tokens = self._count_tokens(joined)
            input_words = _count_words(joined)
        except Exception as exc:  # pragma: no cover - tokenizer init failure
            _LOG.warning(
                "Token counting failed (%s); skipping length budget.", exc
            )
            return None

        if input_tokens <= 0 or input_words <= 0:
            return None

        budget = compute_length_budget(
            input_tokens,
            input_words,
            target_compression_ratio=self.config.target_compression_ratio,
            headroom=self.config.length_budget_headroom,
        )
        _LOG.info(
            "Length budget: input=%d tokens / %d words -> target=%d words "
            "(~%.0f%%), cap=%d words, max_new_tokens=%d",
            budget.input_tokens,
            budget.input_words,
            budget.target_words,
            budget.target_ratio * 100,
            budget.hard_cap_words,
            budget.max_new_tokens,
        )
        return budget

    def _count_tokens(self, text: str) -> int:
        """Use the loaded inference tokenizer to count source tokens.

        Loads the model lazily because the tokenizer is bundled with it in
        mlx-lm; callers always summarize right after, so this is free.
        """

        self.inference.load_model()
        tokenizer = self.inference.tokenizer
        if tokenizer is None:
            return 0
        encoded = tokenizer.encode(text)
        return len(encoded)

    def _first_pass_max_tokens(self, budget: LengthBudget | None) -> int:
        """Cap the first-pass decode at min(budget, configured max_tokens)."""

        if budget is None:
            return self.config.max_tokens
        return max(64, min(budget.max_new_tokens, self.config.max_tokens))

    def _next_summary_retry_max_tokens(self, current_max_tokens: int) -> int:
        """Grow the decode budget for a retry without exceeding config.max_tokens."""

        if current_max_tokens >= self.config.max_tokens:
            return current_max_tokens

        scaled = int(
            round(
                current_max_tokens
                * max(1.0, self.config.summary_truncation_retry_scale)
            )
        )
        stepped = (
            current_max_tokens
            + max(1, self.config.summary_truncation_retry_min_step_tokens)
        )
        grown = max(current_max_tokens + 1, scaled, stepped)
        return min(self.config.max_tokens, grown)


class VerifiedClinicalSummarizerAdapter:
    """Plug a verified ClinicalSummarizer into ``benchmarks.runner.run``.

    The runner expects ``summarize_text(note_text, patient_id=...) -> str``
    and caches the returned string under ``<cache_dir>/<model_tag>__<row_id>.txt``.
    This adapter:

    - Calls ``summarize_text_with_verification`` so the agentic loop runs.
    - Returns ``result.summary`` (the final, post-revision text) for the cache.
    - Writes a sibling ``<model_tag>__<row_id>.verify.json`` next to the cache
      file containing per-pass metadata (claims, unsupported counts, entity
      P/R/F1, total passes used) so the notebook can surface verification
      stats without re-running the loop.

    The ``patient_id`` arg from the runner is also the row_id (the runner
    casts ``int(row_id)`` to ``str``), which is exactly what we need to key
    the sibling JSON file.
    """

    def __init__(
        self,
        summarizer: ClinicalSummarizer,
        *,
        cache_dir: str | Path,
        model_tag: str,
    ) -> None:
        self._summarizer = summarizer
        self._cache_dir = Path(cache_dir)
        self._sanitized_tag = _sanitize_cache_tag(model_tag)

    def summarize_text(
        self,
        note_text: str,
        *,
        patient_id: str | None = None,
    ) -> str:
        result = self._summarizer.summarize_text_with_verification(
            note_text,
            patient_id=patient_id,
        )
        if patient_id is not None:
            self._write_verification_metadata(patient_id, result)
        return result.summary

    def _write_verification_metadata(
        self,
        row_id: str,
        result: VerifiedClinicalSummary,
    ) -> None:
        metadata_path = (
            self._cache_dir / f"{self._sanitized_tag}__{row_id}.verify.json"
        )
        payload = serialize_verified_summary(result)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(metadata_path)


def serialize_verified_summary(result: VerifiedClinicalSummary) -> dict:
    """Convert a VerifiedClinicalSummary into a JSON-safe dict for the cache."""

    return {
        "verified": result.verified,
        "n_passes": len(result.passes),
        "initial_summary_chars": len(result.initial_summary),
        "final_summary_chars": len(result.summary),
        "passes": [
            {
                "pass_index": p.pass_index,
                "n_claims": len(p.claims),
                "n_unsupported": len(p.unsupported_claims),
                "claim_results": [
                    {
                        "claim": cr.claim,
                        "supported": cr.supported,
                        "probability": cr.probability,
                        "citations": [
                            {
                                "snippet": citation.snippet,
                                "start_char": citation.start_char,
                                "end_char": citation.end_char,
                                "start_line": citation.start_line,
                                "end_line": citation.end_line,
                                "score": citation.score,
                                "method": citation.method,
                            }
                            for citation in cr.citations
                        ],
                    }
                    for cr in p.claim_results
                ],
                "unsupported_claims": list(p.unsupported_claims),
                "entity_metrics": (
                    {
                        "source_entity_count": p.entity_metrics.source_entity_count,
                        "summary_entity_count": p.entity_metrics.summary_entity_count,
                        "overlap_count": p.entity_metrics.overlap_count,
                        "precision": p.entity_metrics.precision,
                        "recall": p.entity_metrics.recall,
                        "f1": p.entity_metrics.f1,
                    }
                    if p.entity_metrics is not None
                    else None
                ),
                "passed": p.passed,
            }
            for p in result.passes
        ],
    }


def _sanitize_cache_tag(tag: str) -> str:
    """Mirror benchmarks.runner._sanitize_tag so JSON sidecar paths line up."""

    import re as _re

    cleaned = _re.sub(r"[^A-Za-z0-9._-]+", "-", tag.strip())
    return cleaned.strip("-") or "model"


def build_clinical_summary_prompt(
    notes: Sequence[ClinicalNote],
    *,
    patient_id: str | None = None,
    length_budget: LengthBudget | None = None,
) -> str:
    normalized_notes = _normalize_notes(notes)
    rendered_note_blocks = [
        note.render(index + 1) for index, note in enumerate(normalized_notes)
    ]
    return build_clinical_summary_prompt_from_blocks(
        rendered_note_blocks,
        patient_id=patient_id,
        length_target_words=length_budget.target_words if length_budget else None,
        length_hard_cap_words=length_budget.hard_cap_words if length_budget else None,
        length_target_ratio=length_budget.target_ratio if length_budget else None,
        length_target_tokens=length_budget.max_new_tokens if length_budget else None,
    )


def infer_note_type(path: str | Path) -> str:
    resolved_path = Path(path)
    label = resolved_path.stem.replace("_", " ").replace("-", " ").strip()
    return label or "clinical note"


def load_clinical_notes(
    path: str | Path,
    *,
    recursive: bool = False,
    suffixes: Sequence[str] | None = None,
    encoding: str = "utf-8",
) -> list[ClinicalNote]:
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved_path}")

    if resolved_path.is_file():
        note = ClinicalNote.from_path(resolved_path, encoding=encoding)
        if not note.content.strip():
            raise ValueError(f"Input file is empty: {resolved_path}")
        return [note]

    allowed_suffixes = _normalize_suffixes(suffixes or DEFAULT_DIRECTORY_SUFFIXES)
    file_iter = resolved_path.rglob("*") if recursive else resolved_path.glob("*")
    note_files = sorted(
        file_path
        for file_path in file_iter
        if file_path.is_file() and file_path.suffix.lower() in allowed_suffixes
    )
    if not note_files:
        joined_suffixes = ", ".join(sorted(allowed_suffixes))
        raise ValueError(
            f"No note files found in {resolved_path} matching suffixes: {joined_suffixes}"
        )

    notes = [
        ClinicalNote.from_path(file_path, encoding=encoding)
        for file_path in note_files
    ]
    non_empty_notes = [note for note in notes if note.content.strip()]
    if not non_empty_notes:
        raise ValueError(f"All candidate note files were empty in: {resolved_path}")
    return non_empty_notes


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize clinical notes into a brief hospital course and structured summary."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        help="Path to a note file or a directory of note files.",
    )
    input_group.add_argument(
        "--text",
        help="Raw clinical note text to summarize.",
    )
    parser.add_argument(
        "--patient-id",
        default=None,
        help="Optional patient or encounter identifier echoed into the prompt context.",
    )
    parser.add_argument(
        "--note-type",
        default=None,
        help="Optional note type label when using --text.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Model repo or local path to an MLX-compatible model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1536,
        help="Maximum number of tokens to generate for the summary.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--target-compression-ratio",
        type=float,
        default=None,
        help=(
            "Optional explicit compression target as summary_words / source_note_words "
            "(for example 0.18 for ~18%%)."
        ),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.7,
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
        default=0.05,
        help="Min-p sampling value. Use 0 to disable.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when --input points to a directory.",
    )
    parser.add_argument(
        "--suffix",
        action="append",
        default=None,
        help="Allowed file suffix when reading a directory. Repeat to include multiple suffixes.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens as they are generated.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow models that require trusted remote tokenizer code.",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Qwen thinking mode and show reasoning output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose generation output for non-streamed calls.",
    )
    parser.add_argument(
        "--render-markdown",
        action="store_true",
        help="Render model output as markdown in the terminal after each response completes.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.render_markdown:
        ensure_terminal_markdown_support()
    summarizer = ClinicalSummarizer(
        ClinicalSummarizerConfig(
            model_path=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            target_compression_ratio=args.target_compression_ratio,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            trust_remote_code=args.trust_remote_code,
            enable_thinking=args.thinking,
            verbose=args.verbose,
        )
    )

    if args.text is not None:
        notes = [
            ClinicalNote(
                content=args.text,
                note_type=args.note_type or "clinical note",
                source="--text",
            )
        ]
    else:
        notes = load_clinical_notes(
            args.input,
            recursive=args.recursive,
            suffixes=args.suffix,
        )

    if args.stream:
        chunks: list[str] = []
        for chunk in summarizer.stream_summarize_notes(notes, patient_id=args.patient_id):
            if args.render_markdown:
                chunks.append(chunk)
                continue
            print(chunk, end="", flush=True)
        if args.render_markdown:
            rendered_summary = Qwen35MlxInference._clean_response_text("".join(chunks))
            print_terminal_output(rendered_summary, render_markdown=True)
        else:
            print()
        return

    summary = summarizer.summarize_notes(notes, patient_id=args.patient_id)
    print_terminal_output(summary, render_markdown=args.render_markdown)


def _normalize_notes(notes: Sequence[ClinicalNote]) -> list[ClinicalNote]:
    normalized_notes = [
        ClinicalNote(
            content=note.content.strip(),
            source=note.source,
            note_type=note.note_type,
            note_id=note.note_id,
        )
        for note in notes
        if note.content and note.content.strip()
    ]
    if not normalized_notes:
        raise ValueError("At least one non-empty clinical note is required.")
    return normalized_notes


def _normalize_suffixes(suffixes: Sequence[str]) -> set[str]:
    normalized_suffixes = set()
    for suffix in suffixes:
        cleaned = suffix.strip().lower()
        if not cleaned:
            continue
        normalized_suffixes.add(cleaned if cleaned.startswith(".") else f".{cleaned}")
    if not normalized_suffixes:
        raise ValueError("At least one valid suffix is required.")
    return normalized_suffixes


def _strip_terminal_marker(text: str, marker: str) -> str:
    stripped = text.rstrip()
    if stripped.endswith(marker):
        return stripped[: -len(marker)].rstrip()
    return stripped


def _count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


if __name__ == "__main__":
    main()
