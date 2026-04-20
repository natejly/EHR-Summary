"""Rewrite clinical summaries into patient-friendly language using local MLX inference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

from mlx_qwen_inference import (
    MlxQwenConfig,
    Qwen35MlxInference,
    ensure_terminal_markdown_support,
    print_terminal_output,
)
from prompts import (
    PATIENT_FRIENDLY_CLAIM_DECOMPOSITION_END_MARKER,
    PATIENT_FRIENDLY_FACT_CHECK_SYSTEM_PROMPT,
    PATIENT_FRIENDLY_SIMPLIFICATION_END_MARKER,
    PATIENT_FRIENDLY_SIMPLIFICATION_SYSTEM_PROMPT,
    build_patient_friendly_claim_decomposition_prompt,
    build_patient_friendly_revision_prompt,
    build_patient_friendly_simplification_prompt,
)
from verification import (
    ClaimVerificationResult,
    ClaimVerifier,
    EntityExtractor,
    EntityVerificationMetrics,
    MedSpaCyEntityExtractor,
    MiniCheckClaimVerifier,
    VerificationPassResult,
    _CallableEntityExtractor,
    compute_entity_metrics,
    fallback_claim_split as _fallback_claim_split,
    format_entity_feedback as _format_entity_feedback,
    parse_atomic_claims as _parse_atomic_claims,
    score_claims_via_adapter,
)


DEFAULT_MODEL_PATH = "mlx-community/Qwen3.5-9B-4bit"


@dataclass(slots=True)
class VerifiedSimplificationResult:
    """Final simplified summary plus verification metadata."""

    summary: str
    verified: bool
    passes: tuple[VerificationPassResult, ...]


@dataclass(slots=True)
class PatientFriendlySimplifierConfig:
    """Inference and prompting defaults for patient-friendly rewriting."""

    model_path: str = DEFAULT_MODEL_PATH
    # Headroom so a 4-6 section, 8th-grade rewrite never truncates mid-sentence.
    max_tokens: int = 1536
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 0
    min_p: float = 0.05
    trust_remote_code: bool = False
    enable_thinking: bool = False
    verbose: bool = False
    system_prompt: str = PATIENT_FRIENDLY_SIMPLIFICATION_SYSTEM_PROMPT
    stop_strings: tuple[str, ...] = (PATIENT_FRIENDLY_SIMPLIFICATION_END_MARKER,)
    fact_check_system_prompt: str = PATIENT_FRIENDLY_FACT_CHECK_SYSTEM_PROMPT
    claim_decomposition_stop_strings: tuple[str, ...] = (
        PATIENT_FRIENDLY_CLAIM_DECOMPOSITION_END_MARKER,
    )
    verification_max_passes: int = 3
    claim_support_threshold: float = 0.5
    entity_min_precision: float | None = None
    entity_min_recall: float | None = None


class PatientFriendlySimplifier:
    """Second-stage rewrite layer for converting summaries into plain language."""

    def __init__(
        self,
        config: PatientFriendlySimplifierConfig | None = None,
        inference: Qwen35MlxInference | None = None,
        claim_verifier: ClaimVerifier | object | None = None,
        entity_extractor: EntityExtractor | Callable[[str], set[tuple[str, str]]] | None = None,
    ) -> None:
        self.config = config or PatientFriendlySimplifierConfig()
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
        self.entity_extractor = entity_extractor

    def simplify_text(
        self,
        clinical_summary: str,
        *,
        audience: str | None = None,
        patient_id: str | None = None,
    ) -> str:
        return self._generate_patient_friendly_summary(
            clinical_summary,
            audience=audience,
            patient_id=patient_id,
        )

    def simplify_text_with_verification(
        self,
        clinical_summary: str,
        *,
        source_note_text: str,
        audience: str | None = None,
        patient_id: str | None = None,
    ) -> VerifiedSimplificationResult:
        cleaned_source_note = source_note_text.strip()
        if not cleaned_source_note:
            raise ValueError("Source note text is required for verification.")
        if self.config.verification_max_passes < 1:
            raise ValueError("verification_max_passes must be at least 1.")

        current_summary = self.simplify_text(
            clinical_summary,
            audience=audience,
            patient_id=patient_id,
        )
        passes: list[VerificationPassResult] = []

        for pass_index in range(1, self.config.verification_max_passes + 1):
            pass_result = self.verify_generated_summary(
                current_summary,
                source_note_text=cleaned_source_note,
                pass_index=pass_index,
                patient_id=patient_id,
            )
            passes.append(pass_result)
            if pass_result.passed:
                return VerifiedSimplificationResult(
                    summary=current_summary,
                    verified=True,
                    passes=tuple(passes),
                )
            if pass_index == self.config.verification_max_passes:
                break

            current_summary = self._revise_summary(
                clinical_summary=clinical_summary,
                current_summary=current_summary,
                source_note_text=cleaned_source_note,
                unsupported_claims=pass_result.unsupported_claims,
                audience=audience,
                patient_id=patient_id,
                entity_metrics=pass_result.entity_metrics,
            )

        return VerifiedSimplificationResult(
            summary=current_summary,
            verified=bool(passes and passes[-1].passed),
            passes=tuple(passes),
        )

    def stream_simplify_text(
        self,
        clinical_summary: str,
        *,
        audience: str | None = None,
        patient_id: str | None = None,
    ) -> Iterator[str]:
        prompt = build_patient_friendly_simplification_prompt(
            clinical_summary,
            audience=audience,
            patient_id=patient_id,
        )
        return self.inference.stream(
            prompt,
            system_prompt=self.config.system_prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            min_p=self.config.min_p,
            stop_strings=self.config.stop_strings,
        )

    def simplify_path(
        self,
        path: str | Path,
        *,
        audience: str | None = None,
        patient_id: str | None = None,
    ) -> str:
        return self.simplify_text(
            load_summary_text(path),
            audience=audience,
            patient_id=patient_id,
        )

    def simplify_path_with_verification(
        self,
        path: str | Path,
        *,
        source_note_path: str | Path,
        audience: str | None = None,
        patient_id: str | None = None,
    ) -> VerifiedSimplificationResult:
        return self.simplify_text_with_verification(
            load_summary_text(path),
            source_note_text=load_summary_text(source_note_path),
            audience=audience,
            patient_id=patient_id,
        )

    def decompose_summary_into_claims(
        self,
        summary: str,
        *,
        patient_id: str | None = None,
    ) -> list[str]:
        prompt = build_patient_friendly_claim_decomposition_prompt(
            summary,
            patient_id=patient_id,
        )
        response = self._complete_prompt(
            prompt,
            system_prompt=self.config.fact_check_system_prompt,
            stop_strings=self.config.claim_decomposition_stop_strings,
        )
        claims = _parse_atomic_claims(response)
        if claims:
            return claims
        return _fallback_claim_split(summary)

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
        entity_metrics = self._compute_entity_metrics(source_note_text, summary)
        passed = not unsupported_claims and self._entity_thresholds_pass(entity_metrics)
        return VerificationPassResult(
            pass_index=pass_index,
            summary=summary,
            claims=claims,
            claim_results=claim_results,
            unsupported_claims=unsupported_claims,
            entity_metrics=entity_metrics,
            passed=passed,
        )

    def _generate_patient_friendly_summary(
        self,
        clinical_summary: str,
        *,
        audience: str | None = None,
        patient_id: str | None = None,
    ) -> str:
        prompt = build_patient_friendly_simplification_prompt(
            clinical_summary,
            audience=audience,
            patient_id=patient_id,
        )
        return self._complete_prompt(
            prompt,
            system_prompt=self.config.system_prompt,
            stop_strings=self.config.stop_strings,
        )

    def _complete_prompt(
        self,
        prompt: str,
        *,
        system_prompt: str,
        stop_strings: Sequence[str],
    ) -> str:
        return self.inference.complete(
            prompt,
            system_prompt=system_prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            min_p=self.config.min_p,
            stop_strings=stop_strings,
            verbose=self.config.verbose,
        )

    def _revise_summary(
        self,
        *,
        clinical_summary: str,
        current_summary: str,
        source_note_text: str,
        unsupported_claims: Sequence[str],
        audience: str | None,
        patient_id: str | None,
        entity_metrics: EntityVerificationMetrics | None,
    ) -> str:
        prompt = build_patient_friendly_revision_prompt(
            clinical_summary,
            current_summary,
            source_note_text,
            unsupported_claims,
            audience=audience,
            patient_id=patient_id,
            entity_feedback=_format_entity_feedback(entity_metrics),
        )
        return self._complete_prompt(
            prompt,
            system_prompt=self.config.fact_check_system_prompt,
            stop_strings=self.config.stop_strings,
        )

    def _score_claims(
        self,
        source_note_text: str,
        claims: Sequence[str],
    ) -> list[ClaimVerificationResult]:
        return score_claims_via_adapter(
            self._get_claim_verifier(),
            source_note_text,
            claims,
        )

    def _get_claim_verifier(self) -> ClaimVerifier | object:
        if self.claim_verifier is None:
            self.claim_verifier = MiniCheckClaimVerifier()
        return self.claim_verifier

    def _compute_entity_metrics(
        self,
        source_note_text: str,
        summary: str,
    ) -> EntityVerificationMetrics | None:
        return compute_entity_metrics(
            self._get_entity_extractor(), source_note_text, summary
        )

    def _get_entity_extractor(self) -> EntityExtractor | None:
        if self.entity_extractor is not None:
            return _CallableEntityExtractor(self.entity_extractor)
        if (
            self.config.entity_min_precision is not None
            or self.config.entity_min_recall is not None
        ):
            self.entity_extractor = MedSpaCyEntityExtractor()
            return self.entity_extractor
        return None

    def _entity_thresholds_pass(
        self,
        entity_metrics: EntityVerificationMetrics | None,
    ) -> bool:
        if entity_metrics is None:
            return True
        if (
            self.config.entity_min_precision is not None
            and entity_metrics.precision < self.config.entity_min_precision
        ):
            return False
        if (
            self.config.entity_min_recall is not None
            and entity_metrics.recall < self.config.entity_min_recall
        ):
            return False
        return True


def load_summary_text(path: str | Path, *, encoding: str = "utf-8") -> str:
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved_path}")
    if not resolved_path.is_file():
        raise ValueError(f"Expected a file path, got: {resolved_path}")

    summary_text = resolved_path.read_text(encoding=encoding).strip()
    if not summary_text:
        raise ValueError(f"Input summary file is empty: {resolved_path}")
    return summary_text


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rewrite a clinical summary into patient-friendly language."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        help="Path to a text file containing the clinical summary to rewrite.",
    )
    input_group.add_argument(
        "--text",
        help="Raw clinical summary text to rewrite.",
    )
    verification_group = parser.add_mutually_exclusive_group(required=False)
    verification_group.add_argument(
        "--source-note",
        help="Optional path to the original clinical note used for iterative verification.",
    )
    verification_group.add_argument(
        "--source-text",
        help="Optional raw source clinical note text used for iterative verification.",
    )
    parser.add_argument(
        "--audience",
        default=None,
        help="Optional audience label, for example 'patient' or 'caregiver'.",
    )
    parser.add_argument(
        "--patient-id",
        default=None,
        help="Optional patient or encounter identifier echoed into the prompt context.",
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
        help="Maximum number of tokens to generate for the rewrite.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
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
    parser.add_argument(
        "--max-verification-passes",
        type=int,
        default=3,
        help="Maximum number of fact-checking revision passes when source-note verification is enabled.",
    )
    parser.add_argument(
        "--claim-support-threshold",
        type=float,
        default=0.5,
        help="Minimum MiniCheck probability required for a claim to count as supported.",
    )
    parser.add_argument(
        "--entity-min-precision",
        type=float,
        default=None,
        help="Optional minimum entity precision threshold for the entity cross-check.",
    )
    parser.add_argument(
        "--entity-min-recall",
        type=float,
        default=None,
        help="Optional minimum entity recall threshold for the entity cross-check.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.render_markdown:
        ensure_terminal_markdown_support()
    simplifier = PatientFriendlySimplifier(
        PatientFriendlySimplifierConfig(
            model_path=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            trust_remote_code=args.trust_remote_code,
            enable_thinking=args.thinking,
            verbose=args.verbose,
            verification_max_passes=args.max_verification_passes,
            claim_support_threshold=args.claim_support_threshold,
            entity_min_precision=args.entity_min_precision,
            entity_min_recall=args.entity_min_recall,
        )
    )

    summary_text = args.text if args.text is not None else load_summary_text(args.input)
    source_note_text = None
    if args.source_text is not None:
        source_note_text = args.source_text.strip()
    elif args.source_note is not None:
        source_note_text = load_summary_text(args.source_note)

    if args.stream and source_note_text is not None:
        raise ValueError(
            "Streaming is not supported when iterative verification is enabled."
        )

    if args.stream:
        chunks: list[str] = []
        for chunk in simplifier.stream_simplify_text(
            summary_text,
            audience=args.audience,
            patient_id=args.patient_id,
        ):
            if args.render_markdown:
                chunks.append(chunk)
                continue
            print(chunk, end="", flush=True)
        if args.render_markdown:
            rendered_rewrite = Qwen35MlxInference._clean_response_text("".join(chunks))
            print_terminal_output(rendered_rewrite, render_markdown=True)
        else:
            print()
        return

    if source_note_text is not None:
        rewrite = simplifier.simplify_text_with_verification(
            summary_text,
            source_note_text=source_note_text,
            audience=args.audience,
            patient_id=args.patient_id,
        ).summary
    else:
        rewrite = simplifier.simplify_text(
            summary_text,
            audience=args.audience,
            patient_id=args.patient_id,
        )
    print_terminal_output(rewrite, render_markdown=args.render_markdown)


if __name__ == "__main__":
    main()
