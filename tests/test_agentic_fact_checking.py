from __future__ import annotations

import unittest

from clinical_summarization import (
    ClinicalSummarizer,
    ClinicalSummarizerConfig,
    VerifiedClinicalSummary,
    serialize_verified_summary,
)
from mlx_qwen_inference import CompletionResult
from prompts import (
    CLAIM_VERDICTS_END_MARKER,
    CLINICAL_SUMMARY_CLAIM_DECOMPOSITION_END_MARKER,
)
from verification import (
    ClaimVerificationResult,
    LLMClaimVerifier,
    VerificationPassResult,
    parse_claim_verdicts,
    score_claims_via_adapter,
)


class LLMJudgeVerificationTests(unittest.TestCase):
    def test_parse_claim_verdicts_fails_closed_for_missing_claims(self) -> None:
        claims = ["HTN documented.", "Troponin 9.9 documented."]
        response = (
            '{"index": 1, "supported": true, "confidence": 0.98}\n'
            f"{CLAIM_VERDICTS_END_MARKER}"
        )

        results = parse_claim_verdicts(response, claims)

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].supported)
        self.assertEqual(results[0].probability, 0.98)
        self.assertFalse(results[1].supported)
        self.assertEqual(results[1].probability, 0.0)

    def test_llm_claim_verifier_parses_jsonl_judgments(self) -> None:
        calls: list[tuple[str, str, tuple[str, ...]]] = []

        def fake_complete(
            prompt: str,
            *,
            system_prompt: str,
            stop_strings: tuple[str, ...],
        ) -> str:
            calls.append((prompt, system_prompt, stop_strings))
            return (
                '{"index": 1, "supported": true, "confidence": 0.91}\n'
                '{"index": 2, "supported": false, "confidence": 0.12}\n'
                f"{CLAIM_VERDICTS_END_MARKER}"
            )

        verifier = LLMClaimVerifier(fake_complete)
        results = verifier.score_claims(
            "Source note mentions HTN only.",
            ["HTN documented.", "Troponin 9.9 documented."],
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].supported)
        self.assertFalse(results[1].supported)

    def test_score_claims_adapter_attaches_citation_dicts(self) -> None:
        class FakeVerifier:
            def score_claims(self, source_note, claims):
                return [
                    ClaimVerificationResult(
                        claim=claims[0],
                        supported=True,
                        probability=0.92,
                    ),
                    ClaimVerificationResult(
                        claim=claims[1],
                        supported=False,
                        probability=0.08,
                    ),
                ]

        results = score_claims_via_adapter(
            FakeVerifier(),
            "Assessment: HTN.\nTroponin negative.\nPlan: discharge home.",
            ["HTN documented.", "Troponin 9.9 documented."],
        )

        self.assertEqual(len(results), 2)
        self.assertGreater(len(results[0].citations), 0)
        self.assertEqual(results[0].citations[0].start_line, 1)
        self.assertIn("HTN", results[0].citations[0].snippet)
        self.assertEqual(results[1].citations, ())

    def test_serialized_verification_includes_citations(self) -> None:
        result = VerifiedClinicalSummary(
            summary="Diagnoses:\n- HTN documented.",
            initial_summary="Diagnoses:\n- HTN documented.",
            verified=True,
            passes=(
                VerificationPassResult(
                    pass_index=1,
                    summary="Diagnoses:\n- HTN documented.",
                    claims=("HTN documented.",),
                    claim_results=(
                        ClaimVerificationResult(
                            claim="HTN documented.",
                            supported=True,
                            probability=0.99,
                            citations=score_claims_via_adapter(
                                type(
                                    "Verifier",
                                    (),
                                    {
                                        "score_claims": lambda self, source, claims: [
                                            ClaimVerificationResult(
                                                claim=claims[0],
                                                supported=True,
                                                probability=0.99,
                                            )
                                        ]
                                    },
                                )(),
                                "Assessment: HTN.",
                                ("HTN documented.",),
                            )[0].citations,
                        ),
                    ),
                    unsupported_claims=(),
                    entity_metrics=None,
                    passed=True,
                ),
            ),
        )

        payload = serialize_verified_summary(result)

        citations = payload["passes"][0]["claim_results"][0]["citations"]
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["start_line"], 1)
        self.assertIn("HTN", citations[0]["snippet"])


class ClinicalSummarizerLLMJudgeTests(unittest.TestCase):
    def test_summary_retries_when_generation_hits_length(self) -> None:
        class FakeInference:
            def __init__(self) -> None:
                self.calls: list[int] = []
                self.tokenizer = self

            def load_model(self) -> None:
                return None

            def encode(self, text: str) -> list[int]:
                return list(range(len(text.split())))

            def complete_with_meta(self, prompt: str, **kwargs) -> CompletionResult:
                self.calls.append(int(kwargs["max_tokens"]))
                if len(self.calls) == 1:
                    return CompletionResult(
                        text="Brief Hospital Course:\n- partial summary",
                        finish_reason="length",
                        generation_tokens=int(kwargs["max_tokens"]),
                    )
                return CompletionResult(
                    text="Brief Hospital Course:\n- completed summary",
                    finish_reason="stop_string",
                    generation_tokens=int(kwargs["max_tokens"]),
                )

        fake_inference = FakeInference()
        summarizer = ClinicalSummarizer(
            ClinicalSummarizerConfig(
                max_tokens=320,
                enforce_length_budget=True,
                summary_truncation_retry_attempts=2,
                summary_truncation_retry_scale=1.5,
                summary_truncation_retry_min_step_tokens=64,
            ),
            inference=fake_inference,
        )

        summary = summarizer.summarize_text("word " * 100)

        self.assertEqual(summary, "Brief Hospital Course:\n- completed summary")
        self.assertEqual(len(fake_inference.calls), 2)
        self.assertGreater(fake_inference.calls[1], fake_inference.calls[0])
        self.assertLessEqual(fake_inference.calls[1], summarizer.config.max_tokens)

    def test_verification_loop_uses_llm_judge_and_revises_hallucinated_claims(self) -> None:
        class FakeInference:
            def __init__(self) -> None:
                self.complete_calls: list[str] = []

            def complete(self, prompt: str, **kwargs) -> str:
                self.complete_calls.append(prompt)
                if "Judge each atomic clinical claim" in prompt:
                    if "2. Troponin 9.9 documented." in prompt:
                        return (
                            '{"index": 1, "supported": true, "confidence": 0.99}\n'
                            '{"index": 2, "supported": false, "confidence": 0.01}\n'
                            f"{CLAIM_VERDICTS_END_MARKER}"
                        )
                    return (
                        '{"index": 1, "supported": true, "confidence": 0.99}\n'
                        f"{CLAIM_VERDICTS_END_MARKER}"
                    )
                if "Break the following STRUCTURED clinical summary into atomic factual claims." in prompt:
                    if "Troponin 9.9 documented." in prompt:
                        return (
                            "HTN documented.\n"
                            "Troponin 9.9 documented.\n"
                            f"{CLINICAL_SUMMARY_CLAIM_DECOMPOSITION_END_MARKER}"
                        )
                    return (
                        "HTN documented.\n"
                        f"{CLINICAL_SUMMARY_CLAIM_DECOMPOSITION_END_MARKER}"
                    )
                if "Revise the STRUCTURED clinical summary below" in prompt:
                    return "Diagnoses:\n- HTN documented."
                if "Produce a STRUCTURED, concise discharge-style summary" in prompt:
                    return "Diagnoses:\n- HTN documented.\n- Troponin 9.9 documented."
                raise AssertionError(f"Unexpected prompt:\n{prompt}")

        summarizer = ClinicalSummarizer(
            ClinicalSummarizerConfig(
                enable_verification=True,
                verification_max_passes=2,
                enforce_length_budget=False,
            ),
            inference=FakeInference(),
        )

        result = summarizer.summarize_text_with_verification(
            "Assessment: HTN. No troponin elevation documented.",
        )

        self.assertTrue(result.verified)
        self.assertEqual(result.summary, "Diagnoses:\n- HTN documented.")
        self.assertEqual(len(result.passes), 2)
        self.assertEqual(
            result.passes[0].unsupported_claims,
            ("Troponin 9.9 documented.",),
        )
        self.assertIsNone(result.passes[0].entity_metrics)


if __name__ == "__main__":
    unittest.main()
