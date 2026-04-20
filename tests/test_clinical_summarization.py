from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clinical_summarization import (
    DEFAULT_CLINICAL_STOP_STRINGS,
    ClinicalNote,
    ClinicalSummarizer,
    ClinicalSummarizerConfig,
    VerifiedClinicalSummary,
    _is_truncated,
    _looks_truncated,
    build_arg_parser,
    load_clinical_notes,
)
from mlx_qwen_inference import CompletionResult
from prompts import CLINICAL_SUMMARY_END_MARKER, build_clinical_summary_prompt
from verification import ClaimVerificationResult


class FakeInference:
    def __init__(self) -> None:
        self.complete_calls = []

    def complete(self, prompt: str, **kwargs) -> str:
        self.complete_calls.append((prompt, kwargs))
        return "summary"


class VerificationFakeInference:
    """Inference stub that returns plausible outputs per pipeline stage.

    - Initial summarization returns a short clinical-note-style summary.
    - Claim decomposition returns one claim per line ending with the marker.
    - Revision is never reached because the FakeClaimVerifier marks every
      claim as supported on the first pass.
    """

    def __init__(self) -> None:
        self.complete_calls = []

    def complete(self, prompt: str, **kwargs) -> str:
        self.complete_calls.append((prompt, kwargs))
        if "atomic factual claims" in prompt:
            return (
                "Patient admitted with HTN.\n"
                "Cr 2.4 on admission, improved to 1.6 with IVF.\n"
                "<<END_OF_CLINICAL_ATOMIC_CLAIMS>>"
            )
        if "Revise the STRUCTURED clinical summary" in prompt:
            return "Patient admitted with HTN."
        return (
            "Diagnoses: HTN.\n"
            "Course: AKI on CKD; Cr 2.4 -> 1.6 with IVF."
        )


class AlwaysSupportedVerifier:
    def __init__(self) -> None:
        self.calls = []

    def score_claims(self, source_note, claims):
        self.calls.append((source_note, tuple(claims)))
        return [
            ClaimVerificationResult(claim=c, supported=True, probability=0.99)
            for c in claims
        ]


class AlwaysFailingVerifier:
    def __init__(self) -> None:
        self.calls = []

    def score_claims(self, source_note, claims):
        self.calls.append((source_note, tuple(claims)))
        return [
            ClaimVerificationResult(claim=c, supported=False, probability=0.05)
            for c in claims
        ]


class ClinicalSummarizationTests(unittest.TestCase):
    def test_prompt_contains_required_sections(self) -> None:
        prompt = build_clinical_summary_prompt(
            [
                "\n".join(
                    [
                        "[Note 1]",
                        "Note ID: 123",
                        "Note Type: discharge summary",
                        "Source: note-1.txt",
                        "Content:",
                        "Admitted with hypoxia and treated with IV diuresis.",
                    ]
                )
            ],
            patient_id="enc-001",
        )

        self.assertIn("Brief Hospital Course:", prompt)
        self.assertIn("Diagnoses (primary + secondary):", prompt)
        self.assertIn("Medication changes (new / changed / stopped):", prompt)
        self.assertIn("Patient ID: enc-001", prompt)
        self.assertIn("Note Type: discharge summary", prompt)
        self.assertIn(CLINICAL_SUMMARY_END_MARKER, prompt)
        # The prompt should explicitly tell the model NOT to emit a
        # "Not documented" placeholder for empty sections.
        self.assertIn("Omit empty sections", prompt)

    def test_load_clinical_notes_filters_suffixes_and_skips_empty_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a_note.txt").write_text("First note", encoding="utf-8")
            (root / "b_note.md").write_text("Second note", encoding="utf-8")
            (root / "empty.txt").write_text("   ", encoding="utf-8")
            (root / "ignored.json").write_text('{"note": "ignored"}', encoding="utf-8")

            notes = load_clinical_notes(root, suffixes=["txt", ".md"])

        self.assertEqual([note.note_id for note in notes], ["a_note", "b_note"])

    def test_summarizer_uses_configured_system_prompt(self) -> None:
        fake_inference = FakeInference()
        summarizer = ClinicalSummarizer(
            ClinicalSummarizerConfig(
                system_prompt="clinical system prompt",
                # Ends with a period so the truncation heuristic does not
                # trigger a retry under the fake inference client.
                # (FakeInference returns the literal "summary".)
                retry_on_truncation=False,
            ),
            inference=fake_inference,
        )

        result = summarizer.summarize_text(
            "Discharge home after improvement in volume overload."
        )

        self.assertEqual(result, "summary")
        self.assertEqual(len(fake_inference.complete_calls), 1)
        prompt, kwargs = fake_inference.complete_calls[0]
        self.assertIn("Diagnoses (primary + secondary):", prompt)
        self.assertEqual(kwargs["system_prompt"], "clinical system prompt")
        self.assertEqual(kwargs["stop_strings"], DEFAULT_CLINICAL_STOP_STRINGS)
        self.assertIn(CLINICAL_SUMMARY_END_MARKER, kwargs["stop_strings"])

    def test_arg_parser_accepts_render_markdown_flag(self) -> None:
        args = build_arg_parser().parse_args(["--text", "note", "--render-markdown"])

        self.assertTrue(args.render_markdown)

    def test_looks_truncated_detects_midword_endings(self) -> None:
        self.assertTrue(_looks_truncated("- Diagnoses: HTN, DM2, AKI st"))
        self.assertTrue(_looks_truncated(""))
        self.assertTrue(_looks_truncated("   \n\n  "))
        self.assertTrue(_looks_truncated("trailing-fragment"))

    def test_looks_truncated_accepts_complete_lines(self) -> None:
        self.assertFalse(
            _looks_truncated(
                "- Diagnoses: HTN, DM2, AKI stage 2.\n"
                "- Plan: discharge home, follow-up in 2 weeks."
            )
        )
        self.assertFalse(_looks_truncated("Final line ends with a colon:"))

    def test_is_truncated_trusts_length_finish_reason(self) -> None:
        # Length finish_reason ⇒ truncated regardless of trailing punctuation.
        result = CompletionResult(
            text="- Plan: discharge home.",
            finish_reason="length",
            generation_tokens=384,
        )
        self.assertTrue(_is_truncated(result))

    def test_is_truncated_trusts_clean_stop_signals(self) -> None:
        # Even a weird-looking trailing line is fine when generation stopped
        # cleanly via a stop string or EOS.
        for reason in ("stop", "stop_string"):
            result = CompletionResult(
                text="- Diagnoses: HTN, DM2, AKI st",
                finish_reason=reason,
                generation_tokens=200,
            )
            self.assertFalse(
                _is_truncated(result),
                msg=f"expected not-truncated for finish_reason={reason!r}",
            )

    def test_is_truncated_falls_back_to_heuristic_when_reason_missing(self) -> None:
        # No finish_reason ⇒ delegate to text-shape heuristic.
        truncated = CompletionResult(
            text="- Diagnoses: HTN, DM2, AKI st",
            finish_reason=None,
            generation_tokens=0,
        )
        self.assertTrue(_is_truncated(truncated))
        complete = CompletionResult(
            text="- Plan: discharge home.",
            finish_reason=None,
            generation_tokens=0,
        )
        self.assertFalse(_is_truncated(complete))


class ClinicalSummarizerVerificationTests(unittest.TestCase):
    """Smoke tests for the agentic fact-check loop with mocked dependencies."""

    def test_verification_terminates_after_one_pass_when_all_supported(self) -> None:
        inference = VerificationFakeInference()
        verifier = AlwaysSupportedVerifier()
        summarizer = ClinicalSummarizer(
            ClinicalSummarizerConfig(
                enable_verification=True,
                verification_max_passes=3,
                retry_on_truncation=False,
                enforce_length_budget=False,
            ),
            inference=inference,
            claim_verifier=verifier,
        )

        result = summarizer.summarize_text_with_verification(
            "Brief Hospital Course: HTN; AKI on CKD with Cr improvement on IVF."
        )

        self.assertIsInstance(result, VerifiedClinicalSummary)
        self.assertTrue(result.verified)
        self.assertEqual(len(result.passes), 1)
        self.assertEqual(result.summary, result.initial_summary)
        # First call: initial summary; second call: claim decomposition.
        # No revision call should have happened.
        revision_calls = [
            call for call in inference.complete_calls
            if "Revise the STRUCTURED clinical summary" in call[0]
        ]
        self.assertEqual(revision_calls, [])
        # Verifier should have been called exactly once with the decomposed claims.
        self.assertEqual(len(verifier.calls), 1)

    def test_verification_revises_on_unsupported_claims(self) -> None:
        inference = VerificationFakeInference()
        verifier = AlwaysFailingVerifier()
        summarizer = ClinicalSummarizer(
            ClinicalSummarizerConfig(
                enable_verification=True,
                verification_max_passes=2,
                retry_on_truncation=False,
                enforce_length_budget=False,
            ),
            inference=inference,
            claim_verifier=verifier,
        )

        result = summarizer.summarize_text_with_verification(
            "Brief Hospital Course: HTN."
        )

        # Loop runs both passes (both fail under AlwaysFailingVerifier).
        self.assertFalse(result.verified)
        self.assertEqual(len(result.passes), 2)
        # The first pass should have non-empty unsupported claims and the
        # second-pass summary must be the revised text from the inference stub.
        self.assertGreater(len(result.passes[0].unsupported_claims), 0)
        revision_calls = [
            call for call in inference.complete_calls
            if "Revise the STRUCTURED clinical summary" in call[0]
        ]
        self.assertEqual(len(revision_calls), 1)


if __name__ == "__main__":
    unittest.main()
