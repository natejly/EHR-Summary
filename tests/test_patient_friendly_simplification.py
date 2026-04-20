from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from patient_friendly_simplification import (
    ClaimVerificationResult,
    EntityVerificationMetrics,
    PatientFriendlySimplifier,
    PatientFriendlySimplifierConfig,
    build_arg_parser,
    load_summary_text,
)
from prompts import (
    build_patient_friendly_claim_decomposition_prompt,
    build_patient_friendly_revision_prompt,
    build_patient_friendly_simplification_prompt,
)


class FakeInference:
    def __init__(self) -> None:
        self.complete_calls = []

    def complete(self, prompt: str, **kwargs) -> str:
        self.complete_calls.append((prompt, kwargs))
        return "patient-friendly summary"


class ScenarioInference:
    def __init__(self) -> None:
        self.complete_calls = []

    def complete(self, prompt: str, **kwargs) -> str:
        self.complete_calls.append((prompt, kwargs))
        if "Break the following patient-friendly clinical summary" in prompt:
            if "You had pneumonia.\nYou also had cancer." in prompt:
                return "You had pneumonia.\nYou also had cancer."
            if "You had pneumonia." in prompt:
                return "You had pneumonia."
        if "Revise the patient-friendly clinical summary below" in prompt:
            return "You had pneumonia."
        return "You had pneumonia.\nYou also had cancer."


class FakeClaimVerifier:
    def __init__(self, result_batches) -> None:
        self.result_batches = list(result_batches)
        self.calls = []

    def score_claims(self, source_note: str, claims) -> list[ClaimVerificationResult]:
        self.calls.append((source_note, tuple(claims)))
        return self.result_batches.pop(0)


class FakeEntityExtractor:
    def __init__(self, entities_by_text) -> None:
        self.entities_by_text = entities_by_text

    def extract_entities(self, text: str) -> set[tuple[str, str]]:
        return self.entities_by_text[text]


class PatientFriendlySimplificationTests(unittest.TestCase):
    def test_prompt_contains_patient_friendly_requirements(self) -> None:
        prompt = build_patient_friendly_simplification_prompt(
            "Primary diagnosis: Heart failure exacerbation.",
            audience="caregiver",
            patient_id="enc-002",
        )

        self.assertIn("Rewrite the clinical summary below", prompt)
        self.assertIn("Target audience: caregiver", prompt)
        self.assertIn("Patient ID: enc-002", prompt)
        self.assertIn("Do not add new facts.", prompt)
        self.assertIn("<<END_OF_PATIENT_FRIENDLY_SUMMARY>>", prompt)

    def test_fact_check_prompts_include_expected_context(self) -> None:
        decomposition_prompt = build_patient_friendly_claim_decomposition_prompt(
            "You were admitted for pneumonia.",
            patient_id="enc-002",
        )
        revision_prompt = build_patient_friendly_revision_prompt(
            "Primary diagnosis: Pneumonia.",
            "You were admitted for pneumonia and cancer.",
            "Source note: pneumonia only.",
            ["You were admitted for cancer."],
            audience="patient",
            patient_id="enc-002",
            entity_feedback="- Entity precision: 0.50",
        )

        self.assertIn("atomic factual claims", decomposition_prompt)
        self.assertIn("<<END_OF_ATOMIC_CLAIMS>>", decomposition_prompt)
        self.assertIn("Unsupported claims to fix:", revision_prompt)
        self.assertIn("You were admitted for cancer.", revision_prompt)
        self.assertIn("Entity-level cross-check feedback:", revision_prompt)
        self.assertIn("Target audience: patient", revision_prompt)

    def test_load_summary_text_reads_non_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.txt"
            summary_path.write_text("Structured Clinical Summary", encoding="utf-8")

            result = load_summary_text(summary_path)

        self.assertEqual(result, "Structured Clinical Summary")

    def test_simplifier_uses_configured_system_prompt(self) -> None:
        fake_inference = FakeInference()
        simplifier = PatientFriendlySimplifier(
            PatientFriendlySimplifierConfig(system_prompt="patient simplifier prompt"),
            inference=fake_inference,
        )

        result = simplifier.simplify_text(
            "Brief Hospital Course: Admitted for heart failure.",
            audience="patient",
        )

        self.assertEqual(result, "patient-friendly summary")
        self.assertEqual(len(fake_inference.complete_calls), 1)
        prompt, kwargs = fake_inference.complete_calls[0]
        self.assertIn("Target audience: patient", prompt)
        self.assertEqual(kwargs["system_prompt"], "patient simplifier prompt")
        self.assertEqual(
            kwargs["stop_strings"],
            ("<<END_OF_PATIENT_FRIENDLY_SUMMARY>>",),
        )

    def test_arg_parser_accepts_render_markdown_flag(self) -> None:
        args = build_arg_parser().parse_args(["--text", "summary", "--render-markdown"])

        self.assertTrue(args.render_markdown)

    def test_simplifier_revises_unsupported_claims_during_verification(self) -> None:
        inference = ScenarioInference()
        verifier = FakeClaimVerifier(
            [
                [
                    ClaimVerificationResult(
                        claim="You had pneumonia.",
                        supported=True,
                        probability=0.97,
                    ),
                    ClaimVerificationResult(
                        claim="You also had cancer.",
                        supported=False,
                        probability=0.04,
                    ),
                ],
                [
                    ClaimVerificationResult(
                        claim="You had pneumonia.",
                        supported=True,
                        probability=0.98,
                    )
                ],
            ]
        )
        simplifier = PatientFriendlySimplifier(
            PatientFriendlySimplifierConfig(verification_max_passes=2),
            inference=inference,
            claim_verifier=verifier,
        )

        result = simplifier.simplify_text_with_verification(
            "Primary diagnosis: pneumonia.",
            source_note_text="Discharge summary: treated for pneumonia.",
            audience="patient",
        )

        self.assertTrue(result.verified)
        self.assertEqual(result.summary, "You had pneumonia.")
        self.assertEqual(len(result.passes), 2)
        self.assertEqual(
            result.passes[0].unsupported_claims,
            ("You also had cancer.",),
        )
        revision_prompt = inference.complete_calls[2][0]
        self.assertIn("Unsupported claims to fix:", revision_prompt)
        self.assertIn("You also had cancer.", revision_prompt)

    def test_entity_thresholds_can_fail_an_otherwise_supported_summary(self) -> None:
        fake_inference = FakeInference()
        verifier = FakeClaimVerifier(
            [
                [
                    ClaimVerificationResult(
                        claim="You had pneumonia.",
                        supported=True,
                        probability=0.93,
                    )
                ]
            ]
        )
        entity_extractor = FakeEntityExtractor(
            {
                "Source note: pneumonia and amoxicillin.": {
                    ("pneumonia", "PROBLEM"),
                    ("amoxicillin", "MEDICATION"),
                },
                "You had pneumonia.": {
                    ("pneumonia", "PROBLEM"),
                    ("metformin", "MEDICATION"),
                },
            }
        )
        simplifier = PatientFriendlySimplifier(
            PatientFriendlySimplifierConfig(entity_min_precision=0.75),
            inference=fake_inference,
            claim_verifier=verifier,
            entity_extractor=entity_extractor,
        )

        result = simplifier.verify_generated_summary(
            "You had pneumonia.",
            source_note_text="Source note: pneumonia and amoxicillin.",
        )

        self.assertFalse(result.passed)
        self.assertIsNotNone(result.entity_metrics)
        self.assertEqual(
            result.entity_metrics,
            EntityVerificationMetrics(
                source_entity_count=2,
                summary_entity_count=2,
                overlap_count=1,
                precision=0.5,
                recall=0.5,
                f1=0.5,
            ),
        )
