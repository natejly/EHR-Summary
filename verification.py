"""Shared fact-checking primitives used by both summarization stages.

The default fact-check loop has three steps: claim decomposition, LLM-as-a-judge
claim verification against the source note, and summary revision. Legacy
MiniCheck and entity-extraction adapters remain available here as optional
building blocks, but the active pipeline now relies on the LLM judge only.
"""

from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from prompts import (
    CLAIM_VERDICTS_END_MARKER,
    LLM_CLAIM_JUDGE_SYSTEM_PROMPT,
    build_claim_verification_prompt,
)


def _is_accelerate_importable() -> bool:
    """Return True if `accelerate` can be imported in the current process.

    transformers gates `device_map`/`torch.device` placement on the presence
    of `accelerate` and we hit that gate when MiniCheck loads Flan-T5. This
    helper lets us pre-empt the deep traceback with a clearer message.
    """

    return importlib.util.find_spec("accelerate") is not None


_FROM_PRETRAINED_TARGETS: tuple[str, ...] = (
    "AutoModelForSeq2SeqLM",
    "AutoModelForSequenceClassification",
)


def _pick_single_device() -> tuple[str, object]:
    """Return (device_string, torch_dtype) for the best single-device placement.

    Order: MPS (Apple Silicon) > CUDA > CPU. FP16 on accelerators, FP32 on
    CPU (FP16 on CPU is much slower than FP32 in practice).
    """

    try:
        import torch
    except ImportError:
        return ("cpu", None)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return ("mps", torch.float16)
    if torch.cuda.is_available():
        return ("cuda", torch.float16)
    return ("cpu", torch.float32)


class _ForceSingleDevicePlacement:
    """Context manager that loads transformers models onto one device, no offload.

    MiniCheck's Inferencer hardcodes `device_map='auto'` for its Flan-T5
    checkpoint. On Apple Silicon (and any small-VRAM box) accelerate then
    picks a partial disk-offload plan and `from_pretrained` fails with
    "weights offloaded to the disk ... please provide an `offload_folder`".

    We side-step that by replacing `device_map='auto'` (or None) with an
    explicit single-device map (`{"": "mps"|"cuda"|"cpu"}`) and an
    accelerator-friendly dtype (FP16 on MPS/CUDA). Flan-T5-Large (~770M
    params) is ~1.5 GB in FP16, which fits comfortably in unified memory
    alongside MLX-Qwen-9B-4bit (~5 GB).

    Caller-supplied explicit device maps and dtypes are preserved.
    """

    def __init__(self) -> None:
        try:
            import transformers  # noqa: F401
        except ImportError:
            self._enabled = False
            return
        self._enabled = True
        self._patches: list[tuple[object, object]] = []
        self._device, self._dtype = _pick_single_device()

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> object:
        return self._dtype

    def __enter__(self) -> "_ForceSingleDevicePlacement":
        if not self._enabled:
            return self
        import transformers

        device = self._device
        dtype = self._dtype

        for name in _FROM_PRETRAINED_TARGETS:
            cls = getattr(transformers, name, None)
            if cls is None:
                continue
            original = cls.from_pretrained

            def _wrap(orig):
                def _patched(cls_self, *args, **kwargs):
                    if kwargs.get("device_map") in (None, "auto"):
                        kwargs["device_map"] = {"": device}
                    if dtype is not None and "torch_dtype" not in kwargs:
                        kwargs["torch_dtype"] = dtype
                    return (
                        orig.__func__(cls_self, *args, **kwargs)
                        if hasattr(orig, "__func__")
                        else orig(cls_self, *args, **kwargs)
                    )

                return _patched

            cls.from_pretrained = classmethod(_wrap(original))
            self._patches.append((cls, original))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._enabled:
            return
        for cls, original in self._patches:
            cls.from_pretrained = original
        self._patches.clear()


# Backward-compat alias; older imports referenced the prior name.
_ForceCpuDeviceMap = _ForceSingleDevicePlacement


def _ensure_nltk_punkt() -> None:
    """Make sure NLTK's punkt tokenizer corpora are present.

    MiniCheck sentence-splits incoming claims with NLTK and requires the
    `punkt_tab` resource (newer NLTK) or `punkt` (older). Download lazily
    so users don't have to run a manual `nltk.download(...)` step.
    """

    try:
        import nltk
    except ImportError:
        return

    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                # Network error / sandbox; leave it to MiniCheck to surface
                # the original LookupError with a clearer trace at use time.
                pass


# Default verifier model. Uses MiniCheck's short identifier `flan-t5-large`,
# which the package maps to the HF checkpoint `lytang/MiniCheck-Flan-T5-Large`
# (~770M params). Other valid model_name values per the upstream package:
# 'roberta-large', 'deberta-v3-large', 'flan-t5-large', 'Bespoke-MiniCheck-7B'.
DEFAULT_MINICHECK_MODEL = "flan-t5-large"


@dataclass(slots=True)
class ClaimCitation:
    """A source-note snippet that can be used to cite a supported claim."""

    snippet: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    score: float
    method: str = "token_overlap"


@dataclass(slots=True)
class ClaimVerificationResult:
    """Support score for one atomic claim."""

    claim: str
    supported: bool
    probability: float
    citations: tuple[ClaimCitation, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class EntityVerificationMetrics:
    """Entity-level overlap between the source note and a generated summary."""

    source_entity_count: int
    summary_entity_count: int
    overlap_count: int
    precision: float
    recall: float
    f1: float


@dataclass(slots=True)
class VerificationPassResult:
    """One decomposition, scoring, and optional entity-check pass."""

    pass_index: int
    summary: str
    claims: tuple[str, ...]
    claim_results: tuple[ClaimVerificationResult, ...]
    unsupported_claims: tuple[str, ...]
    entity_metrics: EntityVerificationMetrics | None
    passed: bool


class ClaimVerifier(Protocol):
    """Scores atomic claims against the source note."""

    def score_claims(
        self,
        source_note: str,
        claims: Sequence[str],
    ) -> Sequence[ClaimVerificationResult]: ...


class ClaimJudgeCompleter(Protocol):
    """Callable adapter used by the LLM claim verifier."""

    def __call__(
        self,
        prompt: str,
        *,
        system_prompt: str,
        stop_strings: Sequence[str],
    ) -> str: ...


class EntityExtractor(Protocol):
    """Extracts normalized clinical entities from free text."""

    def extract_entities(self, text: str) -> set[tuple[str, str]]: ...


class MiniCheckClaimVerifier:
    """Thin adapter around MiniCheck's `score` interface."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MINICHECK_MODEL,
        enable_prefix_caching: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        try:
            # Upstream layout: `from minicheck.minicheck import MiniCheck`.
            # Older docs sometimes show `from minicheck import MiniCheck`,
            # so try the canonical location first then fall back.
            try:
                from minicheck.minicheck import MiniCheck
            except ImportError:
                from minicheck import MiniCheck  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError(
                "MiniCheck is required for claim verification. Install it via "
                "`pip install \"minicheck @ git+https://github.com/Liyan06/"
                "MiniCheck.git@main\"` or inject a custom claim verifier."
            ) from exc

        # Pre-flight: MiniCheck loads its underlying transformer with
        # `device_map`, which transformers refuses to honor without
        # `accelerate`. Catch this here so the failure points users at the
        # exact Python that needs the package, instead of the deep
        # transformers traceback.
        if not _is_accelerate_importable() and not model_name.lower().startswith(
            "bespoke"
        ):
            import sys

            raise RuntimeError(
                "transformers requires `accelerate` to load MiniCheck's "
                f"`{model_name}` checkpoint with device_map. Install it into "
                f"this Python ({sys.executable}) with:\n"
                f"    {sys.executable} -m pip install accelerate\n"
                "and restart the Jupyter kernel so transformers re-detects it."
            )

        # Pre-fetch NLTK's punkt tokenizer; MiniCheck sentence-splits claims
        # before scoring and otherwise raises `LookupError: punkt_tab not found`
        # on first use.
        _ensure_nltk_punkt()

        init_kwargs: dict[str, object] = {"model_name": model_name}
        # Only pass kwargs the underlying constructor accepts; the prefix-cache
        # flag exists for the vLLM-backed Bespoke-MiniCheck-7B variant but is
        # not on the Flan-T5 path.
        if model_name.lower().startswith("bespoke"):
            init_kwargs["enable_prefix_caching"] = enable_prefix_caching
        if cache_dir is not None:
            init_kwargs["cache_dir"] = cache_dir

        # Wrap MiniCheck construction so its hardcoded `device_map='auto'`
        # gets rewritten to a single-device placement before transformers
        # tries to pick an offloaded layout that needs an `offload_folder`.
        # Prefers MPS > CUDA > CPU. The HF / vLLM Bespoke-MiniCheck-7B path
        # uses vLLM directly and does not hit this code path.
        if model_name.lower().startswith("bespoke"):
            self._checker = MiniCheck(**init_kwargs)
        else:
            placement = _ForceSingleDevicePlacement()
            with placement:
                self._checker = MiniCheck(**init_kwargs)
            print(
                f"[MiniCheckClaimVerifier] {model_name} loaded on "
                f"{placement.device} (dtype={placement.dtype})"
            )

    def score_claims(
        self,
        source_note: str,
        claims: Sequence[str],
    ) -> list[ClaimVerificationResult]:
        if not claims:
            return []

        # MiniCheck.score returns a 4-tuple: (pred_label, raw_prob, _, _).
        # Older versions returned 2 values; accept either shape.
        scored = self._checker.score(
            docs=[source_note] * len(claims),
            claims=list(claims),
        )
        labels, probabilities = scored[0], scored[1]
        return [
            ClaimVerificationResult(
                claim=claim,
                supported=bool(label),
                probability=float(probability),
            )
            for claim, label, probability in zip(
                claims,
                labels,
                probabilities,
                strict=False,
            )
        ]


class LLMClaimVerifier:
    """LLM-as-a-judge verifier for atomic hallucination detection."""

    def __init__(
        self,
        complete_prompt: ClaimJudgeCompleter,
        *,
        system_prompt: str = LLM_CLAIM_JUDGE_SYSTEM_PROMPT,
        stop_strings: Sequence[str] = (CLAIM_VERDICTS_END_MARKER,),
    ) -> None:
        self._complete_prompt = complete_prompt
        self._system_prompt = system_prompt
        self._stop_strings = tuple(stop_strings)

    def score_claims(
        self,
        source_note: str,
        claims: Sequence[str],
    ) -> list[ClaimVerificationResult]:
        if not claims:
            return []
        prompt = build_claim_verification_prompt(source_note, claims)
        response = self._complete_prompt(
            prompt,
            system_prompt=self._system_prompt,
            stop_strings=self._stop_strings,
        )
        return parse_claim_verdicts(response, claims)


class MedSpaCyEntityExtractor:
    """Adapter for clinical entity extraction.

    Despite the name, plain `medspacy.load()` does NOT install an NER
    component (only sentencizer + context + section detection), so calling
    `doc.ents` on it returns an empty set. To actually get entities back
    we try, in order:

    1. `scispacy` (e.g. `en_core_sci_sm` or `en_ner_bc5cdr_md`) wrapped in
       a medSpaCy pipeline. This is the recommended setup.
    2. `medspacy.load(enable=["medspacy_target_matcher"])` with a small
       built-in clinical concept rule set, if scispacy is unavailable but
       medspacy is.

    If neither path produces a working NER, `__init__` raises with a clear
    install hint rather than silently returning empty sets and bricking
    the verification loop.
    """

    _SCISPACY_MODEL_CANDIDATES: tuple[str, ...] = (
        "en_core_sci_sm",
        "en_core_sci_md",
        "en_ner_bc5cdr_md",
    )

    def __init__(self, *, model: str | None = None) -> None:
        # PyRuSH (medSpaCy's sentencizer) emits very chatty DEBUG logs via
        # loguru on every document. Silence them so the benchmark notebook
        # / runner output stays readable.
        try:
            from loguru import logger as _loguru_logger

            _loguru_logger.disable("PyRuSH")
        except ImportError:
            pass
        self._nlp = self._load_pipeline(model)

    @classmethod
    def _load_pipeline(cls, model: str | None):
        import importlib.util

        if model is not None:
            return cls._load_named_model(model)

        # Prefer scispacy if it's installed AND a model checkpoint is on disk.
        for candidate in cls._SCISPACY_MODEL_CANDIDATES:
            if importlib.util.find_spec(candidate) is not None:
                try:
                    return cls._load_named_model(candidate)
                except Exception:
                    continue

        # Fall back to medSpaCy with a minimal target-matcher rule set so
        # at least common clinical entities are surfaced. This is much
        # weaker than scispacy but better than the silent no-op default.
        try:
            return cls._load_medspacy_with_basic_rules()
        except Exception as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "Could not initialize a clinical entity extractor.\n"
                "Plain `medspacy.load()` ships without an NER component, so the "
                "verification entity gate would compare empty entity sets and "
                "always fail closed.\n\n"
                "Install one of the following into the active Python:\n"
                "  pip install scispacy && \\\n"
                "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
                "releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz\n\n"
                "Or set ENTITY_MIN_PRECISION = ENTITY_MIN_RECALL = None in the "
                "notebook config to disable the gate entirely."
            ) from exc

    @staticmethod
    def _load_named_model(name: str):
        import spacy

        nlp = spacy.load(name)
        # Optional: layer medSpaCy's section/context detectors on top so
        # entity labels are unchanged but downstream consumers still get
        # the medspacy attributes if they want them. Skipped if medspacy
        # is missing.
        try:
            from medspacy.target_matcher import TargetMatcher  # noqa: F401
            # Don't add target matcher here unless we have rules; it would
            # fail on apply otherwise. We rely on the loaded model's NER.
        except ImportError:
            pass
        return nlp

    @classmethod
    def _load_medspacy_with_basic_rules(cls):
        import medspacy
        from medspacy.target_matcher import TargetRule

        nlp = medspacy.load(enable=["medspacy_target_matcher"])
        target_matcher = nlp.get_pipe("medspacy_target_matcher")
        # A pragmatic seed rule set covering common discharge-summary
        # entities. Not exhaustive; users with scispacy installed get
        # much better recall via path #1 above.
        seed_rules = [
            TargetRule(literal=lit, category=cat)
            for cat, literals in cls._BASIC_CONCEPT_RULES.items()
            for lit in literals
        ]
        target_matcher.add(seed_rules)
        return nlp

    _BASIC_CONCEPT_RULES: dict[str, tuple[str, ...]] = {
        "PROBLEM": (
            "hypertension", "htn", "diabetes", "dm", "dm2", "type 2 diabetes",
            "copd", "asthma", "chf", "heart failure", "cad",
            "coronary artery disease", "atrial fibrillation", "afib", "a-fib",
            "stroke", "cva", "tia", "pneumonia", "uti",
            "urinary tract infection", "sepsis", "septic shock",
            "acute kidney injury", "aki", "ckd", "chronic kidney disease",
            "esrd", "cirrhosis", "hcv", "hiv", "cancer", "malignancy",
            "myocardial infarction", "mi", "stemi", "nstemi",
            "pulmonary embolism", "pe", "dvt", "deep vein thrombosis",
            "pneumothorax", "ptx", "respiratory failure", "hypoxia",
            "encephalopathy", "delirium", "anemia", "thrombocytopenia",
            "leukocytosis", "neutropenia", "rib fracture", "fracture",
            "emphysema", "aspiration pneumonia",
        ),
        "TREATMENT": (
            "lactulose", "rifaximin", "vancomycin", "vanco", "zosyn",
            "piperacillin-tazobactam", "ceftriaxone", "metronidazole",
            "metoprolol", "lisinopril", "amlodipine", "hydrochlorothiazide",
            "furosemide", "lasix", "spironolactone", "warfarin", "coumadin",
            "heparin", "lovenox", "enoxaparin", "aspirin", "asa", "plavix",
            "clopidogrel", "atorvastatin", "simvastatin", "metformin",
            "insulin", "albuterol", "ipratropium", "duonebs", "nebs",
            "prednisone", "solu-medrol", "methylprednisolone", "tylenol",
            "acetaminophen", "tordol", "ketorolac", "dilaudid", "hydromorphone",
            "neurontin", "gabapentin", "lidocaine patch", "senna", "colace",
            "docusate", "midodrine", "octreotide", "pca",
        ),
        "TEST": (
            "cbc", "bmp", "cmp", "lfts", "troponin", "bnp", "lactate",
            "abg", "vbg", "ekg", "ecg", "cxr", "chest x-ray", "ct chest",
            "ct head", "ct abdomen", "mri", "ultrasound", "echocardiogram",
            "echo", "urinalysis", "ua", "blood culture", "urine culture",
        ),
    }

    def extract_entities(self, text: str) -> set[tuple[str, str]]:
        if not text or not text.strip():
            return set()
        doc = self._nlp(text)
        return {
            (entity.text.strip().lower(), entity.label_.strip().upper())
            for entity in doc.ents
            if entity.text and entity.text.strip()
        }


class _CallableEntityExtractor:
    """Wraps either a callable `(text) -> entities` or an EntityExtractor."""

    def __init__(
        self,
        extractor: EntityExtractor | Callable[[str], set[tuple[str, str]]],
    ) -> None:
        self._extractor = extractor

    def extract_entities(self, text: str) -> set[tuple[str, str]]:
        if hasattr(self._extractor, "extract_entities"):
            entities = self._extractor.extract_entities(text)
        else:
            entities = self._extractor(text)
        return {
            (str(entity_text).strip().lower(), str(label).strip().upper())
            for entity_text, label in entities
            if str(entity_text).strip()
        }


def normalize_claim_result(
    result: ClaimVerificationResult | tuple[str, bool, float] | object,
) -> ClaimVerificationResult:
    """Coerce verifier output into a ClaimVerificationResult.

    Accepts either an already-typed ClaimVerificationResult or a 3-tuple of
    (claim, supported, probability) from MiniCheck-style scorers that return
    raw tuples.
    """

    if isinstance(result, ClaimVerificationResult):
        return result
    if isinstance(result, tuple) and len(result) == 3:
        claim, supported, probability = result
        return ClaimVerificationResult(
            claim=str(claim),
            supported=bool(supported),
            probability=float(probability),
        )
    raise TypeError("Claim verifier returned an unsupported result format.")


def score_claims_via_adapter(
    verifier: object,
    source_note_text: str,
    claims: Sequence[str],
) -> list[ClaimVerificationResult]:
    """Call either the `score_claims` or the raw MiniCheck `score` interface.

    Centralized so both the clinical and patient-friendly loops can plug in
    a verifier object that implements either method without each call site
    duplicating the dispatch logic.
    """

    if not claims:
        return []
    if hasattr(verifier, "score_claims"):
        scored_claims = verifier.score_claims(source_note_text, claims)
        return _attach_citations(
            [normalize_claim_result(result) for result in scored_claims],
            source_note_text,
        )
    if hasattr(verifier, "score"):
        # MiniCheck.score returns (labels, raw_prob, _, _) on >=0.2 and
        # (labels, raw_prob) on older versions; index defensively.
        scored = verifier.score(
            docs=[source_note_text] * len(claims),
            claims=list(claims),
        )
        labels, probabilities = scored[0], scored[1]
        return _attach_citations([
            ClaimVerificationResult(
                claim=claim,
                supported=bool(label),
                probability=float(probability),
            )
            for claim, label, probability in zip(
                claims,
                labels,
                probabilities,
                strict=False,
            )
        ], source_note_text)
    raise TypeError(
        "Claim verifier must expose `score_claims(source_note, claims)` or "
        "`score(docs=..., claims=...)`."
    )


_CITATION_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[./-][A-Za-z0-9]+)*")
_CLAIM_CITATION_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
    "documented",
    "noted",
    "not",
    "patient",
    "pt",
    "shows",
    "showed",
    "found",
    "had",
    "has",
}


def _attach_citations(
    results: Sequence[ClaimVerificationResult],
    source_note_text: str,
) -> list[ClaimVerificationResult]:
    return [
        ClaimVerificationResult(
            claim=result.claim,
            supported=result.supported,
            probability=result.probability,
            citations=(
                result.citations
                if result.citations
                else _find_claim_citations(source_note_text, result.claim)
            )
            if result.supported
            else tuple(),
        )
        for result in results
    ]


def _find_claim_citations(
    source_note_text: str,
    claim: str,
    *,
    max_citations: int = 2,
    min_score: float = 0.35,
) -> tuple[ClaimCitation, ...]:
    claim_tokens = _citation_tokens(claim)
    if not claim_tokens or not source_note_text.strip():
        return tuple()

    candidates: list[ClaimCitation] = []
    for snippet, start_char, end_char, start_line, end_line in _iter_citation_spans(
        source_note_text
    ):
        snippet_tokens = _citation_tokens(snippet)
        if not snippet_tokens:
            continue
        shared_tokens = claim_tokens & snippet_tokens
        if not shared_tokens:
            continue
        recall = len(shared_tokens) / len(claim_tokens)
        precision = len(shared_tokens) / len(snippet_tokens)
        score = (0.7 * recall) + (0.3 * precision)
        if any(token.isdigit() for token in shared_tokens):
            score += 0.1
        if score < min_score:
            continue
        candidates.append(
            ClaimCitation(
                snippet=snippet,
                start_char=start_char,
                end_char=end_char,
                start_line=start_line,
                end_line=end_line,
                score=min(score, 1.0),
            )
        )

    candidates.sort(
        key=lambda item: (-item.score, item.start_char, -(item.end_char - item.start_char))
    )
    deduped: list[ClaimCitation] = []
    seen: set[tuple[int, int]] = set()
    for candidate in candidates:
        key = (candidate.start_char, candidate.end_char)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
        if len(deduped) >= max_citations:
            break
    return tuple(deduped)


def _iter_citation_spans(
    text: str,
) -> list[tuple[str, int, int, int, int]]:
    spans: list[tuple[str, int, int, int, int]] = []
    offset = 0
    for line_number, raw_line in enumerate(text.splitlines(keepends=True), start=1):
        line_without_newline = raw_line.rstrip("\r\n")
        stripped_line = line_without_newline.strip()
        if not stripped_line:
            offset += len(raw_line)
            continue

        leading_ws = len(line_without_newline) - len(line_without_newline.lstrip())
        line_start = offset + leading_ws
        line_end = line_start + len(stripped_line)
        spans.append((stripped_line, line_start, line_end, line_number, line_number))

        for match in re.finditer(r"[^.;!?]+[.;!?]?", stripped_line):
            sentence = match.group().strip()
            if not sentence or sentence == stripped_line:
                continue
            sentence_start = line_start + match.start() + (
                len(match.group()) - len(match.group().lstrip())
            )
            sentence_end = sentence_start + len(sentence)
            spans.append(
                (sentence, sentence_start, sentence_end, line_number, line_number)
            )
        offset += len(raw_line)
    return spans


def _citation_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _CITATION_WORD_RE.findall(text)
        if token and token.lower() not in _CLAIM_CITATION_STOPWORDS
    }


def compute_entity_metrics(
    extractor: EntityExtractor | None,
    source_note_text: str,
    summary: str,
) -> EntityVerificationMetrics | None:
    """Compute entity-level precision/recall/F1 between source and summary.

    Returns None when:
    - no extractor is wired up (callers haven't opted into the entity gate), or
    - the extractor returns zero entities for the source note. The latter
      indicates the extractor is non-functional for this input (e.g. plain
      `medspacy.load()` with no NER component attached, or a non-clinical
      pipeline). Failing OPEN in that case is the only correct policy: an
      F1 of 0 from 0/0 isn't evidence of a bad summary, and treating it as
      a gate failure causes the verification loop to over-revise and
      degrade scores.
    """

    if extractor is None:
        return None

    source_entities = extractor.extract_entities(source_note_text)
    if not source_entities:
        return None

    summary_entities = extractor.extract_entities(summary)
    overlap_entities = source_entities & summary_entities
    precision = _safe_divide(len(overlap_entities), len(summary_entities))
    recall = _safe_divide(len(overlap_entities), len(source_entities))
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return EntityVerificationMetrics(
        source_entity_count=len(source_entities),
        summary_entity_count=len(summary_entities),
        overlap_count=len(overlap_entities),
        precision=precision,
        recall=recall,
        f1=f1,
    )


def parse_atomic_claims(text: str) -> list[str]:
    """Parse the LLM's claim-decomposition output into a list of claims.

    Tolerates numbered lists ("1.", "1)", "1 -"), bulleted lists ("-", "*",
    "•"), and trailing whitespace. Empty lines are skipped.
    """

    claims: list[str] = []
    for raw_line in text.splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line:
            continue
        if stripped_line.startswith("<<END_") and stripped_line.endswith(">>"):
            continue
        cleaned_line = re.sub(r"^\d+[\).\s-]*", "", stripped_line)
        cleaned_line = re.sub(r"^[-*•]\s*", "", cleaned_line).strip()
        if cleaned_line:
            claims.append(cleaned_line)
    return claims


def parse_claim_verdicts(
    text: str,
    claims: Sequence[str],
) -> list[ClaimVerificationResult]:
    """Parse JSON-lines claim verdicts emitted by the LLM judge.

    Missing or malformed verdicts fail closed: the corresponding claim is
    treated as unsupported with zero confidence so hallucinations are more
    likely to be caught than silently passed through.
    """

    verdicts_by_index: dict[int, ClaimVerificationResult] = {}
    for raw_line in text.splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line or stripped_line == CLAIM_VERDICTS_END_MARKER:
            continue
        try:
            payload = json.loads(stripped_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        index = payload.get("index")
        if not isinstance(index, int):
            continue
        if index < 1 or index > len(claims):
            continue
        confidence = payload.get("confidence", 0.0)
        try:
            probability = float(confidence)
        except (TypeError, ValueError):
            probability = 0.0
        probability = max(0.0, min(1.0, probability))
        verdicts_by_index[index] = ClaimVerificationResult(
            claim=claims[index - 1],
            supported=bool(payload.get("supported", False)),
            probability=probability,
        )

    return [
        verdicts_by_index.get(
            index,
            ClaimVerificationResult(
                claim=claim,
                supported=False,
                probability=0.0,
            ),
        )
        for index, claim in enumerate(claims, start=1)
    ]


def fallback_claim_split(text: str) -> list[str]:
    """Sentence-split a summary as a last resort when claim parsing fails."""

    cleaned_text = text.strip()
    if not cleaned_text:
        return []
    fragments = re.split(r"(?<=[.!?])\s+|\n+", cleaned_text)
    claims = [fragment.strip(" -\t") for fragment in fragments if fragment.strip(" -\t")]
    return claims or [cleaned_text]


def format_entity_feedback(
    entity_metrics: EntityVerificationMetrics | None,
) -> str | None:
    """Render an entity-overlap summary suitable for inclusion in a revision prompt."""

    if entity_metrics is None:
        return None
    return (
        f"- Entity precision: {entity_metrics.precision:.2f}\n"
        f"- Entity recall: {entity_metrics.recall:.2f}\n"
        f"- Entity F1: {entity_metrics.f1:.2f}\n"
        f"- Overlap: {entity_metrics.overlap_count} shared entities across "
        f"{entity_metrics.summary_entity_count} summary entities and "
        f"{entity_metrics.source_entity_count} source entities."
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


__all__ = [
    "DEFAULT_MINICHECK_MODEL",
    "ClaimCitation",
    "ClaimVerificationResult",
    "EntityVerificationMetrics",
    "VerificationPassResult",
    "ClaimVerifier",
    "LLMClaimVerifier",
    "EntityExtractor",
    "MiniCheckClaimVerifier",
    "MedSpaCyEntityExtractor",
    "normalize_claim_result",
    "score_claims_via_adapter",
    "compute_entity_metrics",
    "parse_atomic_claims",
    "parse_claim_verdicts",
    "fallback_claim_split",
    "format_entity_feedback",
]
