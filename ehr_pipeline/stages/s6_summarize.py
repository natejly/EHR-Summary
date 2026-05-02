"""Stage 6: clinician-facing Markdown summary generation.

Sees ONLY the verified fact sheet -- never the original notes or any
unverified claims -- so the surface area for hallucination is bounded by
what stage 5 lets through.

The summary is constrained to a target compression ratio (default 0.20-0.30
of the source-note length in characters). The model is told the explicit
character budget; we also record the achieved ratio in the orchestrator's
audit JSON so benchmarking can report it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .. import config
from ..ollama_client import chat_text
from ..prompts import S6_SUMMARY_TEMPLATE
from ..schemas import FactSheet

log = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = S6_SUMMARY_TEMPLATE


@dataclass(frozen=True)
class SummaryResult:
    markdown: str
    note_chars: int
    summary_chars: int
    target_chars_min: int
    target_chars_max: int
    achieved_ratio: float
    in_target_band: bool


def _format_fact_sheet(fs: FactSheet) -> str:
    lines: list[str] = [f"case_id: {fs.case_id}", ""]
    for section, entries in fs.sections.items():
        lines.append(f"[{section}]")
        if not entries:
            lines.append("  (none)")
        for entry in entries:
            ids = "".join(f"[E:{eid.removeprefix('E:')}]" for eid in entry.evidence_ids)
            lines.append(f"  - {entry.text} {ids}")
        lines.append("")
    return "\n".join(lines)


def run(
    *,
    fact_sheet: FactSheet,
    output_dir: Path,
    note_chars: int,
    min_ratio: float = config.SUMMARY_TARGET_COMPRESSION_MIN,
    max_ratio: float = config.SUMMARY_TARGET_COMPRESSION_MAX,
) -> SummaryResult:
    log.info("Stage 6: generating summary markdown")
    safe_note_chars = max(int(note_chars), 1)
    min_chars = max(int(safe_note_chars * min_ratio), 200)
    max_chars = max(int(safe_note_chars * max_ratio), min_chars + 100)

    system_prompt = S6_SUMMARY_TEMPLATE.format(
        note_chars=safe_note_chars,
        min_chars=min_chars,
        max_chars=max_chars,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
    )

    user = (
        "Write the clinician summary using only this fact sheet, respecting "
        f"the {min_chars}-{max_chars} character target.\n\n"
        + _format_fact_sheet(fact_sheet)
    )
    markdown = chat_text(
        model=config.MODELS.summary_generation,
        system=system_prompt,
        user=user,
        temperature=0.1,
    ).strip()

    out_path = output_dir / "summary.md"
    out_path.write_text(markdown + "\n", encoding="utf-8")

    summary_chars = len(markdown)
    achieved_ratio = summary_chars / safe_note_chars
    in_band = min_ratio <= achieved_ratio <= max_ratio
    log.info(
        "  wrote %s (%d chars, ratio=%.2f, target=%.2f-%.2f, in_band=%s)",
        out_path,
        summary_chars,
        achieved_ratio,
        min_ratio,
        max_ratio,
        in_band,
    )

    return SummaryResult(
        markdown=markdown,
        note_chars=safe_note_chars,
        summary_chars=summary_chars,
        target_chars_min=min_chars,
        target_chars_max=max_chars,
        achieved_ratio=achieved_ratio,
        in_target_band=in_band,
    )
