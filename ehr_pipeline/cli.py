"""Command-line entry point.

Usage:
  python -m ehr_pipeline.cli --bundle data/sample_bundle.json \\
      --notes data/sample_notes --case-id sample
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import config
from .pipeline import run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the EHR summarization pipeline against a FHIR bundle."
    )
    parser.add_argument(
        "--bundle", required=True, type=Path, help="Path to FHIR R4 Bundle JSON."
    )
    parser.add_argument(
        "--notes",
        type=Path,
        help="Optional directory of free-text notes (.txt/.md).",
    )
    parser.add_argument(
        "--case-id",
        required=True,
        help="Identifier for this run; used to namespace outputs/.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip stages whose output is newer than its inputs.",
    )
    parser.add_argument(
        "--allow-violations",
        action="store_true",
        help="Emit summary even if deterministic check finds violations.",
    )
    parser.add_argument(
        "--no-context-agent",
        action="store_true",
        help="Skip stage 4 (context agent). Useful for benchmarking.",
    )
    parser.add_argument(
        "--summary-min-ratio",
        type=float,
        default=config.SUMMARY_TARGET_COMPRESSION_MIN,
        help="Minimum summary length as a fraction of source notes.",
    )
    parser.add_argument(
        "--summary-max-ratio",
        type=float,
        default=config.SUMMARY_TARGET_COMPRESSION_MAX,
        help="Maximum summary length as a fraction of source notes.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.bundle.exists():
        print(f"Bundle not found: {args.bundle}", file=sys.stderr)
        return 2

    result = run_pipeline(
        case_id=args.case_id,
        bundle_path=args.bundle,
        notes_dir=args.notes,
        resume=args.resume,
        allow_violations=args.allow_violations,
        enable_context_agent=not args.no_context_agent,
        summary_min_ratio=args.summary_min_ratio,
        summary_max_ratio=args.summary_max_ratio,
    )

    print()
    print(f"Case:           {result.case_id}")
    print(f"Output dir:     {result.output_dir}")
    print(f"Summary:        {result.summary_path or '(blocked by check)'}")
    print(f"Audit:          {result.audit_path}")
    print(f"Check pass:     {result.check_passed}")
    print(f"Context agent:  {'on' if result.context_agent_enabled else 'off'}")
    if result.compression:
        print(
            f"Compression:    {result.compression.summary_chars}/"
            f"{result.compression.note_chars} chars "
            f"= {result.compression.achieved_ratio:.2f} "
            f"(target {args.summary_min_ratio:.2f}-{args.summary_max_ratio:.2f}, "
            f"in_band={result.compression.in_target_band})"
        )
    total = sum(t.seconds for t in result.timings)
    print(f"Total time:     {total:.2f}s")
    print("Per stage:")
    for t in result.timings:
        marker = " (cached)" if t.skipped else ""
        print(f"  {t.name:24s} {t.seconds:6.2f}s{marker}")
    return 0 if result.check_passed or args.allow_violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
