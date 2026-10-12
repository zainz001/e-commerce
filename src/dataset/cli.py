"""Command-line entry points for dataset operations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .inspector import inspect_directory
from .preview import create_previews
from .report import create_coverage_report
from .splitter import create_split
from .validator import validate_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataset",
        description="Local, non-destructive tooling for the Eastern-Wear dataset.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser(
        "inspect",
        help="Hash source files and inspect video metadata with ffprobe.",
    )
    inspect_parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory to scan. Source files are never modified.",
    )
    inspect_parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path. Without it, the report is printed to stdout.",
    )
    inspect_parser.add_argument(
        "--ffprobe",
        type=str,
        help="Optional path to the ffprobe executable; otherwise it is searched on PATH.",
    )
    inspect_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="Maximum ffprobe time per video (default: 60).",
    )
    inspect_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicitly confirm the default read-only behavior.",
    )

    validate_parser = commands.add_parser(
        "validate",
        help="Validate a JSONL manifest, paths, linkage, and rights status.",
    )
    validate_parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="JSONL manifest to validate.",
    )
    validate_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("."),
        help="Root used to resolve relative media paths (default: current directory).",
    )
    validate_parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON validation report path. Otherwise print to stdout.",
    )
    validate_parser.add_argument(
        "--require-approved",
        action="store_true",
        help="Require every non-fictional record to be approved, rights-verified, checksummed, and QA-passed.",
    )
    validate_parser.add_argument("--rights-registry", type=Path, help="Restricted JSONL rights registry for cross-checking.")

    preview_parser = commands.add_parser(
        "preview",
        help="Create timestamped derived frames and a contact sheet from a manifest.",
    )
    preview_parser.add_argument("--manifest", required=True, type=Path, help="JSONL manifest to process.")
    preview_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("."),
        help="Root used to resolve source and derived paths (default: current directory).",
    )
    preview_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/derived/previews"),
        help="Derived output directory inside dataset-root.",
    )
    preview_parser.add_argument("--sample-fps", type=float, default=3.0, help="Preview rate from 2 to 5 fps.")
    preview_parser.add_argument("--ffmpeg", type=str, help="Optional path to ffmpeg executable.")
    preview_parser.add_argument("--ffprobe", type=str, help="Optional path to ffprobe executable.")
    preview_parser.add_argument("--timeout-seconds", type=float, default=120.0)
    preview_parser.add_argument("--dry-run", action="store_true", help="Report selected frames without writing derivatives.")
    preview_parser.add_argument("--report-output", type=Path, help="Optional JSON preview report path.")
    preview_parser.add_argument("--rights-registry", type=Path, help="Optional restricted JSONL rights registry.")

    report_parser = commands.add_parser(
        "report",
        help="Create privacy-conscious coverage and QA counts from a manifest.",
    )
    report_parser.add_argument("--manifest", required=True, type=Path, help="JSONL manifest to report.")
    report_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("."),
        help="Root used to validate relative media paths (default: current directory).",
    )
    report_parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    report_parser.add_argument("--rights-registry", type=Path, help="Optional restricted JSONL rights registry.")

    split_parser = commands.add_parser(
        "split",
        help="Create deterministic person/garment/session-grouped split assignments.",
    )
    split_parser.add_argument("--manifest", required=True, type=Path, help="JSONL manifest to split.")
    split_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("."),
        help="Root used to validate relative media paths (default: current directory).",
    )
    split_parser.add_argument("--output-manifest", required=True, type=Path, help="New split JSONL path.")
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--require-approved", action="store_true")
    split_parser.add_argument("--rights-registry", type=Path, help="Restricted JSONL rights registry for cross-checking.")
    split_parser.add_argument("--report-output", type=Path, help="Optional JSON split report path.")
    return parser


def _write_report(report: dict[str, object], output: Path | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(rendered)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Wrote inspection report: {output}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        try:
            report = inspect_directory(
                args.input,
                ffprobe_executable=args.ffprobe,
                ffprobe_timeout_seconds=args.timeout_seconds,
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        _write_report(report, args.output)
        return 0

    if args.command == "validate":
        try:
            report = validate_manifest(
                args.manifest,
                dataset_root=args.dataset_root,
                require_approved=args.require_approved,
                rights_registry_path=args.rights_registry,
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        _write_report(report, args.output)
        return 1 if report["status"] == "failed" else 0

    if args.command == "preview":
        try:
            report = create_previews(
                args.manifest,
                dataset_root=args.dataset_root,
                output_root=args.output_root,
                sample_fps=args.sample_fps,
                ffmpeg_executable=args.ffmpeg,
                ffprobe_executable=args.ffprobe,
                timeout_seconds=args.timeout_seconds,
                dry_run=args.dry_run,
                rights_registry_path=args.rights_registry,
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        _write_report(report, args.report_output)
        return 1 if report["summary"]["failed"] or report["summary"]["conflicts"] else 0

    if args.command == "report":
        try:
            report = create_coverage_report(
                args.manifest,
                dataset_root=args.dataset_root,
                rights_registry_path=args.rights_registry,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        _write_report(report, args.output)
        return 1 if report["status"] == "failed" else 0

    if args.command == "split":
        try:
            report = create_split(
                args.manifest,
                dataset_root=args.dataset_root,
                output_manifest=args.output_manifest,
                seed=args.seed,
                require_approved=args.require_approved,
                rights_registry_path=args.rights_registry,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        _write_report(report, args.report_output)
        return 1 if report["status"] == "failed" else 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
