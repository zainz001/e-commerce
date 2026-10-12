"""Deterministic, leakage-aware grouped split generation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .validator import validate_manifest

SPLITS = ("train", "validation", "test")
TARGET_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def _read_records(manifest_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8-sig") as source:
        for raw_line in source:
            if raw_line.strip():
                record = json.loads(raw_line)
                if isinstance(record, dict):
                    records.append(record)
    return records


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _stable_order(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _group_records(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    union_find = _UnionFind()
    record_nodes: list[tuple[dict[str, Any], tuple[str, ...]]] = []
    for record in records:
        nodes = tuple(
            f"{field}:{record[field]}"
            for field in ("person_id", "garment_family_id", "session_id")
            if isinstance(record.get(field), str) and record[field]
        )
        if not nodes:
            continue
        for node in nodes:
            union_find.add(node)
        for node in nodes[1:]:
            union_find.union(nodes[0], node)
        record_nodes.append((record, nodes))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record, nodes in record_nodes:
        grouped[union_find.find(nodes[0])].append(record)
    return list(grouped.values())


def _assign_groups(groups: list[list[dict[str, Any]]], seed: int) -> dict[str, str]:
    assignments: dict[str, str] = {}
    counts = {split: 0 for split in SPLITS}
    ordered_groups = sorted(
        groups,
        key=lambda group: (
            -len(group),
            _stable_order(seed, "|".join(sorted(record["clip_id"] for record in group))),
        ),
    )
    for group in ordered_groups:
        group_size = len(group)
        target_score = {
            split: counts[split] / TARGET_RATIOS[split] if TARGET_RATIOS[split] else float("inf")
            for split in SPLITS
        }
        selected_split = min(
            SPLITS,
            key=lambda split: (target_score[split], _stable_order(seed, split)),
        )
        for record in group:
            assignments[record["clip_id"]] = selected_split
        counts[selected_split] += group_size
    return assignments


def _verify_no_leakage(records: list[dict[str, Any]], assignments: dict[str, str]) -> list[str]:
    issues: list[str] = []
    for field in ("person_id", "garment_family_id", "session_id"):
        values_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
        for record in records:
            split = assignments.get(record.get("clip_id"))
            value = record.get(field)
            if split in values_by_split and isinstance(value, str):
                values_by_split[split].add(value)
        for index, left_split in enumerate(SPLITS):
            for right_split in SPLITS[index + 1 :]:
                overlap = values_by_split[left_split] & values_by_split[right_split]
                if overlap:
                    issues.append(f"{field} leakage between {left_split} and {right_split}: {sorted(overlap)}")
    return issues


def create_split(
    manifest_path: Path,
    *,
    dataset_root: Path,
    output_manifest: Path,
    seed: int = 42,
    require_approved: bool = False,
    rights_registry_path: Path | None = None,
) -> dict[str, Any]:
    """Write a deterministic grouped split manifest without changing the source manifest."""

    manifest = manifest_path.expanduser().resolve()
    root = dataset_root.expanduser().resolve()
    output = output_manifest.expanduser().resolve(strict=False)
    if not manifest.is_file():
        raise ValueError(f"Manifest does not exist: {manifest_path}")
    if not root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {dataset_root}")
    try:
        output.relative_to(root)
    except ValueError as error:
        raise ValueError("output_manifest must be inside dataset_root") from error

    validation = validate_manifest(
        manifest,
        dataset_root=root,
        require_approved=require_approved,
        rights_registry_path=rights_registry_path,
    )
    report: dict[str, Any] = {
        "report_schema_version": "0.1.0",
        "tool": "dataset split",
        "tool_version": __version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_name": manifest.name,
        "output_manifest_name": output.name,
        "policy": "person-and-garment-disjoint",
        "seed": seed,
        "source_validation": validation,
        "status": "failed" if validation["status"] == "failed" else "passed",
        "warnings": [],
        "summary": {
            "source_records": 0,
            "split_records": 0,
            "fictional_records_skipped": 0,
            "unverified_rights_skipped": 0,
            "groups": 0,
            "train_records": 0,
            "validation_records": 0,
            "test_records": 0,
        },
    }
    if validation["status"] == "failed":
        report["warnings"].append("split generation blocked by manifest validation errors")
        return report

    source_records = _read_records(manifest)
    eligible: list[dict[str, Any]] = []
    for record in source_records:
        report["summary"]["source_records"] += 1
        if record.get("record_status") == "fictional_example":
            report["summary"]["fictional_records_skipped"] += 1
        elif record.get("rights_status") != "verified":
            report["summary"]["unverified_rights_skipped"] += 1
        else:
            eligible.append(record)

    groups = _group_records(eligible)
    report["summary"]["groups"] = len(groups)
    if len(groups) < 3:
        report["warnings"].append(
            "fewer than three independent groups are available; train/validation/test coverage is not meaningful"
        )
    if not eligible:
        report["warnings"].append("no rights-verified records are eligible for splitting")
        return report

    assignments = _assign_groups(groups, seed)
    leakage = _verify_no_leakage(eligible, assignments)
    if leakage:
        report["status"] = "failed"
        report["warnings"].extend(leakage)
        return report

    output_lines: list[str] = []
    for record in source_records:
        clip_id = record.get("clip_id")
        split = assignments.get(clip_id) if isinstance(clip_id, str) else None
        if split is None:
            continue
        enriched = dict(record)
        enriched["split"] = split
        output_lines.append(json.dumps(enriched, ensure_ascii=False, separators=(",", ":")))
        report["summary"]["split_records"] += 1
        report["summary"][f"{split}_records"] += 1

    rendered = "\n".join(output_lines) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8-sig") != rendered:
            report["status"] = "failed"
            report["warnings"].append("output manifest exists with different content; no overwrite performed")
            return report
        report["warnings"].append("identical output manifest already exists; skipped rewrite")
        return report

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    report["output_written"] = True
    return report
