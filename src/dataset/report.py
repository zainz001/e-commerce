"""Coverage and quality reporting for dataset manifests."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .validator import validate_manifest

RECOMMENDED_PILOT_POSES = ("front", "left_turn", "right_turn")


def _read_records(manifest_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8-sig") as source:
        for raw_line in source:
            if raw_line.strip():
                record = json.loads(raw_line)
                if isinstance(record, dict):
                    records.append(record)
    return records


def _sorted_counts(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def create_coverage_report(
    manifest_path: Path,
    *,
    dataset_root: Path,
    rights_registry_path: Path | None = None,
) -> dict[str, Any]:
    """Create a privacy-conscious coverage report using opaque IDs only."""

    manifest = manifest_path.expanduser().resolve()
    root = dataset_root.expanduser().resolve()
    if not manifest.is_file():
        raise ValueError(f"Manifest does not exist: {manifest_path}")
    if not root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {dataset_root}")

    validation = validate_manifest(
        manifest,
        dataset_root=root,
        rights_registry_path=rights_registry_path,
    )
    records = [record for record in _read_records(manifest) if record.get("record_status") != "fictional_example"]
    by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_garment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if isinstance(record.get("person_id"), str):
            by_person[record["person_id"]].append(record)
        if isinstance(record.get("garment_family_id"), str):
            by_garment[record["garment_family_id"]].append(record)

    pose_values = [pose for record in records for pose in record.get("pose_tags", []) if isinstance(pose, str)]
    lighting_values = [
        record["lighting_group"]
        for record in records
        if isinstance(record.get("lighting_group"), str) and record["lighting_group"]
    ]
    occlusion_values = [tag for record in records for tag in record.get("occlusion_tags", []) if isinstance(tag, str)]

    person_coverage = {}
    for person_id, person_records in sorted(by_person.items()):
        person_coverage[person_id] = {
            "record_count": len(person_records),
            "session_count": len({record.get("session_id") for record in person_records}),
            "garment_family_count": len({record.get("garment_family_id") for record in person_records}),
            "pose_tags": sorted({pose for record in person_records for pose in record.get("pose_tags", [])}),
            "lighting_groups": sorted({record.get("lighting_group") for record in person_records if record.get("lighting_group")}),
        }

    garment_coverage = {}
    for garment_family_id, garment_records in sorted(by_garment.items()):
        garment_coverage[garment_family_id] = {
            "record_count": len(garment_records),
            "person_count": len({record.get("person_id") for record in garment_records}),
            "garment_count": len({record.get("garment_id") for record in garment_records}),
            "pose_tags": sorted({pose for record in garment_records for pose in record.get("pose_tags", [])}),
            "lighting_groups": sorted({record.get("lighting_group") for record in garment_records if record.get("lighting_group")}),
        }

    pose_gaps = [pose for pose in RECOMMENDED_PILOT_POSES if pose not in set(pose_values)]
    flagged_records = [
        record.get("clip_id")
        for record in records
        if record.get("qa_status") != "passed" or record.get("rights_status") != "verified"
    ]

    return {
        "report_schema_version": "0.1.0",
        "tool": "dataset report",
        "tool_version": __version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_name": manifest.name,
        "dataset_root_name": root.name,
        "status": "failed" if validation["status"] == "failed" else "passed",
        "source_validation": validation,
        "summary": {
            "records_total": len(records),
            "people": len(by_person),
            "garment_families": len(by_garment),
            "sessions": len({record.get("session_id") for record in records}),
            "clips_with_unpassed_qa_or_unverified_rights": len(flagged_records),
            "dupatta_present_records": sum(record.get("dupatta_present") is True for record in records),
        },
        "status_counts": _sorted_counts([str(record.get("record_status")) for record in records]),
        "rights_status_counts": _sorted_counts([str(record.get("rights_status")) for record in records]),
        "qa_status_counts": _sorted_counts([str(record.get("qa_status")) for record in records]),
        "annotation_status_counts": _sorted_counts([str(record.get("annotation_status")) for record in records]),
        "pose_counts": _sorted_counts(pose_values),
        "lighting_group_counts": _sorted_counts(lighting_values),
        "occlusion_counts": _sorted_counts(occlusion_values),
        "coverage_gaps": {
            "recommended_pilot_poses_missing": pose_gaps,
            "flagged_clip_ids": sorted(value for value in flagged_records if isinstance(value, str)),
        },
        "per_person_opaque_id_coverage": person_coverage,
        "per_garment_family_opaque_id_coverage": garment_coverage,
    }
