from __future__ import annotations

import json
from pathlib import Path

from dataset.splitter import _assign_groups, _group_records, _verify_no_leakage
from dataset.splitter import create_split


def _record(number: int) -> dict[str, str]:
    return {
        "clip_id": f"clip_{number:06d}",
        "person_id": f"person_{number:04d}",
        "garment_family_id": f"family_{number:04d}",
        "session_id": f"session_{number:04d}",
    }


def test_grouped_assignment_is_deterministic_and_leakage_safe() -> None:
    records = [_record(number) for number in range(1, 7)]
    groups = _group_records(records)

    first = _assign_groups(groups, seed=42)
    second = _assign_groups(groups, seed=42)

    assert first == second
    assert set(first.values()) == {"train", "validation", "test"}
    assert _verify_no_leakage(records, first) == []


def test_split_manifest_is_written_and_is_idempotent(tmp_path: Path) -> None:
    for directory in ("garments", "references", "videos"):
        (tmp_path / directory).mkdir()
    (tmp_path / "garments/front.jpg").write_bytes(b"synthetic garment")
    (tmp_path / "references/person.jpg").write_bytes(b"synthetic reference")
    (tmp_path / "videos/take.mp4").write_bytes(b"synthetic video")
    records = []
    for number in range(1, 7):
        records.append(
            {
                "schema_version": "0.1.0",
                "record_status": "candidate",
                "clip_id": f"clip_{number:06d}",
                "person_id": f"person_{number:04d}",
                "garment_id": f"garment_{number:04d}",
                "garment_family_id": f"family_{number:04d}",
                "session_id": f"session_{number:04d}",
                "take_id": "take_01",
                "category": "womens_stitched_straight_cut_suit",
                "garment_front_path": "garments/front.jpg",
                "garment_back_path": None,
                "reference_image_path": "references/person.jpg",
                "worn_video_path": "videos/take.mp4",
                "original_sha256": None,
                "width": None,
                "height": None,
                "fps_num": None,
                "fps_den": None,
                "duration_seconds": None,
                "frame_count": None,
                "pose_tags": ["front"],
                "dupatta_present": False,
                "lighting_group": "indoor_soft_01",
                "occlusion_tags": [],
                "source_type": "self_recorded_with_consent",
                "rights_record_id": f"rights_{number:04d}",
                "rights_status": "verified",
                "annotation_status": "not_started",
                "qa_status": "pending",
                "split": None,
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    output = tmp_path / "splits/v0.1.0.jsonl"

    first = create_split(manifest, dataset_root=tmp_path, output_manifest=output, seed=42)
    second = create_split(manifest, dataset_root=tmp_path, output_manifest=output, seed=42)

    assert first["status"] == "passed"
    assert first["summary"]["split_records"] == 6
    assert second["status"] == "passed"
    assert any("identical output" in warning for warning in second["warnings"])
