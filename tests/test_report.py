from __future__ import annotations

import json
from pathlib import Path

from dataset.report import create_coverage_report


def test_report_counts_coverage_without_personal_identity_fields(tmp_path: Path) -> None:
    for directory in ("garments", "references", "videos"):
        (tmp_path / directory).mkdir()
    (tmp_path / "garments/front.jpg").write_bytes(b"synthetic garment")
    (tmp_path / "references/person.jpg").write_bytes(b"synthetic reference")
    (tmp_path / "videos/take.mp4").write_bytes(b"synthetic video")
    record = {
        "schema_version": "0.1.0",
        "record_status": "candidate",
        "clip_id": "clip_000001",
        "person_id": "person_0001",
        "garment_id": "garment_0001",
        "garment_family_id": "family_0001",
        "session_id": "session_0001",
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
        "pose_tags": ["front", "left_turn"],
        "dupatta_present": False,
        "lighting_group": "indoor_soft_01",
        "occlusion_tags": ["hair_over_collar"],
        "source_type": "self_recorded_with_consent",
        "rights_record_id": "rights_000001",
        "rights_status": "verified",
        "annotation_status": "not_started",
        "qa_status": "pending",
        "split": None,
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    report = create_coverage_report(manifest, dataset_root=tmp_path)

    assert report["status"] == "passed"
    assert report["summary"]["people"] == 1
    assert report["summary"]["garment_families"] == 1
    assert report["pose_counts"] == {"front": 1, "left_turn": 1}
    assert report["coverage_gaps"]["recommended_pilot_poses_missing"] == ["right_turn"]
    rendered = json.dumps(report).lower()
    assert "person_name" not in rendered
    assert "email" not in rendered
    assert "phone" not in rendered
