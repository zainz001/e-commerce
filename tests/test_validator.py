from __future__ import annotations

import json
from pathlib import Path

from dataset.validator import validate_manifest


def _record(*, root: Path, status: str = "candidate", rights: str = "verified") -> dict[str, object]:
    (root / "garments").mkdir()
    (root / "references").mkdir()
    (root / "videos").mkdir()
    (root / "garments/front.jpg").write_bytes(b"synthetic garment")
    (root / "references/person.jpg").write_bytes(b"synthetic reference")
    (root / "videos/take.mp4").write_bytes(b"synthetic video")
    return {
        "schema_version": "0.1.0",
        "record_status": status,
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
        "pose_tags": ["front"],
        "dupatta_present": False,
        "lighting_group": "indoor_soft_01",
        "occlusion_tags": [],
        "source_type": "self_recorded_with_consent",
        "rights_record_id": "rights_000001",
        "rights_status": rights,
        "annotation_status": "not_started",
        "qa_status": "pending",
        "split": None,
    }


def _write_manifest(root: Path, record: dict[str, object]) -> Path:
    manifest = root / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return manifest


def test_candidate_with_verified_rights_and_existing_links_passes(tmp_path: Path) -> None:
    record = _record(root=tmp_path)
    report = validate_manifest(_write_manifest(tmp_path, record), dataset_root=tmp_path)

    assert report["status"] == "passed"
    assert report["summary"]["errors"] == 0
    assert report["summary"]["warnings"] == 0


def test_approved_record_requires_rights_checksum_and_qa(tmp_path: Path) -> None:
    record = _record(root=tmp_path, status="approved", rights="pending")
    report = validate_manifest(_write_manifest(tmp_path, record), dataset_root=tmp_path)

    assert report["status"] == "failed"
    codes = {issue["code"] for issue in report["issues"]}
    assert {"rights_not_verified", "missing_checksum", "qa_not_passed"} <= codes


def test_path_traversal_is_rejected(tmp_path: Path) -> None:
    record = _record(root=tmp_path)
    record["worn_video_path"] = "../outside.mp4"
    report = validate_manifest(_write_manifest(tmp_path, record), dataset_root=tmp_path)

    assert report["status"] == "failed"
    assert any(issue["code"] == "path_traversal" for issue in report["issues"])


def test_fictional_example_is_not_treated_as_real_data(tmp_path: Path) -> None:
    record = _record(root=tmp_path, status="fictional_example", rights="unknown")
    record["garment_front_path"] = "data/raw/fictional/front.jpg"
    record["reference_image_path"] = "data/raw/fictional/reference.jpg"
    record["worn_video_path"] = "data/raw/fictional/video.mp4"
    report = validate_manifest(_write_manifest(tmp_path, record), dataset_root=tmp_path)

    assert report["status"] == "passed"
    assert report["summary"]["fictional_records"] == 1
    assert any(issue["code"] == "fictional_example_not_eligible" for issue in report["issues"])


def test_utf8_bom_is_accepted_for_windows_jsonl(tmp_path: Path) -> None:
    record = _record(root=tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\ufeff" + json.dumps(record) + "\n", encoding="utf-8")

    report = validate_manifest(manifest, dataset_root=tmp_path)

    assert report["status"] == "passed"


def test_approved_record_cross_checks_restricted_rights_registry(tmp_path: Path) -> None:
    record = _record(root=tmp_path, status="approved", rights="verified")
    record["original_sha256"] = "0" * 64
    record["qa_status"] = "passed"
    manifest = _write_manifest(tmp_path, record)
    registry = tmp_path / "rights_registry.jsonl"
    registry.write_text(
        json.dumps(
            {
                "rights_record_id": "rights_000001",
                "withdrawal_status": "active",
                "participant_consent": {
                    "capture": True,
                    "storage": True,
                    "quality_review": True,
                    "model_training": True,
                    "commercial_training": True,
                },
                "garment_permission": {"product_images": True, "commercial_use": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_manifest(
        manifest,
        dataset_root=tmp_path,
        require_approved=True,
        rights_registry_path=registry,
    )

    assert report["status"] == "passed"
