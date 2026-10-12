"""Validation for rights-aware JSONL dataset manifests."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

REQUIRED_FIELDS = (
    "schema_version",
    "record_status",
    "clip_id",
    "person_id",
    "garment_id",
    "garment_family_id",
    "session_id",
    "take_id",
    "category",
    "garment_front_path",
    "garment_back_path",
    "reference_image_path",
    "worn_video_path",
    "original_sha256",
    "width",
    "height",
    "fps_num",
    "fps_den",
    "duration_seconds",
    "frame_count",
    "pose_tags",
    "dupatta_present",
    "lighting_group",
    "occlusion_tags",
    "source_type",
    "rights_record_id",
    "rights_status",
    "annotation_status",
    "qa_status",
    "split",
)

PATH_FIELDS = (
    "garment_front_path",
    "garment_back_path",
    "reference_image_path",
    "worn_video_path",
)

ID_FIELDS = (
    "clip_id",
    "person_id",
    "garment_id",
    "garment_family_id",
    "session_id",
    "take_id",
    "rights_record_id",
)

ID_EXPECTED_PREFIXES = {
    "clip_id": "clip",
    "person_id": "person",
    "garment_id": "garment",
    "garment_family_id": "family",
    "session_id": "session",
    "take_id": "take",
    "rights_record_id": "rights",
}

ALLOWED_RECORD_STATUSES = frozenset({"fictional_example", "candidate", "approved", "withdrawn"})
ALLOWED_RIGHTS_STATUSES = frozenset({"unknown", "pending", "verified", "restricted", "withdrawn"})
ALLOWED_ANNOTATION_STATUSES = frozenset({"not_started", "in_progress", "reviewed", "rejected"})
ALLOWED_QA_STATUSES = frozenset({"pending", "passed", "flagged", "rejected"})
ALLOWED_SPLITS = frozenset({None, "train", "validation", "test"})
ALLOWED_WITHDRAWAL_STATUSES = frozenset({"active", "withdrawn", "expired"})
ID_PREFIXES = frozenset({"clip", "person", "garment", "family", "session", "take", "rights"})


def _issue(
    issues: list[dict[str, Any]],
    *,
    line: int,
    code: str,
    message: str,
    severity: str = "error",
    clip_id: str | None = None,
) -> None:
    issues.append(
        {
            "line": line,
            "severity": severity,
            "code": code,
            "message": message,
            **({"clip_id": clip_id} if clip_id else {}),
        }
    )


def _is_opaque_id(value: Any, expected_prefix: str | None = None) -> bool:
    if not isinstance(value, str) or not value or " " in value:
        return False
    prefix, separator, suffix = value.partition("_")
    if not separator or not suffix or prefix not in ID_PREFIXES:
        return False
    return expected_prefix is None or prefix == expected_prefix


def _validate_relative_path(
    value: Any,
    *,
    field: str,
    dataset_root: Path,
    line: int,
    clip_id: str | None,
    issues: list[dict[str, Any]],
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        _issue(
            issues,
            line=line,
            clip_id=clip_id,
            code="missing_path",
            message=f"{field} must be a non-empty relative path",
        )
        return None

    normalized = value.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(normalized)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        _issue(
            issues,
            line=line,
            clip_id=clip_id,
            code="absolute_path",
            message=f"{field} must be relative to the dataset root",
        )
        return None
    if ".." in posix_path.parts:
        _issue(
            issues,
            line=line,
            clip_id=clip_id,
            code="path_traversal",
            message=f"{field} must not contain '..' path traversal",
        )
        return None

    candidate = (dataset_root / Path(*posix_path.parts)).resolve(strict=False)
    try:
        candidate.relative_to(dataset_root)
    except ValueError:
        _issue(
            issues,
            line=line,
            clip_id=clip_id,
            code="outside_dataset_root",
            message=f"{field} resolves outside the dataset root",
        )
        return None

    if not candidate.exists():
        _issue(
            issues,
            line=line,
            clip_id=clip_id,
            code="missing_file",
            message=f"{field} does not exist: {value}",
        )
    elif not candidate.is_file():
        _issue(
            issues,
            line=line,
            clip_id=clip_id,
            code="not_a_file",
            message=f"{field} is not a regular file: {value}",
        )
    return candidate


def _validate_record(
    record: dict[str, Any],
    *,
    line: int,
    dataset_root: Path,
    require_approved: bool,
    rights_registry: dict[str, dict[str, Any]] | None,
    issues: list[dict[str, Any]],
) -> None:
    clip_id = record.get("clip_id") if isinstance(record.get("clip_id"), str) else None
    missing_fields = [field for field in REQUIRED_FIELDS if field not in record]
    for field in missing_fields:
        _issue(
            issues,
            line=line,
            clip_id=clip_id,
            code="missing_field",
            message=f"missing required field: {field}",
        )

    if record.get("schema_version") != "0.1.0":
        _issue(
            issues,
            line=line,
            clip_id=clip_id,
            code="unsupported_schema_version",
            message="schema_version must be 0.1.0",
        )

    record_status = record.get("record_status")
    if record_status not in ALLOWED_RECORD_STATUSES:
        _issue(
            issues,
            line=line,
            clip_id=clip_id,
            code="invalid_record_status",
            message=f"record_status must be one of {sorted(ALLOWED_RECORD_STATUSES)}",
        )
    if require_approved and record_status not in {"approved", "fictional_example"}:
        _issue(
            issues,
            line=line,
            clip_id=clip_id,
            code="not_approved",
            message="approved validation requires record_status=approved",
        )

    for field in ID_FIELDS:
        value = record.get(field)
        expected_prefix = ID_EXPECTED_PREFIXES[field]
        if not _is_opaque_id(value, expected_prefix):
            _issue(
                issues,
                line=line,
                clip_id=clip_id,
                code="invalid_id",
                message=f"{field} must be an opaque ID with prefix {expected_prefix}_",
            )

    if not isinstance(record.get("pose_tags"), list) or not all(
        isinstance(value, str) and value for value in record.get("pose_tags", [])
    ):
        _issue(
            issues,
            line=line,
            clip_id=clip_id,
            code="invalid_pose_tags",
            message="pose_tags must be a list of non-empty strings",
        )
    if not isinstance(record.get("occlusion_tags"), list) or not all(
        isinstance(value, str) and value for value in record.get("occlusion_tags", [])
    ):
        _issue(
            issues,
            line=line,
            clip_id=clip_id,
            code="invalid_occlusion_tags",
            message="occlusion_tags must be a list of non-empty strings",
        )
    if not isinstance(record.get("dupatta_present"), bool):
        _issue(
            issues,
            line=line,
            clip_id=clip_id,
            code="invalid_dupatta_value",
            message="dupatta_present must be a boolean",
        )

    for field, allowed in (
        ("rights_status", ALLOWED_RIGHTS_STATUSES),
        ("annotation_status", ALLOWED_ANNOTATION_STATUSES),
        ("qa_status", ALLOWED_QA_STATUSES),
    ):
        if record.get(field) not in allowed:
            _issue(
                issues,
                line=line,
                clip_id=clip_id,
                code=f"invalid_{field}",
                message=f"{field} must be one of {sorted(allowed)}",
            )
    if record.get("split") not in ALLOWED_SPLITS:
        _issue(
            issues,
            line=line,
            clip_id=clip_id,
            code="invalid_split",
            message="split must be null, train, validation, or test",
        )

    if record_status == "fictional_example":
        _issue(
            issues,
            line=line,
            clip_id=clip_id,
            code="fictional_example_not_eligible",
            severity="warning",
            message="fictional example is schema documentation only and cannot enter an approved dataset",
        )
        return

    for field in PATH_FIELDS:
        value = record.get(field)
        if field == "garment_back_path" and value is None:
            continue
        _validate_relative_path(
            value,
            field=field,
            dataset_root=dataset_root,
            line=line,
            clip_id=clip_id,
            issues=issues,
        )

    rights_status = record.get("rights_status")
    if rights_status != "verified":
        severity = "error" if record_status == "approved" or require_approved else "warning"
        _issue(
            issues,
            line=line,
            clip_id=clip_id,
            code="rights_not_verified",
            severity=severity,
            message="rights_status must be verified before this record can be approved",
        )

    if rights_registry is not None:
        rights_record_id = record.get("rights_record_id")
        registry_record = rights_registry.get(rights_record_id)
        if registry_record is None:
            severity = "error" if record_status == "approved" or require_approved else "warning"
            _issue(
                issues,
                line=line,
                clip_id=clip_id,
                code="rights_registry_record_missing",
                severity=severity,
                message="rights_record_id is not present in the supplied rights registry",
            )
        else:
            withdrawal_status = registry_record.get("withdrawal_status")
            if withdrawal_status != "active":
                severity = "error" if record_status == "approved" or require_approved else "warning"
                _issue(
                    issues,
                    line=line,
                    clip_id=clip_id,
                    code="rights_not_active",
                    severity=severity,
                    message="rights registry withdrawal_status must be active",
                )
            participant_consent = registry_record.get("participant_consent")
            garment_permission = registry_record.get("garment_permission")
            consent_ok = isinstance(participant_consent, dict) and all(
                participant_consent.get(field) is True
                for field in ("capture", "storage", "quality_review", "model_training")
            )
            commercial_ok = isinstance(participant_consent, dict) and participant_consent.get("commercial_training") is True
            garment_ok = isinstance(garment_permission, dict) and all(
                garment_permission.get(field) is True
                for field in ("product_images", "commercial_use")
            )
            if not consent_ok or not garment_ok:
                severity = "error" if record_status == "approved" or require_approved else "warning"
                _issue(
                    issues,
                    line=line,
                    clip_id=clip_id,
                    code="rights_scope_incomplete",
                    severity=severity,
                    message="rights registry lacks required capture, storage, review, training, or garment permissions",
                )
            if not commercial_ok and (record_status == "approved" or require_approved):
                _issue(
                    issues,
                    line=line,
                    clip_id=clip_id,
                    code="commercial_training_not_allowed",
                    message="approved validation requires explicit commercial_training permission",
                )

    if record_status == "approved" or require_approved:
        if not isinstance(record.get("original_sha256"), str) or len(record["original_sha256"]) != 64:
            _issue(
                issues,
                line=line,
                clip_id=clip_id,
                code="missing_checksum",
                message="approved records require a 64-character original_sha256",
            )
        if record.get("qa_status") != "passed":
            _issue(
                issues,
                line=line,
                clip_id=clip_id,
                code="qa_not_passed",
                message="approved records require qa_status=passed",
            )


def validate_manifest(
    manifest_path: Path,
    *,
    dataset_root: Path,
    require_approved: bool = False,
    rights_registry_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a JSONL manifest without changing the manifest or source files."""

    manifest = manifest_path.expanduser().resolve()
    root = dataset_root.expanduser().resolve()
    if not manifest.exists() or not manifest.is_file():
        raise ValueError(f"Manifest does not exist: {manifest_path}")
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {dataset_root}")

    issues: list[dict[str, Any]] = []
    rights_registry: dict[str, dict[str, Any]] | None = None
    if rights_registry_path is not None:
        registry_path = rights_registry_path.expanduser().resolve()
        if not registry_path.is_file():
            raise ValueError(f"Rights registry does not exist: {rights_registry_path}")
        rights_registry = {}
        with registry_path.open("r", encoding="utf-8-sig") as registry_source:
            for registry_line_number, raw_registry_line in enumerate(registry_source, start=1):
                if not raw_registry_line.strip():
                    continue
                try:
                    registry_record = json.loads(raw_registry_line)
                except json.JSONDecodeError as error:
                    _issue(
                        issues,
                        line=registry_line_number,
                        code="invalid_rights_registry_json",
                        message=f"invalid rights registry JSON: {error.msg}",
                    )
                    continue
                if not isinstance(registry_record, dict):
                    _issue(
                        issues,
                        line=registry_line_number,
                        code="rights_registry_record_not_object",
                        message="each rights registry JSONL record must be an object",
                    )
                    continue
                rights_record_id = registry_record.get("rights_record_id")
                if not _is_opaque_id(rights_record_id, "rights"):
                    _issue(
                        issues,
                        line=registry_line_number,
                        code="invalid_rights_registry_id",
                        message="rights registry records require an opaque rights_record_id",
                    )
                    continue
                if rights_record_id in rights_registry:
                    _issue(
                        issues,
                        line=registry_line_number,
                        code="duplicate_rights_registry_id",
                        message="rights_record_id is duplicated in the rights registry",
                    )
                    continue
                if registry_record.get("withdrawal_status") not in ALLOWED_WITHDRAWAL_STATUSES:
                    _issue(
                        issues,
                        line=registry_line_number,
                        code="invalid_withdrawal_status",
                        message="withdrawal_status must be active, withdrawn, or expired",
                    )
                rights_registry[rights_record_id] = registry_record
    elif require_approved:
        _issue(
            issues,
            line=0,
            code="rights_registry_required",
            message="approved validation requires --rights-registry",
        )
    records_total = 0
    fictional_records = 0
    eligible_records = 0
    seen_clip_ids: dict[str, int] = {}
    seen_take_keys: dict[tuple[Any, ...], int] = {}

    with manifest.open("r", encoding="utf-8-sig") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                _issue(
                    issues,
                    line=line_number,
                    code="blank_line",
                    message="blank lines are not valid JSONL records",
                )
                continue
            records_total += 1
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                _issue(
                    issues,
                    line=line_number,
                    code="invalid_json",
                    message=f"invalid JSON: {error.msg}",
                )
                continue
            if not isinstance(record, dict):
                _issue(
                    issues,
                    line=line_number,
                    code="record_not_object",
                    message="each JSONL record must be an object",
                )
                continue

            clip_id = record.get("clip_id")
            if isinstance(clip_id, str):
                if clip_id in seen_clip_ids:
                    _issue(
                        issues,
                        line=line_number,
                        clip_id=clip_id,
                        code="duplicate_clip_id",
                        message=f"clip_id was already used on line {seen_clip_ids[clip_id]}",
                    )
                else:
                    seen_clip_ids[clip_id] = line_number

            take_key = tuple(record.get(field) for field in ("person_id", "garment_id", "session_id", "take_id"))
            if all(isinstance(value, str) and value for value in take_key):
                if take_key in seen_take_keys:
                    _issue(
                        issues,
                        line=line_number,
                        clip_id=clip_id if isinstance(clip_id, str) else None,
                        code="duplicate_take_key",
                        message=f"person/garment/session/take combination was already used on line {seen_take_keys[take_key]}",
                    )
                else:
                    seen_take_keys[take_key] = line_number

            _validate_record(
                record,
                line=line_number,
                dataset_root=root,
                require_approved=require_approved,
                rights_registry=rights_registry,
                issues=issues,
            )
            if record.get("record_status") == "fictional_example":
                fictional_records += 1
            elif record.get("record_status") in {"candidate", "approved"}:
                eligible_records += 1

    counts = Counter(issue["severity"] for issue in issues)
    error_count = counts.get("error", 0)
    warning_count = counts.get("warning", 0)
    return {
        "report_schema_version": "0.1.0",
        "tool": "dataset validate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_name": manifest.name,
        "dataset_root_name": root.name,
        "require_approved": require_approved,
        "status": "failed" if error_count else "passed",
        "summary": {
            "records_total": records_total,
            "eligible_records": eligible_records,
            "fictional_records": fictional_records,
            "errors": error_count,
            "warnings": warning_count,
        },
        "issues": issues,
    }
