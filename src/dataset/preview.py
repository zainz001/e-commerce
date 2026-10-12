"""Non-destructive timestamped preview and contact-sheet generation."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from . import __version__
from .inspector import sha256_file
from .validator import validate_manifest


def _trim(value: str, limit: int = 4000) -> str:
    value = value.strip()
    return value if len(value) <= limit else value[:limit] + "…"


def _tool_version(executable: str | None) -> str | None:
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0].strip() if lines else None


def _resolve_tool(dataset_root: Path, name: str, supplied: str | None) -> str | None:
    if supplied:
        return supplied
    on_path = shutil.which(name)
    if on_path:
        return on_path
    local_name = f"{name}.exe" if name == "ffprobe" or name == "ffmpeg" else name
    local = dataset_root / "tools" / "ffmpeg" / "bin" / local_name
    return str(local) if local.is_file() else None


def _resolve_relative_media_path(value: Any, dataset_root: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(normalized)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        return None
    if ".." in posix_path.parts:
        return None
    candidate = (dataset_root / Path(*posix_path.parts)).resolve(strict=False)
    try:
        candidate.relative_to(dataset_root)
    except ValueError:
        return None
    return candidate


def _run_ffprobe_frames(
    video_path: Path,
    ffprobe: str,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], str | None]:
    command = [
        ffprobe,
        "-v",
        "warning",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time,pkt_pts_time,pkt_dts_time,coded_picture_number,display_picture_number",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return [], f"ffprobe executable not found: {ffprobe}"
    except subprocess.TimeoutExpired:
        return [], f"ffprobe timed out after {timeout_seconds} seconds"
    except OSError as error:
        return [], f"could not run ffprobe: {error}"
    if result.returncode != 0:
        return [], _trim(result.stderr or "ffprobe returned a non-zero exit code")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        return [], f"ffprobe returned invalid JSON: {error}"
    frames = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(frames, list):
        return [], "ffprobe returned no video frame list"
    return [frame for frame in frames if isinstance(frame, dict)], None


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def select_timestamped_frames(
    frames: list[dict[str, Any]],
    sample_fps: float,
) -> list[dict[str, Any]]:
    """Select the first source frame at or after each sampling time."""

    interval = 1.0 / sample_fps
    next_target = 0.0
    selected: list[dict[str, Any]] = []
    for frame in frames:
        timestamp = _number(
            frame.get("best_effort_timestamp_time")
            or frame.get("pkt_pts_time")
            or frame.get("pkt_dts_time")
        )
        if timestamp is None or timestamp < 0:
            continue
        if timestamp + 1e-9 < next_target:
            continue
        selected.append(
            {
                "source_presentation_timestamp_seconds": timestamp,
                "source_frame_number": _integer(
                    frame.get("display_picture_number")
                    if frame.get("display_picture_number") is not None
                    else frame.get("coded_picture_number")
                ),
            }
        )
        next_target += interval
    return selected


def _extract_frame(
    video_path: Path,
    output_path: Path,
    timestamp: float,
    ffmpeg: str,
    timeout_seconds: float,
) -> tuple[bool, str | None]:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-ss",
        f"{timestamp:.6f}",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-an",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return False, f"ffmpeg executable not found: {ffmpeg}"
    except subprocess.TimeoutExpired:
        return False, f"ffmpeg timed out after {timeout_seconds} seconds"
    except OSError as error:
        return False, f"could not run ffmpeg: {error}"
    if result.returncode != 0:
        return False, _trim(result.stderr or "ffmpeg returned a non-zero exit code")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return False, "ffmpeg did not create a non-empty frame"
    return True, None


def _create_contact_sheet(
    frame_paths: list[Path],
    output_path: Path,
    ffmpeg: str,
    timeout_seconds: float,
) -> tuple[bool, str | None]:
    if not frame_paths:
        return False, "no extracted frames available for contact sheet"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        "1",
        "-i",
        str(frame_paths[0].parent / "frame_%06d.jpg"),
        "-vf",
        "tile=4x4:padding=8:margin=8:color=black",
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return False, f"ffmpeg executable not found: {ffmpeg}"
    except subprocess.TimeoutExpired:
        return False, f"ffmpeg timed out after {timeout_seconds} seconds"
    except OSError as error:
        return False, f"could not run ffmpeg: {error}"
    if result.returncode != 0:
        return False, _trim(result.stderr or "contact-sheet ffmpeg command failed")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return False, "ffmpeg did not create a non-empty contact sheet"
    return True, None


def _read_manifest_records(manifest_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8-sig") as source:
        for raw_line in source:
            if raw_line.strip():
                value = json.loads(raw_line)
                if isinstance(value, dict):
                    records.append(value)
    return records


def _existing_derivative_status(
    clip_dir: Path,
    *,
    source_sha256: str,
    sample_fps: float,
) -> str | None:
    metadata_path = clip_dir / "preview_metadata.json"
    if not clip_dir.exists():
        return None
    if not metadata_path.is_file():
        return "conflict"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "conflict"
    if (
        metadata.get("source_sha256") == source_sha256
        and float(metadata.get("sample_fps", -1)) == sample_fps
        and (clip_dir / "contact_sheet.jpg").is_file()
    ):
        return "same"
    return "conflict"


def create_previews(
    manifest_path: Path,
    *,
    dataset_root: Path,
    output_root: Path,
    sample_fps: float = 3.0,
    ffmpeg_executable: str | None = None,
    ffprobe_executable: str | None = None,
    timeout_seconds: float = 120.0,
    dry_run: bool = False,
    rights_registry_path: Path | None = None,
) -> dict[str, Any]:
    """Create timestamped derived previews without modifying source media."""

    if not 2.0 <= sample_fps <= 5.0:
        raise ValueError("sample_fps must be between 2 and 5")
    if timeout_seconds <= 0:
        raise ValueError("timeout must be greater than zero")

    manifest = manifest_path.expanduser().resolve()
    root = dataset_root.expanduser().resolve()
    output = output_root.expanduser().resolve(strict=False)
    if not manifest.is_file():
        raise ValueError(f"Manifest does not exist: {manifest_path}")
    if not root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {dataset_root}")
    try:
        output.relative_to(root)
    except ValueError as error:
        raise ValueError("output_root must be inside dataset_root") from error

    validation = validate_manifest(
        manifest,
        dataset_root=root,
        rights_registry_path=rights_registry_path,
    )
    report: dict[str, Any] = {
        "report_schema_version": "0.1.0",
        "tool": "dataset preview",
        "tool_version": __version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_name": manifest.name,
        "output_root_name": output.name,
        "sample_fps": sample_fps,
        "dry_run": dry_run,
        "source_validation": validation,
        "summary": {
            "records_total": 0,
            "processed": 0,
            "skipped_fictional": 0,
            "skipped_unverified_rights": 0,
            "skipped_existing": 0,
            "conflicts": 0,
            "failed": 0,
        },
        "clips": [],
    }
    if validation["status"] == "failed":
        report["summary"]["failed"] = 1
        report["clips"].append({"status": "blocked_by_manifest_validation"})
        return report

    ffprobe = _resolve_tool(root, "ffprobe", ffprobe_executable)
    ffmpeg = _resolve_tool(root, "ffmpeg", ffmpeg_executable)
    report["tools"] = {
        "ffprobe": _tool_version(ffprobe),
        "ffmpeg": _tool_version(ffmpeg),
    }
    records = _read_manifest_records(manifest)
    report["summary"]["records_total"] = len(records)

    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)

    for record in records:
        clip_id = record.get("clip_id")
        clip_result: dict[str, Any] = {"clip_id": clip_id, "status": "failed"}
        report["clips"].append(clip_result)
        if record.get("record_status") == "fictional_example":
            clip_result["status"] = "skipped_fictional"
            report["summary"]["skipped_fictional"] += 1
            continue
        if record.get("rights_status") != "verified":
            clip_result["status"] = "skipped_unverified_rights"
            report["summary"]["skipped_unverified_rights"] += 1
            continue
        if not isinstance(clip_id, str) or not clip_id:
            clip_result["error"] = "record has no usable clip_id"
            report["summary"]["failed"] += 1
            continue
        source_path = _resolve_relative_media_path(record.get("worn_video_path"), root)
        if source_path is None or not source_path.is_file():
            clip_result["error"] = "worn_video_path is not a safe existing file"
            report["summary"]["failed"] += 1
            continue
        if ffprobe is None or ffmpeg is None:
            clip_result["error"] = "ffprobe and ffmpeg are required for preview generation"
            report["summary"]["failed"] += 1
            continue

        source_sha256 = sha256_file(source_path)
        expected_sha256 = record.get("original_sha256")
        if isinstance(expected_sha256, str) and expected_sha256 and expected_sha256 != source_sha256:
            clip_result["status"] = "checksum_mismatch"
            clip_result["error"] = "source checksum differs from manifest original_sha256"
            report["summary"]["failed"] += 1
            continue

        clip_dir = output / clip_id
        existing = _existing_derivative_status(
            clip_dir,
            source_sha256=source_sha256,
            sample_fps=sample_fps,
        )
        if existing == "same":
            clip_result["status"] = "skipped_existing"
            report["summary"]["skipped_existing"] += 1
            continue
        if existing == "conflict":
            clip_result["status"] = "conflict_existing_output"
            clip_result["error"] = "existing output does not match source checksum/sample rate; no overwrite performed"
            report["summary"]["conflicts"] += 1
            continue

        frames, frame_error = _run_ffprobe_frames(source_path, ffprobe, timeout_seconds)
        if frame_error:
            clip_result["error"] = frame_error
            report["summary"]["failed"] += 1
            continue
        selected = select_timestamped_frames(frames, sample_fps)
        if not selected:
            clip_result["error"] = "no timestamped video frames were available"
            report["summary"]["failed"] += 1
            continue
        if dry_run:
            clip_result.update(
                {
                    "status": "would_process",
                    "source_sha256": source_sha256,
                    "selected_frame_count": len(selected),
                }
            )
            report["summary"]["processed"] += 1
            continue

        frames_dir = clip_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=False)
        extracted: list[dict[str, Any]] = []
        frame_paths: list[Path] = []
        extraction_error: str | None = None
        for index, selection in enumerate(selected, start=1):
            frame_path = frames_dir / f"frame_{index:06d}.jpg"
            ok, error = _extract_frame(
                source_path,
                frame_path,
                selection["source_presentation_timestamp_seconds"],
                ffmpeg,
                timeout_seconds,
            )
            if not ok:
                extraction_error = error
                break
            frame_paths.append(frame_path)
            extracted.append(
                {
                    "derived_frame_path": frame_path.relative_to(root).as_posix(),
                    **selection,
                }
            )
        if extraction_error:
            shutil.rmtree(clip_dir, ignore_errors=True)
            clip_result["error"] = extraction_error
            report["summary"]["failed"] += 1
            continue

        contact_sheet_path = clip_dir / "contact_sheet.jpg"
        contact_ok, contact_error = _create_contact_sheet(
            frame_paths,
            contact_sheet_path,
            ffmpeg,
            timeout_seconds,
        )
        if not contact_ok:
            shutil.rmtree(clip_dir, ignore_errors=True)
            clip_result["error"] = contact_error
            report["summary"]["failed"] += 1
            continue

        metadata = {
            "schema_version": "0.1.0",
            "clip_id": clip_id,
            "source_video_path": source_path.relative_to(root).as_posix(),
            "source_sha256": source_sha256,
            "sample_fps": sample_fps,
            "frames": extracted,
            "contact_sheet_path": contact_sheet_path.relative_to(root).as_posix(),
            "transform": "ffprobe source timestamps; ffmpeg single-frame JPEG extraction; ffmpeg tile contact sheet",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "software": {
                "dataset_tool_version": __version__,
                "ffmpeg_version": _tool_version(ffmpeg),
                "ffprobe_version": _tool_version(ffprobe),
            },
        }
        (clip_dir / "preview_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        clip_result.update(
            {
                "status": "processed",
                "source_sha256": source_sha256,
                "selected_frame_count": len(extracted),
                "output_directory": clip_dir.relative_to(root).as_posix(),
            }
        )
        report["summary"]["processed"] += 1

    return report
