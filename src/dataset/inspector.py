"""Read-only source-file and video metadata inspection."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__

VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})
IMAGE_EXTENSIONS = frozenset({".bmp", ".gif", ".heic", ".jpeg", ".jpg", ".png", ".webp"})
HASH_CHUNK_SIZE = 1024 * 1024
MAX_ERROR_TEXT = 4000


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without changing it."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _trim_error(value: str) -> str:
    value = value.strip()
    if len(value) <= MAX_ERROR_TEXT:
        return value
    return value[:MAX_ERROR_TEXT] + "…"


def _parse_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_rational(value: Any) -> dict[str, int] | None:
    if not isinstance(value, str) or "/" not in value:
        return None
    numerator, denominator = value.split("/", maxsplit=1)
    try:
        numerator_int = int(numerator)
        denominator_int = int(denominator)
    except ValueError:
        return None
    if denominator_int == 0:
        return None
    return {"num": numerator_int, "den": denominator_int}


def _rotation_degrees(stream: dict[str, Any]) -> int | None:
    tags = stream.get("tags")
    if isinstance(tags, dict):
        rotation = _parse_number(tags.get("rotate"))
        if rotation is not None:
            return int(rotation)
    return None


def _technical_metadata(probe_json: dict[str, Any]) -> dict[str, Any]:
    streams = probe_json.get("streams")
    video_stream: dict[str, Any] = {}
    if isinstance(streams, list):
        video_stream = next(
            (
                stream
                for stream in streams
                if isinstance(stream, dict) and stream.get("codec_type") == "video"
            ),
            {},
        )

    format_data = probe_json.get("format")
    if not isinstance(format_data, dict):
        format_data = {}

    stream_duration = _parse_number(video_stream.get("duration"))
    format_duration = _parse_number(format_data.get("duration"))
    duration = stream_duration if stream_duration is not None else format_duration

    stream_frame_count = _parse_int(video_stream.get("nb_frames"))
    fps_value = video_stream.get("avg_frame_rate")
    if fps_value in (None, "0/0"):
        fps_value = video_stream.get("r_frame_rate")

    return {
        "format_name": format_data.get("format_name"),
        "codec_name": video_stream.get("codec_name"),
        "pixel_format": video_stream.get("pix_fmt"),
        "width": _parse_int(video_stream.get("width")),
        "height": _parse_int(video_stream.get("height")),
        "r_frame_rate": video_stream.get("r_frame_rate"),
        "avg_frame_rate": video_stream.get("avg_frame_rate"),
        "fps": _parse_rational(fps_value),
        "duration_seconds": duration,
        "frame_count": stream_frame_count,
        "rotation_degrees": _rotation_degrees(video_stream),
    }


def _ffprobe_version(executable: str | None) -> str | None:
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
    first_line = result.stdout.splitlines()
    return first_line[0].strip() if first_line else None


def _run_ffprobe(
    path: Path,
    executable: str | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    if not executable:
        return {"status": "unavailable", "error": "ffprobe was not found on PATH"}

    command = [
        executable,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
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
        return {"status": "unavailable", "error": f"ffprobe executable not found: {executable}"}
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"ffprobe timed out after {timeout_seconds} seconds"}
    except OSError as error:
        return {"status": "error", "error": f"could not run ffprobe: {error}"}

    if result.returncode != 0:
        return {
            "status": "error",
            "returncode": result.returncode,
            "error": _trim_error(result.stderr or "ffprobe returned a non-zero exit code"),
        }

    try:
        probe_json = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        return {"status": "error", "error": f"ffprobe returned invalid JSON: {error}"}

    if not isinstance(probe_json, dict):
        return {"status": "error", "error": "ffprobe JSON root was not an object"}
    return {"status": "ok", "metadata": _technical_metadata(probe_json)}


def _file_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    return "other"


def _inspect_file(
    path: Path,
    root: Path,
    ffprobe_executable: str | None,
    ffprobe_timeout_seconds: float,
) -> dict[str, Any]:
    relative_path = path.relative_to(root).as_posix()
    kind = _file_kind(path)
    record: dict[str, Any] = {
        "relative_path": relative_path,
        "kind": kind,
        "extension": path.suffix.lower(),
        "size_bytes": None,
        "sha256": None,
        "inspection_status": "ok",
        "flags": [],
    }

    try:
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = sha256_file(path)
    except OSError as error:
        record["inspection_status"] = "error"
        record["flags"] = [f"could_not_read_file: {error}"]
        return record

    if kind != "video":
        record["ffprobe"] = {"status": "not_applicable"}
        return record

    ffprobe = _run_ffprobe(path, ffprobe_executable, ffprobe_timeout_seconds)
    record["ffprobe"] = ffprobe
    if ffprobe["status"] in {"error", "unavailable"}:
        record["inspection_status"] = "flagged"
        record["flags"] = [f"ffprobe_{ffprobe['status']}"]
    return record


def inspect_directory(
    input_directory: Path,
    *,
    ffprobe_executable: str | None = None,
    ffprobe_timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Inspect a directory recursively without changing its contents."""

    root = input_directory.expanduser().resolve()
    if not root.exists():
        raise ValueError(f"Input directory does not exist: {input_directory}")
    if not root.is_dir():
        raise ValueError(f"Input path is not a directory: {input_directory}")
    if ffprobe_timeout_seconds <= 0:
        raise ValueError("ffprobe timeout must be greater than zero")

    resolved_ffprobe = ffprobe_executable or shutil.which("ffprobe")
    if resolved_ffprobe is None:
        local_ffprobe = Path.cwd() / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"
        if local_ffprobe.is_file():
            resolved_ffprobe = str(local_ffprobe)
    records: list[dict[str, Any]] = []
    scan_errors: list[dict[str, str]] = []
    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix().lower()):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            records.append(
                _inspect_file(
                    path,
                    root,
                    resolved_ffprobe,
                    ffprobe_timeout_seconds,
                )
            )
        except (OSError, ValueError) as error:
            scan_errors.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "error": str(error),
                }
            )

    by_hash: dict[str, list[str]] = defaultdict(list)
    for record in records:
        digest = record.get("sha256")
        if digest:
            by_hash[digest].append(record["relative_path"])
    duplicate_groups = [
        {"sha256": digest, "relative_paths": sorted(paths)}
        for digest, paths in sorted(by_hash.items())
        if len(paths) > 1
    ]

    video_records = [record for record in records if record["kind"] == "video"]
    report = {
        "report_schema_version": "0.1.0",
        "tool": "dataset inspect",
        "tool_version": __version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": True,
        "input_directory_name": root.name,
        "environment": {
            "python_version": platform.python_version(),
            "ffprobe_version": _ffprobe_version(resolved_ffprobe),
        },
        "summary": {
            "total_files": len(records),
            "video_files": len(video_records),
            "image_files": sum(record["kind"] == "image" for record in records),
            "other_files": sum(record["kind"] == "other" for record in records),
            "flagged_files": sum(record["inspection_status"] == "flagged" for record in records),
            "unreadable_files": sum(record["inspection_status"] == "error" for record in records),
            "duplicate_groups": len(duplicate_groups),
            "scan_errors": len(scan_errors),
        },
        "duplicate_groups": duplicate_groups,
        "scan_errors": scan_errors,
        "files": records,
    }
    return report
