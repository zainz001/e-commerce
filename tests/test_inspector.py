from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dataset.inspector import inspect_directory, sha256_file


def test_sha256_file_matches_standard_library(tmp_path: Path) -> None:
    source = tmp_path / "sample.bin"
    source.write_bytes(b"synthetic fixture\n")

    expected = hashlib.sha256(b"synthetic fixture\n").hexdigest()
    assert sha256_file(source) == expected


def test_inspection_is_read_only_and_reports_duplicates(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    video_a = source / "take_a.mp4"
    video_b = source / "take_b.mp4"
    image = source / "garment.jpg"
    video_a.write_bytes(b"not a real video; synthetic fixture")
    video_b.write_bytes(video_a.read_bytes())
    image.write_bytes(b"synthetic image fixture")

    before = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    report = inspect_directory(
        source,
        ffprobe_executable="definitely-not-installed-ffprobe",
    )
    after = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }

    assert before == after
    assert report["dry_run"] is True
    assert report["summary"]["total_files"] == 3
    assert report["summary"]["video_files"] == 2
    assert report["summary"]["image_files"] == 1
    assert report["summary"]["duplicate_groups"] == 1
    assert report["duplicate_groups"][0]["relative_paths"] == ["take_a.mp4", "take_b.mp4"]
    assert all(record["sha256"] for record in report["files"])
    video_records = [record for record in report["files"] if record["kind"] == "video"]
    assert all(record["ffprobe"]["status"] == "unavailable" for record in video_records)


def test_missing_input_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        inspect_directory(tmp_path / "missing")
