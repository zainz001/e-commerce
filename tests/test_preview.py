from __future__ import annotations

from dataset.preview import select_timestamped_frames


def test_timestamp_selection_preserves_source_timestamps_and_numbers() -> None:
    frames = [
        {"best_effort_timestamp_time": "0.000", "display_picture_number": "0"},
        {"best_effort_timestamp_time": "0.040", "display_picture_number": "1"},
        {"best_effort_timestamp_time": "0.360", "display_picture_number": "9"},
        {"best_effort_timestamp_time": "0.720", "display_picture_number": "18"},
        {"best_effort_timestamp_time": "1.080", "display_picture_number": "27"},
    ]

    selected = select_timestamped_frames(frames, sample_fps=3.0)

    assert [item["source_presentation_timestamp_seconds"] for item in selected] == [0.0, 0.36, 0.72, 1.08]
    assert [item["source_frame_number"] for item in selected] == [0, 9, 18, 27]


def test_timestamp_selection_uses_pts_fallback() -> None:
    frames = [
        {"pkt_pts_time": "0.0", "coded_picture_number": "4"},
        {"pkt_pts_time": "0.5", "coded_picture_number": "9"},
    ]

    selected = select_timestamped_frames(frames, sample_fps=2.0)

    assert selected[1]["source_presentation_timestamp_seconds"] == 0.5
    assert selected[1]["source_frame_number"] == 9
