from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from capture import capture_screen_sample
from control_runner import apply_custom_lut, reset_display
from profile_writer import write_custom_lut_profile
from visibility_engine import (
    analyze_frames,
    find_frame_files,
    find_latest_frame_folder,
    save_masked_debug,
    save_zone_overlay,
)


DEFAULT_DISPLAY_INDEX = 1
DEBUG_ROOT = Path("debug_views")


# ============================================================
# Frame Loading
# ============================================================

def load_frames_from_paths(paths: list[Path]) -> list[np.ndarray]:
    """
    Load saved frame image files into OpenCV arrays.
    """
    frames: list[np.ndarray] = []

    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if frame is not None:
            frames.append(frame)

    return frames


def select_valid_frame_paths(frame_paths: list[Path]) -> list[Path]:
    """
    Prefer frames marked valid by the capture stage.

    Falls back to all frames if no valid frames exist.
    """
    valid_paths = [
        path
        for path in frame_paths
        if "valid" in path.stem.lower()
    ]

    return valid_paths if valid_paths else frame_paths


# ============================================================
# Reports and Debug Output
# ============================================================

def save_json(path: Path, data: dict[str, Any]) -> None:
    """
    Save a JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def write_debug_views(
    frame_paths: list[Path],
    output_folder: Path,
    limit: int = 10,
) -> None:
    """
    Save masked and zoned debug images.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    selected = frame_paths[:limit]

    for index, path in enumerate(selected, start=1):
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if frame is None:
            continue

        save_masked_debug(
            frame,
            output_folder / f"{index:03}_{path.stem}_masked.jpg",
        )

        save_zone_overlay(
            frame,
            output_folder / f"{index:03}_{path.stem}_zones.jpg",
        )


# ============================================================
# Visibility Analysis
# ============================================================

def analyze_frame_folder(
    frame_folder: Path,
    display_index: int,
    write_debug: bool,
) -> dict[str, Any]:
    """
    Analyze a saved frame folder with the masked zonal visibility engine.
    """
    frame_paths = find_frame_files(frame_folder)
    selected_paths = select_valid_frame_paths(frame_paths)

    if not selected_paths:
        raise RuntimeError(f"No frame files found in {frame_folder}")

    frames = load_frames_from_paths(selected_paths)

    if not frames:
        raise RuntimeError(f"No readable frames found in {frame_folder}")

    report = analyze_frames(frames)
    recommendation = report["lut_recommendation"]

    control_profile = write_custom_lut_profile(
        recommendation=recommendation,
        display_index=display_index,
    )

    report["frame_folder"] = str(frame_folder)
    report["frame_files_used"] = [str(path) for path in selected_paths]
    report["control_profile"] = control_profile

    report_path = frame_folder / "visibility_report.json"
    save_json(report_path, report)

    if write_debug:
        debug_folder = DEBUG_ROOT / frame_folder.name
        write_debug_views(selected_paths, debug_folder)

    print_recommendation(report, report_path)

    return report


def print_recommendation(report: dict[str, Any], report_path: Path) -> None:
    """
    Print a clean recommendation summary.
    """
    recommendation = report["lut_recommendation"]
    pressures = report["average_pressures"]

    print()
    print("RaidVision Recommendation")
    print()
    print("Average Pressures")
    for key, value in pressures.items():
        print(f"{key}: {value}")

    print()
    print("Recommended Custom LUT")
    print(f"shadow_lift: {recommendation['shadow_lift']}")
    print(f"midtone: {recommendation['midtone']}")
    print(f"highlight_protect: {recommendation['highlight_protect']}")

    print()
    print("Reasoning")
    for reason in recommendation["reasoning"]:
        print(reason)

    print()
    print(f"Saved report to: {report_path}")
    print("Saved profile to: control_profile.json")


# ============================================================
# CLI Runner
# ============================================================

def should_apply_now() -> bool:
    """
    Ask the user whether to apply the recommended profile.
    """
    answer = input("\nApply this profile now? Type y or n: ").strip().lower()
    return answer in {"y", "yes"}


def build_parser() -> argparse.ArgumentParser:
    """
    Build command line options.
    """
    parser = argparse.ArgumentParser(description="RaidVision runner")

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the recommendation automatically.",
    )

    parser.add_argument(
        "--no-apply",
        action="store_true",
        help="Do not apply the recommendation.",
    )

    parser.add_argument(
        "--analyze-latest",
        action="store_true",
        help="Analyze the latest saved frame folder instead of capturing a new sample.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write masked and zone overlay debug images.",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the display and exit.",
    )

    parser.add_argument(
        "--display",
        type=int,
        default=DEFAULT_DISPLAY_INDEX,
        help="C# display index.",
    )

    return parser


def main() -> int:
    """
    Run the main RaidVision workflow.
    """
    parser = build_parser()
    args = parser.parse_args()

    if args.reset:
        return reset_display(args.display)

    if args.analyze_latest:
        frame_folder = find_latest_frame_folder()

        if frame_folder is None:
            print("No saved frame folder found.")
            return 1
    else:
        capture_result = capture_screen_sample()
        frame_folder = Path(capture_result["frame_folder"])

    report = analyze_frame_folder(
        frame_folder=frame_folder,
        display_index=args.display,
        write_debug=args.debug,
    )

    if args.no_apply:
        print("Profile not applied.")
        return 0

    if args.apply or should_apply_now():
        return apply_custom_lut(
            recommendation=report["lut_recommendation"],
            display_index=args.display,
        )

    print("Profile not applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
