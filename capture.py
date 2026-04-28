from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import mss
import numpy as np


# ============================================================
# Capture Configuration
# ============================================================

MSS_MONITOR_INDEX = 3

SAMPLE_DURATION_SECONDS = 20
SAMPLE_INTERVAL_SECONDS = 0.5

FRAME_WIDTH = 640
FRAME_HEIGHT = 360
FRAME_QUALITY = 80

FRAME_ROOT = Path("frames")


# ============================================================
# Folder and File Helpers
# ============================================================

def create_raid_folder() -> Path:
    """
    Create a timestamped folder for the current screen sample.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = FRAME_ROOT / f"raid_{timestamp}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_json(path: Path, data: dict[str, Any]) -> None:
    """
    Save JSON metadata.
    """
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


# ============================================================
# Basic Capture Validation
# ============================================================

def analyze_basic_frame(frame: np.ndarray) -> dict[str, float]:
    """
    Calculate simple frame metrics used only for capture validation.

    The real visibility logic lives in visibility_engine.py.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = float(np.sum(hist))

    if total <= 0:
        return {
            "avg_brightness": 0.0,
            "shadow_pct": 100.0,
            "midtone_pct": 0.0,
            "highlight_pct": 0.0,
            "contrast_spread": 0.0,
        }

    return {
        "avg_brightness": float(np.mean(gray)),
        "shadow_pct": float((np.sum(hist[0:90]) / total) * 100.0),
        "midtone_pct": float((np.sum(hist[90:170]) / total) * 100.0),
        "highlight_pct": float((np.sum(hist[230:256]) / total) * 100.0),
        "contrast_spread": float(np.std(gray)),
    }


def classify_capture_frame(metrics: dict[str, float]) -> tuple[bool, str]:
    """
    Reject obvious bad captures such as loading screens or whiteout frames.
    """
    if metrics["avg_brightness"] < 8 and metrics["shadow_pct"] > 97:
        return False, "black_loading_or_bad_capture"

    if metrics["highlight_pct"] > 90:
        return False, "mostly_sky_or_whiteout"

    if metrics["midtone_pct"] < 1 and metrics["contrast_spread"] < 8:
        return False, "no_usable_detail"

    return True, "valid"


def save_frame(frame: np.ndarray, folder: Path, index: int, status: str) -> Path:
    """
    Save a resized capture frame to disk.
    """
    resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    filename = f"frame_{index:03}_{status}.jpg"
    path = folder / filename

    cv2.imwrite(
        str(path),
        resized,
        [cv2.IMWRITE_JPEG_QUALITY, FRAME_QUALITY],
    )

    return path


# ============================================================
# Public Capture Function
# ============================================================

def capture_screen_sample(
    duration_seconds: float = SAMPLE_DURATION_SECONDS,
    interval_seconds: float = SAMPLE_INTERVAL_SECONDS,
    mss_monitor_index: int = MSS_MONITOR_INDEX,
    stop_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Capture a short screen sample from the configured monitor.

    This is the single source of truth for RaidVision sampling.

    Optional hooks:
    - stop_check: returns True when capture should stop early
    - progress_callback: receives per-frame progress dictionaries
    """
    frame_folder = create_raid_folder()

    saved_frames: list[str] = []
    captured_metrics: list[dict[str, float]] = []
    valid_count = 0
    rejected_count = 0
    rejection_counts: dict[str, int] = {}

    with mss.MSS() as sct:
        if mss_monitor_index >= len(sct.monitors):
            raise ValueError(
                f"MSS monitor index {mss_monitor_index} is invalid. "
                f"Available monitor count: {len(sct.monitors) - 1}"
            )

        monitor = sct.monitors[mss_monitor_index]
        start = time.time()
        frame_index = 1

        print()
        print("RaidVision Capture")
        print(f"Monitor index: {mss_monitor_index}")
        print(f"Sample duration: {duration_seconds} seconds")
        print("Sampling now. Tab into Tarkov and hold the view steady.")
        print()

        while time.time() - start < duration_seconds:
            if stop_check is not None and stop_check():
                print("Capture stopped early.")
                break

            frame = np.array(sct.grab(monitor))
            metrics = analyze_basic_frame(frame)

            is_valid, status = classify_capture_frame(metrics)

            if is_valid:
                valid_count += 1
            else:
                rejected_count += 1
                rejection_counts[status] = rejection_counts.get(status, 0) + 1

            saved_path = save_frame(frame, frame_folder, frame_index, status)

            captured_metrics.append(metrics)
            saved_frames.append(str(saved_path))

            progress = {
                "frame_index": frame_index,
                "status": status,
                "is_valid": is_valid,
                "valid_count": valid_count,
                "rejected_count": rejected_count,
                "saved_path": str(saved_path),
                "metrics": metrics,
            }

            if progress_callback is not None:
                progress_callback(progress)

            print(
                f"Frame:{frame_index:03} | "
                f"{status} | "
                f"Avg:{metrics['avg_brightness']:.1f} | "
                f"Shadow:{metrics['shadow_pct']:.1f}% | "
                f"Mid:{metrics['midtone_pct']:.1f}% | "
                f"Highlight:{metrics['highlight_pct']:.1f}% | "
                f"Contrast:{metrics['contrast_spread']:.1f}"
            )

            frame_index += 1
            time.sleep(interval_seconds)

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "frame_folder": str(frame_folder),
        "saved_frames": saved_frames,
        "captured_frames": len(saved_frames),
        "valid_frames": valid_count,
        "rejected_frames": rejected_count,
        "rejection_counts": rejection_counts,
        "sample_config": {
            "mss_monitor_index": mss_monitor_index,
            "duration_seconds": duration_seconds,
            "interval_seconds": interval_seconds,
            "frame_size": [FRAME_WIDTH, FRAME_HEIGHT],
        },
    }

    metadata_path = frame_folder / "capture_metadata.json"
    save_json(metadata_path, result)

    print()
    print("Capture Complete")
    print(f"Captured frames: {len(saved_frames)}")
    print(f"Valid frames: {valid_count}")
    print(f"Rejected frames: {rejected_count}")
    print(f"Saved frames to: {frame_folder}")
    print(f"Saved capture metadata to: {metadata_path}")

    return result


if __name__ == "__main__":
    capture_screen_sample()
