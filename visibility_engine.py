from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


FRAME_ROOT = Path("frames")
DEBUG_ROOT = Path("debug_views")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

UI_MASK_REGIONS = [
    {
        "name": "top_hotbar",
        "x1": 0.29,
        "y1": 0.00,
        "x2": 0.77,
        "y2": 0.095,
    },
    {
        "name": "left_body_status",
        "x1": 0.00,
        "y1": 0.00,
        "x2": 0.085,
        "y2": 0.32,
    },
    {
        "name": "bottom_left_status_bars",
        "x1": 0.00,
        "y1": 0.91,
        "x2": 0.12,
        "y2": 1.00,
    },
    {
        "name": "bottom_right_text",
        "x1": 0.76,
        "y1": 0.70,
        "x2": 1.00,
        "y2": 1.00,
    },
]


ZONES = {
    "center_focus": {
        "x1": 0.30,
        "y1": 0.28,
        "x2": 0.70,
        "y2": 0.68,
        "base_weight": 3.0,
    },
    "lower_center": {
        "x1": 0.24,
        "y1": 0.55,
        "x2": 0.76,
        "y2": 0.88,
        "base_weight": 2.6,
    },
    "lower_third": {
        "x1": 0.12,
        "y1": 0.66,
        "x2": 0.88,
        "y2": 0.94,
        "base_weight": 2.0,
    },
    "upper_third": {
        "x1": 0.10,
        "y1": 0.08,
        "x2": 0.90,
        "y2": 0.34,
        "base_weight": 1.1,
    },
    "left_edge": {
        "x1": 0.00,
        "y1": 0.10,
        "x2": 0.10,
        "y2": 0.90,
        "base_weight": 0.5,
    },
    "right_edge": {
        "x1": 0.90,
        "y1": 0.10,
        "x2": 1.00,
        "y2": 0.90,
        "base_weight": 0.5,
    },
    "full_frame": {
        "x1": 0.00,
        "y1": 0.00,
        "x2": 1.00,
        "y2": 1.00,
        "base_weight": 1.0,
    },
}


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def clamp01(value: float) -> float:
    return clamp(value, 0.0, 1.0)


def to_bgr(frame: np.ndarray) -> np.ndarray:
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    if len(frame.shape) == 3 and frame.shape[2] == 3:
        return frame

    raise ValueError("Expected BGR or BGRA frame.")


def rect_from_region(region: dict[str, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1 = int(region["x1"] * width)
    y1 = int(region["y1"] * height)
    x2 = int(region["x2"] * width)
    y2 = int(region["y2"] * height)

    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))

    return x1, y1, x2, y2


def build_ui_mask(height: int, width: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=np.uint8)

    for region in UI_MASK_REGIONS:
        x1, y1, x2, y2 = rect_from_region(region, width, height)
        mask[y1:y2, x1:x2] = 0

    return mask


def metrics_from_pixels(gray_pixels: np.ndarray) -> dict[str, float]:
    if gray_pixels.size == 0:
        return empty_metrics()

    hist = cv2.calcHist([gray_pixels], [0], None, [256], [0, 256]).flatten()
    total = float(np.sum(hist))

    if total <= 0:
        return empty_metrics()

    return {
        "avg_brightness": float(np.mean(gray_pixels)),
        "deep_shadow_pct": float((np.sum(hist[0:40]) / total) * 100.0),
        "shadow_pct": float((np.sum(hist[0:90]) / total) * 100.0),
        "midtone_pct": float((np.sum(hist[90:170]) / total) * 100.0),
        "bright_pct": float((np.sum(hist[170:230]) / total) * 100.0),
        "highlight_pct": float((np.sum(hist[230:256]) / total) * 100.0),
        "contrast_spread": float(np.std(gray_pixels)),
    }


def empty_metrics() -> dict[str, float]:
    return {
        "avg_brightness": 0.0,
        "deep_shadow_pct": 0.0,
        "shadow_pct": 0.0,
        "midtone_pct": 0.0,
        "bright_pct": 0.0,
        "highlight_pct": 0.0,
        "contrast_spread": 0.0,
    }


def zone_metrics(gray: np.ndarray, ui_mask: np.ndarray, zone: dict[str, float]) -> dict[str, float]:
    height, width = gray.shape[:2]
    x1, y1, x2, y2 = rect_from_region(zone, width, height)

    zone_gray = gray[y1:y2, x1:x2]
    zone_mask = ui_mask[y1:y2, x1:x2]

    total_pixels = zone_gray.size

    if total_pixels == 0:
        metrics = empty_metrics()
        metrics["usable_pixel_ratio"] = 0.0
        return metrics

    usable_pixels = zone_gray[zone_mask == 1]
    usable_pixel_ratio = usable_pixels.size / total_pixels

    metrics = metrics_from_pixels(usable_pixels)
    metrics["usable_pixel_ratio"] = float(usable_pixel_ratio)

    return metrics


def pressure(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0

    return clamp01((value - low) / (high - low))


def inverse_pressure(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0

    return clamp01((high - value) / (high - low))


def detail_score(metrics: dict[str, float]) -> float:
    contrast = metrics["contrast_spread"]
    midtones = metrics["midtone_pct"]

    contrast_component = pressure(contrast, 12.0, 45.0)
    midtone_component = pressure(midtones, 5.0, 28.0)

    return clamp(0.35 + contrast_component * 0.45 + midtone_component * 0.20, 0.20, 1.15)


def effective_weight(zone_name: str, zone_data: dict[str, float]) -> float:
    base_weight = ZONES[zone_name]["base_weight"]
    usable_ratio = zone_data["usable_pixel_ratio"]

    if usable_ratio < 0.20:
        return 0.0

    return float(base_weight * usable_ratio * detail_score(zone_data))


def weighted_mean(items: list[tuple[float, float]]) -> float:
    usable_items = [(value, weight) for value, weight in items if weight > 0]

    if not usable_items:
        return 0.0

    total_weight = sum(weight for _, weight in usable_items)

    if total_weight <= 0:
        return 0.0

    return float(sum(value * weight for value, weight in usable_items) / total_weight)


def build_zone_summary(frame: np.ndarray) -> dict[str, Any]:
    bgr = to_bgr(frame)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]

    ui_mask = build_ui_mask(height, width)

    zones: dict[str, Any] = {}

    for zone_name, zone_def in ZONES.items():
        metrics = zone_metrics(gray, ui_mask, zone_def)
        metrics["effective_weight"] = effective_weight(zone_name, metrics)
        zones[zone_name] = metrics

    return {
        "width": width,
        "height": height,
        "ui_mask_usable_ratio": float(np.mean(ui_mask)),
        "zones": zones,
    }


def build_pressures(zone_summary: dict[str, Any]) -> dict[str, float]:
    zones = zone_summary["zones"]

    shadow_pressure = weighted_mean(
        [
            (
                pressure(zones["center_focus"]["shadow_pct"], 34.0, 78.0),
                zones["center_focus"]["effective_weight"] * 3.0,
            ),
            (
                pressure(zones["lower_center"]["shadow_pct"], 38.0, 82.0),
                zones["lower_center"]["effective_weight"] * 2.6,
            ),
            (
                pressure(zones["lower_third"]["shadow_pct"], 40.0, 84.0),
                zones["lower_third"]["effective_weight"] * 2.0,
            ),
            (
                pressure(zones["full_frame"]["shadow_pct"], 38.0, 78.0),
                zones["full_frame"]["effective_weight"] * 1.0,
            ),
        ]
    )

    deep_shadow_pressure = weighted_mean(
        [
            (
                pressure(zones["center_focus"]["deep_shadow_pct"], 12.0, 48.0),
                zones["center_focus"]["effective_weight"] * 3.0,
            ),
            (
                pressure(zones["lower_center"]["deep_shadow_pct"], 15.0, 52.0),
                zones["lower_center"]["effective_weight"] * 2.6,
            ),
            (
                pressure(zones["full_frame"]["deep_shadow_pct"], 12.0, 50.0),
                zones["full_frame"]["effective_weight"] * 1.0,
            ),
        ]
    )

    midtone_pressure = weighted_mean(
        [
            (
                inverse_pressure(zones["center_focus"]["midtone_pct"], 18.0, 38.0),
                zones["center_focus"]["effective_weight"] * 3.0,
            ),
            (
                inverse_pressure(zones["lower_center"]["midtone_pct"], 18.0, 36.0),
                zones["lower_center"]["effective_weight"] * 2.5,
            ),
            (
                inverse_pressure(zones["full_frame"]["midtone_pct"], 18.0, 38.0),
                zones["full_frame"]["effective_weight"] * 1.0,
            ),
        ]
    )

    highlight_pressure = weighted_mean(
        [
            (
                pressure(zones["upper_third"]["highlight_pct"], 6.0, 24.0),
                zones["upper_third"]["effective_weight"] * 2.4,
            ),
            (
                pressure(zones["center_focus"]["highlight_pct"], 5.0, 18.0),
                zones["center_focus"]["effective_weight"] * 1.4,
            ),
            (
                pressure(zones["full_frame"]["highlight_pct"], 6.0, 22.0),
                zones["full_frame"]["effective_weight"] * 1.0,
            ),
        ]
    )

    low_contrast_pressure = weighted_mean(
        [
            (
                inverse_pressure(zones["center_focus"]["contrast_spread"], 24.0, 52.0),
                zones["center_focus"]["effective_weight"] * 3.0,
            ),
            (
                inverse_pressure(zones["lower_center"]["contrast_spread"], 24.0, 50.0),
                zones["lower_center"]["effective_weight"] * 2.5,
            ),
            (
                inverse_pressure(zones["full_frame"]["contrast_spread"], 26.0, 55.0),
                zones["full_frame"]["effective_weight"] * 1.0,
            ),
        ]
    )

    backlight_pressure = clamp01(
        0.55 * pressure(zones["upper_third"]["highlight_pct"], 8.0, 25.0)
        + 0.45 * pressure(zones["lower_center"]["shadow_pct"], 42.0, 82.0)
    )

    night_pressure = clamp01(
        0.45 * pressure(zones["full_frame"]["shadow_pct"], 62.0, 92.0)
        + 0.35 * inverse_pressure(zones["full_frame"]["highlight_pct"], 2.0, 10.0)
        + 0.20 * inverse_pressure(zones["full_frame"]["avg_brightness"], 32.0, 82.0)
    )

    return {
        "shadow_pressure": round(shadow_pressure, 4),
        "deep_shadow_pressure": round(deep_shadow_pressure, 4),
        "midtone_pressure": round(midtone_pressure, 4),
        "highlight_pressure": round(highlight_pressure, 4),
        "low_contrast_pressure": round(low_contrast_pressure, 4),
        "backlight_pressure": round(backlight_pressure, 4),
        "night_pressure": round(night_pressure, 4),
    }


def build_lut_recommendation(pressures: dict[str, float]) -> dict[str, Any]:
    shadow_pressure = pressures["shadow_pressure"]
    deep_shadow_pressure = pressures["deep_shadow_pressure"]
    midtone_pressure = pressures["midtone_pressure"]
    highlight_pressure = pressures["highlight_pressure"]
    low_contrast_pressure = pressures["low_contrast_pressure"]
    backlight_pressure = pressures["backlight_pressure"]
    night_pressure = pressures["night_pressure"]

    shadow_lift = clamp(
        0.22
        + shadow_pressure * 0.30
        + deep_shadow_pressure * 0.22
        + backlight_pressure * 0.08
        + night_pressure * 0.12
        - highlight_pressure * 0.04,
        0.20,
        0.85,
    )

    midtone = clamp(
        0.14
        + midtone_pressure * 0.22
        + low_contrast_pressure * 0.18
        + shadow_pressure * 0.10,
        0.10,
        0.50,
    )

    highlight_protect = clamp(
        0.28
        + highlight_pressure * 0.32
        + backlight_pressure * 0.28
        + low_contrast_pressure * 0.05,
        0.25,
        0.85,
    )

    reasoning: list[str] = []

    if shadow_pressure > 0.55:
        reasoning.append("Playable zones are shadow heavy.")

    if deep_shadow_pressure > 0.45:
        reasoning.append("Deep shadows are elevated in important zones.")

    if midtone_pressure > 0.50:
        reasoning.append("Midtone information is below target.")

    if highlight_pressure > 0.45:
        reasoning.append("Highlights need protection.")

    if backlight_pressure > 0.45:
        reasoning.append("Scene appears partially backlit.")

    if low_contrast_pressure > 0.55:
        reasoning.append("Usable zones have low contrast.")

    if night_pressure > 0.50:
        reasoning.append("Overall sample trends night like or very dark.")

    if not reasoning:
        reasoning.append("Scene is already close to usable baseline.")

    return {
        "mode": "custom_lut",
        "shadow_lift": round(shadow_lift, 3),
        "midtone": round(midtone, 3),
        "highlight_protect": round(highlight_protect, 3),
        "reasoning": reasoning,
    }


def analyze_frame(frame: np.ndarray) -> dict[str, Any]:
    zone_summary = build_zone_summary(frame)
    pressures = build_pressures(zone_summary)
    lut_recommendation = build_lut_recommendation(pressures)

    return {
        "zone_summary": zone_summary,
        "pressures": pressures,
        "lut_recommendation": lut_recommendation,
    }


def average_dicts(dicts: list[dict[str, float]]) -> dict[str, float]:
    if not dicts:
        return {}

    keys = dicts[0].keys()

    return {
        key: round(float(sum(item[key] for item in dicts) / len(dicts)), 4)
        for key in keys
    }


def analyze_frames(frames: list[np.ndarray]) -> dict[str, Any]:
    frame_results = [analyze_frame(frame) for frame in frames]

    pressure_average = average_dicts(
        [result["pressures"] for result in frame_results]
    )

    lut_recommendation = build_lut_recommendation(pressure_average)

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "frame_count": len(frames),
        "average_pressures": pressure_average,
        "lut_recommendation": lut_recommendation,
        "frame_results": frame_results,
    }


def find_latest_frame_folder() -> Path | None:
    if not FRAME_ROOT.exists():
        return None

    folders = [path for path in FRAME_ROOT.iterdir() if path.is_dir()]

    if not folders:
        return None

    return max(folders, key=lambda path: path.stat().st_mtime)


def find_frame_files(folder: Path) -> list[Path]:
    files = [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(files)


def save_masked_debug(frame: np.ndarray, output_path: Path) -> None:
    bgr = to_bgr(frame)
    height, width = bgr.shape[:2]
    mask = build_ui_mask(height, width)

    masked = bgr.copy()
    masked[mask == 0] = (0, 0, 0)

    cv2.imwrite(str(output_path), masked)


def save_zone_overlay(frame: np.ndarray, output_path: Path) -> None:
    bgr = to_bgr(frame).copy()
    height, width = bgr.shape[:2]

    for region in UI_MASK_REGIONS:
        x1, y1, x2, y2 = rect_from_region(region, width, height)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(
            bgr,
            region["name"],
            (x1 + 3, min(height - 5, y1 + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    for zone_name, zone in ZONES.items():
        x1, y1, x2, y2 = rect_from_region(zone, width, height)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.putText(
            bgr,
            zone_name,
            (x1 + 3, min(height - 5, y1 + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), bgr)


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def main() -> int:
    latest_folder = find_latest_frame_folder()

    if latest_folder is None:
        print("No frame folders found.")
        print("Run python main.py first.")
        return 1

    frame_files = find_frame_files(latest_folder)

    if not frame_files:
        print(f"No image files found in: {latest_folder}")
        return 1

    output_folder = DEBUG_ROOT / latest_folder.name
    output_folder.mkdir(parents=True, exist_ok=True)

    selected_files = frame_files[: min(20, len(frame_files))]
    frames: list[np.ndarray] = []

    print("RaidVision Visibility Engine")
    print(f"Source folder: {latest_folder}")
    print(f"Frames found: {len(frame_files)}")
    print(f"Frames analyzed: {len(selected_files)}")
    print(f"Debug output: {output_folder}")
    print()

    for index, frame_path in enumerate(selected_files, start=1):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)

        if frame is None:
            print(f"Skipped unreadable frame: {frame_path}")
            continue

        frames.append(frame)

        save_masked_debug(
            frame,
            output_folder / f"{index:03}_{frame_path.stem}_masked.jpg",
        )

        save_zone_overlay(
            frame,
            output_folder / f"{index:03}_{frame_path.stem}_zones.jpg",
        )

    if not frames:
        print("No readable frames to analyze.")
        return 1

    report = analyze_frames(frames)

    report_path = output_folder / "visibility_engine_report.json"
    save_json(report_path, report)

    print("Average Pressures")
    for key, value in report["average_pressures"].items():
        print(f"{key}: {value}")

    print()
    print("Recommended Custom LUT")
    print(f"shadow_lift: {report['lut_recommendation']['shadow_lift']}")
    print(f"midtone: {report['lut_recommendation']['midtone']}")
    print(f"highlight_protect: {report['lut_recommendation']['highlight_protect']}")

    print()
    print("Reasoning")
    for reason in report["lut_recommendation"]["reasoning"]:
        print(reason)

    print()
    print(f"Saved report to: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())