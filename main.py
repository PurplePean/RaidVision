import json
import time
from datetime import datetime

import cv2
import mss
import numpy as np


MSS_MONITOR_INDEX = 3
SAMPLE_DURATION_SECONDS = 10
SAMPLE_INTERVAL_SECONDS = 0.5
HISTORY_FILE = "history.json"


def analyze_histogram(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = np.sum(hist)

    return {
        "avg_brightness": float(np.mean(gray)),
        "deep_shadow_pct": float((np.sum(hist[0:40]) / total) * 100),
        "shadow_pct": float((np.sum(hist[0:90]) / total) * 100),
        "midtone_pct": float((np.sum(hist[90:170]) / total) * 100),
        "bright_pct": float((np.sum(hist[170:230]) / total) * 100),
        "highlight_pct": float((np.sum(hist[230:256]) / total) * 100),
        "contrast_spread": float(np.std(gray)),
    }


def average_metrics(samples):
    return {
        key: sum(sample[key] for sample in samples) / len(samples)
        for key in samples[0]
    }


def classify_scene(m):
    if m["shadow_pct"] > 75 and m["midtone_pct"] < 10:
        return "extreme_shadow_heavy"

    if m["highlight_pct"] > 12:
        return "highlight_heavy"

    if m["shadow_pct"] > 55:
        return "shadow_heavy"

    if m["midtone_pct"] > 35 and m["highlight_pct"] < 8:
        return "balanced"

    return "mixed"


def visibility_score(m):
    score = 100

    score -= max(0, m["shadow_pct"] - 45) * 0.7
    score -= max(0, m["highlight_pct"] - 8) * 1.2
    score -= max(0, 25 - m["midtone_pct"]) * 0.8

    if m["contrast_spread"] < 35:
        score -= (35 - m["contrast_spread"]) * 0.5

    if m["contrast_spread"] > 80:
        score -= (m["contrast_spread"] - 80) * 0.4

    return round(max(0, min(100, score)), 1)


def build_recommendation(m):
    gamma_adjustment = 0.0
    brightness_adjustment = 0
    contrast_adjustment = 0
    reasons = []

    if m["shadow_pct"] > 75:
        gamma_adjustment += 0.30
        brightness_adjustment += 10
        reasons.append("Extreme shadow dominance detected")
    elif m["shadow_pct"] > 55:
        gamma_adjustment += 0.20
        brightness_adjustment += 6
        reasons.append("Scene is shadow heavy")

    if m["deep_shadow_pct"] > 60:
        gamma_adjustment += 0.10
        reasons.append("Large amount of crushed deep shadow")

    if m["midtone_pct"] < 15:
        gamma_adjustment += 0.10
        contrast_adjustment -= 2
        reasons.append("Midtones are too compressed")
    elif m["midtone_pct"] < 25:
        gamma_adjustment += 0.05
        reasons.append("Midtones are below target")

    if m["highlight_pct"] > 12:
        brightness_adjustment -= 8
        contrast_adjustment -= 3
        reasons.append("Highlights are too strong")
    elif m["highlight_pct"] < 4 and m["shadow_pct"] > 55:
        brightness_adjustment += 3
        reasons.append("Highlights are safe, room to lift exposure")

    if m["contrast_spread"] < 35:
        contrast_adjustment += 5
        reasons.append("Low contrast image")
    elif m["contrast_spread"] > 75:
        contrast_adjustment -= 4
        reasons.append("High contrast image, avoid crushing more shadow")

    gamma_adjustment = round(gamma_adjustment, 2)

    return {
        "gamma_adjustment": gamma_adjustment,
        "brightness_adjustment": brightness_adjustment,
        "contrast_adjustment": contrast_adjustment,
        "reasons": reasons,
    }


def sample_screen():
    with mss.MSS() as sct:
        monitor = sct.monitors[MSS_MONITOR_INDEX]
        samples = []
        start = time.time()

        print("\nSampling scene...")

        while time.time() - start < SAMPLE_DURATION_SECONDS:
            img = np.array(sct.grab(monitor))
            metrics = analyze_histogram(img)
            samples.append(metrics)

            print(
                f"Avg:{metrics['avg_brightness']:.1f} | "
                f"Deep:{metrics['deep_shadow_pct']:.1f}% | "
                f"Shadow:{metrics['shadow_pct']:.1f}% | "
                f"Mid:{metrics['midtone_pct']:.1f}% | "
                f"Highlight:{metrics['highlight_pct']:.1f}% | "
                f"Contrast:{metrics['contrast_spread']:.1f}"
            )

            time.sleep(SAMPLE_INTERVAL_SECONDS)

    return average_metrics(samples)


def save_log(entry):
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            history = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    history.append(entry)

    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def ask_user_log(metrics, scene, score, recommendation):
    print("\nEnter what you actually used.")
    print("Leave blank if you did not change a setting.\n")

    map_name = input("Map: ").strip()
    raid_time = input("Raid time/weather notes: ").strip()

    actual_gamma = input("Actual Gamma setting: ").strip()
    actual_brightness = input("Actual Brightness setting: ").strip()
    actual_contrast = input("Actual Contrast setting: ").strip()
    actual_saturation = input("Actual Saturation / Digital Vibrance setting: ").strip()

    rating = input("Visibility rating 1 to 10: ").strip()
    notes = input("Notes: ").strip()

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "map": map_name,
        "raid_time_weather": raid_time,
        "scene": scene,
        "visibility_score": score,
        "metrics": metrics,
        "recommendation": recommendation,
        "actual_user_settings": {
            "gamma": actual_gamma,
            "brightness": actual_brightness,
            "contrast": actual_contrast,
            "saturation_or_digital_vibrance": actual_saturation,
        },
        "rating": rating,
        "notes": notes,
    }

    save_log(entry)
    print(f"\nSaved to {HISTORY_FILE}")


def main():
    metrics = sample_screen()
    scene = classify_scene(metrics)
    score = visibility_score(metrics)
    recommendation = build_recommendation(metrics)

    print("\nScene Analysis")
    print("Scene:", scene)
    print("Visibility Score:", score, "/ 100")
    print("Average Brightness:", round(metrics["avg_brightness"], 2))
    print("Deep Shadow %:", round(metrics["deep_shadow_pct"], 2))
    print("Shadow %:", round(metrics["shadow_pct"], 2))
    print("Midtone %:", round(metrics["midtone_pct"], 2))
    print("Bright %:", round(metrics["bright_pct"], 2))
    print("Highlight %:", round(metrics["highlight_pct"], 2))
    print("Contrast Spread:", round(metrics["contrast_spread"], 2))

    print("\nTargeted Dynamic Recommendation")
    print("Gamma Adjustment:", recommendation["gamma_adjustment"])
    print("Brightness Adjustment:", recommendation["brightness_adjustment"])
    print("Contrast Adjustment:", recommendation["contrast_adjustment"])

    print("\nReasoning:")
    for reason in recommendation["reasons"]:
        print("-", reason)

    ask_user_log(metrics, scene, score, recommendation)


if __name__ == "__main__":
    main()