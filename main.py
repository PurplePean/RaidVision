import json
import os
import time
from datetime import datetime

import cv2
import mss
import numpy as np


MSS_MONITOR_INDEX = 3

SAMPLE_DURATION_SECONDS = 20
SAMPLE_INTERVAL_SECONDS = 0.5

SAVE_FRAMES = True
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
FRAME_QUALITY = 80

HISTORY_FILE = "history.json"
FRAME_ROOT = "frames"


def create_raid_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(FRAME_ROOT, f"raid_{timestamp}")
    os.makedirs(folder, exist_ok=True)
    return folder


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
        key: float(sum(sample[key] for sample in samples) / len(samples))
        for key in samples[0]
    }


def volatility_metrics(samples):
    keys = samples[0].keys()

    return {
        f"{key}_volatility": float(np.std([sample[key] for sample in samples]))
        for key in keys
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


def confidence_score(volatility):
    shadow_vol = volatility["shadow_pct_volatility"]
    midtone_vol = volatility["midtone_pct_volatility"]
    highlight_vol = volatility["highlight_pct_volatility"]

    penalty = 0
    penalty += shadow_vol * 2.0
    penalty += midtone_vol * 1.5
    penalty += highlight_vol * 2.0

    return round(max(0, min(100, 100 - penalty)), 1)


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


def build_recommendation(m, confidence):
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
        reasons.append("Midtones are heavily compressed")
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

    if confidence < 60:
        gamma_adjustment *= 0.75
        brightness_adjustment = int(brightness_adjustment * 0.75)
        contrast_adjustment = int(contrast_adjustment * 0.75)
        reasons.append("Low scene confidence, recommendation softened")

    return {
        "gamma_adjustment": round(gamma_adjustment, 2),
        "brightness_adjustment": brightness_adjustment,
        "contrast_adjustment": contrast_adjustment,
        "reasons": reasons,
    }


def save_frame(frame, folder, index):
    resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    path = os.path.join(folder, f"frame_{index:03}.jpg")
    cv2.imwrite(path, resized, [cv2.IMWRITE_JPEG_QUALITY, FRAME_QUALITY])
    return path


def sample_screen():
    frame_folder = create_raid_folder() if SAVE_FRAMES else None

    with mss.MSS() as sct:
        monitor = sct.monitors[MSS_MONITOR_INDEX]
        samples = []
        saved_frames = []

        start = time.time()
        frame_index = 1

        print("\nSampling scene...")

        while time.time() - start < SAMPLE_DURATION_SECONDS:
            img = np.array(sct.grab(monitor))
            metrics = analyze_histogram(img)
            samples.append(metrics)

            if SAVE_FRAMES:
                frame_path = save_frame(img, frame_folder, frame_index)
                saved_frames.append(frame_path)

            print(
                f"Frame:{frame_index:03} | "
                f"Avg:{metrics['avg_brightness']:.1f} | "
                f"Deep:{metrics['deep_shadow_pct']:.1f}% | "
                f"Shadow:{metrics['shadow_pct']:.1f}% | "
                f"Mid:{metrics['midtone_pct']:.1f}% | "
                f"Highlight:{metrics['highlight_pct']:.1f}% | "
                f"Contrast:{metrics['contrast_spread']:.1f}"
            )

            frame_index += 1
            time.sleep(SAMPLE_INTERVAL_SECONDS)

    return samples, frame_folder, saved_frames


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_history(entry):
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            history = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    history.append(entry)

    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def ask_user_log():
    print("\nEnter what you actually used.")
    print("Leave blank if you did not change a setting.\n")

    return {
        "map": input("Map: ").strip(),
        "raid_time_weather": input("Raid time/weather notes: ").strip(),
        "actual_settings": {
            "gamma": input("Actual Gamma setting: ").strip(),
            "brightness": input("Actual Brightness setting: ").strip(),
            "contrast": input("Actual Contrast setting: ").strip(),
            "saturation_or_digital_vibrance": input("Actual Saturation / Digital Vibrance setting: ").strip(),
        },
        "rating": input("Visibility rating 1 to 10: ").strip(),
        "notes": input("Notes: ").strip(),
    }


def main():
    samples, frame_folder, saved_frames = sample_screen()

    metrics = average_metrics(samples)
    volatility = volatility_metrics(samples)

    scene = classify_scene(metrics)
    confidence = confidence_score(volatility)
    visibility = visibility_score(metrics)
    recommendation = build_recommendation(metrics, confidence)

    raid_profile = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sample_config": {
            "duration_seconds": SAMPLE_DURATION_SECONDS,
            "interval_seconds": SAMPLE_INTERVAL_SECONDS,
            "frames_sampled": len(samples),
            "mss_monitor_index": MSS_MONITOR_INDEX,
            "save_frames": SAVE_FRAMES,
            "frame_size": [FRAME_WIDTH, FRAME_HEIGHT],
        },
        "frame_folder": frame_folder,
        "saved_frames": saved_frames,
        "scene": scene,
        "confidence_score": confidence,
        "visibility_score": visibility,
        "metrics_average": metrics,
        "metrics_volatility": volatility,
        "recommendation": recommendation,
    }

    print("\nScene Analysis")
    print("Scene:", scene)
    print("Confidence Score:", confidence, "/ 100")
    print("Visibility Score:", visibility, "/ 100")
    print("Average Brightness:", round(metrics["avg_brightness"], 2))
    print("Deep Shadow %:", round(metrics["deep_shadow_pct"], 2))
    print("Shadow %:", round(metrics["shadow_pct"], 2))
    print("Midtone %:", round(metrics["midtone_pct"], 2))
    print("Bright %:", round(metrics["bright_pct"], 2))
    print("Highlight %:", round(metrics["highlight_pct"], 2))
    print("Contrast Spread:", round(metrics["contrast_spread"], 2))

    print("\nStability")
    print("Shadow Volatility:", round(volatility["shadow_pct_volatility"], 2))
    print("Midtone Volatility:", round(volatility["midtone_pct_volatility"], 2))
    print("Highlight Volatility:", round(volatility["highlight_pct_volatility"], 2))

    print("\nTargeted Dynamic Recommendation")
    print("Gamma Adjustment:", recommendation["gamma_adjustment"])
    print("Brightness Adjustment:", recommendation["brightness_adjustment"])
    print("Contrast Adjustment:", recommendation["contrast_adjustment"])

    print("\nReasoning:")
    for reason in recommendation["reasons"]:
        print("-", reason)

    if frame_folder:
        metadata_path = os.path.join(frame_folder, "metadata.json")
        save_json(metadata_path, raid_profile)
        print(f"\nSaved frames to: {frame_folder}")
        print(f"Saved metadata to: {metadata_path}")

    user_log = ask_user_log()
    raid_profile["user_log"] = user_log

    save_history(raid_profile)
    print(f"\nSaved raid log to {HISTORY_FILE}")


if __name__ == "__main__":
    main()