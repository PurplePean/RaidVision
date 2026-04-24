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

MIN_VALID_FRAMES = 10

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


def is_environment_frame(metrics):
    return metrics["midtone_pct"] > 5 and metrics["contrast_spread"] > 15


def classify_frame_type(metrics):
    if metrics["avg_brightness"] < 15:
        return "dark"
    if metrics["midtone_pct"] < 3:
        return "flat"
    if is_environment_frame(metrics):
        return "environment"
    return "mixed"


def is_valid_frame(metrics):
    if metrics["avg_brightness"] < 15 and metrics["shadow_pct"] > 95:
        return False, "black_loading_or_bad_capture"

    if metrics["midtone_pct"] < 1 and metrics["contrast_spread"] < 10:
        return False, "no_usable_detail"

    if metrics["highlight_pct"] > 85:
        return False, "mostly_sky_or_whiteout"

    return True, "valid"


def frame_weight(metrics):
    weight = 1.0

    weight += metrics["midtone_pct"] * 0.05

    if is_environment_frame(metrics):
        weight += 1.5

    if 35 <= metrics["contrast_spread"] <= 70:
        weight += 1.0

    if metrics["shadow_pct"] > 80:
        weight *= 0.7

    if metrics["highlight_pct"] > 12:
        weight *= 0.8

    if metrics["avg_brightness"] < 20:
        weight *= 0.75

    return max(0.1, float(weight))


def weighted_average(samples, weights):
    total_weight = sum(weights)
    return {
        key: float(
            sum(sample[key] * weight for sample, weight in zip(samples, weights))
            / total_weight
        )
        for key in samples[0]
    }


def average_metrics(samples):
    return {
        key: float(sum(sample[key] for sample in samples) / len(samples))
        for key in samples[0]
    }


def volatility_metrics(samples):
    return {
        f"{key}_volatility": float(np.std([sample[key] for sample in samples]))
        for key in samples[0]
    }


def classify_scene(metrics):
    if metrics["shadow_pct"] > 75 and metrics["midtone_pct"] < 10:
        return "extreme_shadow_heavy"

    if metrics["highlight_pct"] > 12:
        return "highlight_heavy"

    if metrics["shadow_pct"] > 55:
        return "shadow_heavy"

    if metrics["midtone_pct"] > 35 and metrics["highlight_pct"] < 8:
        return "balanced"

    return "mixed"


def confidence_score(volatility, valid_count, captured_count):
    valid_ratio = valid_count / captured_count if captured_count else 0

    penalty = 0
    penalty += volatility["shadow_pct_volatility"] * 1.8
    penalty += volatility["midtone_pct_volatility"] * 1.3
    penalty += volatility["highlight_pct_volatility"] * 1.8
    penalty += max(0, 0.65 - valid_ratio) * 50

    return round(max(0, min(100, 100 - penalty)), 1)


def visibility_score(metrics):
    score = 100

    score -= max(0, metrics["shadow_pct"] - 45) * 0.7
    score -= max(0, metrics["highlight_pct"] - 8) * 1.2
    score -= max(0, 25 - metrics["midtone_pct"]) * 0.8

    if metrics["contrast_spread"] < 35:
        score -= (35 - metrics["contrast_spread"]) * 0.5

    if metrics["contrast_spread"] > 80:
        score -= (metrics["contrast_spread"] - 80) * 0.4

    return round(max(0, min(100, score)), 1)


def build_recommendation(metrics, confidence):
    gamma_adjustment = 0.0
    brightness_adjustment = 0
    contrast_adjustment = 0
    reasons = []

    if metrics["shadow_pct"] > 75:
        gamma_adjustment += 0.30
        brightness_adjustment += 10
        reasons.append("Extreme shadow dominance")
    elif metrics["shadow_pct"] > 55:
        gamma_adjustment += 0.20
        brightness_adjustment += 6
        reasons.append("Shadow heavy scene")

    if metrics["deep_shadow_pct"] > 60:
        gamma_adjustment += 0.10
        reasons.append("Large amount of crushed deep shadow")

    if metrics["midtone_pct"] < 15:
        gamma_adjustment += 0.10
        contrast_adjustment -= 2
        reasons.append("Compressed midtones")
    elif metrics["midtone_pct"] < 25:
        gamma_adjustment += 0.05
        reasons.append("Midtones below target")

    if metrics["highlight_pct"] > 12:
        brightness_adjustment -= 8
        contrast_adjustment -= 3
        reasons.append("Highlight clipping risk")
    elif metrics["highlight_pct"] < 4 and metrics["shadow_pct"] > 55:
        brightness_adjustment += 3
        reasons.append("Highlights safe, room to lift exposure")

    if metrics["contrast_spread"] < 35:
        contrast_adjustment += 5
        reasons.append("Low contrast image")
    elif metrics["contrast_spread"] > 75:
        contrast_adjustment -= 4
        reasons.append("High contrast image")

    gamma_adjustment = min(gamma_adjustment, 0.35)
    brightness_adjustment = max(min(brightness_adjustment, 10), -10)
    contrast_adjustment = max(min(contrast_adjustment, 8), -8)

    if confidence < 60:
        gamma_adjustment *= 0.75
        brightness_adjustment = int(brightness_adjustment * 0.75)
        contrast_adjustment = int(contrast_adjustment * 0.75)
        reasons.append("Lower confidence sample, recommendation softened")

    return {
        "gamma_adjustment": round(gamma_adjustment, 2),
        "brightness_adjustment": brightness_adjustment,
        "contrast_adjustment": contrast_adjustment,
        "reasons": reasons,
    }


def save_frame(frame, folder, index, status):
    resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    filename = f"frame_{index:03}_{status}.jpg"
    path = os.path.join(folder, filename)
    cv2.imwrite(path, resized, [cv2.IMWRITE_JPEG_QUALITY, FRAME_QUALITY])
    return path


def sample_screen():
    frame_folder = create_raid_folder() if SAVE_FRAMES else None

    captured_samples = []
    valid_samples = []
    valid_weights = []
    saved_frames = []
    rejection_counts = {}
    frame_type_counts = {}

    with mss.MSS() as sct:
        monitor = sct.monitors[MSS_MONITOR_INDEX]
        start = time.time()
        frame_index = 1

        print("\nSampling scene...")

        while time.time() - start < SAMPLE_DURATION_SECONDS:
            img = np.array(sct.grab(monitor))
            metrics = analyze_histogram(img)

            captured_samples.append(metrics)

            frame_type = classify_frame_type(metrics)
            frame_type_counts[frame_type] = frame_type_counts.get(frame_type, 0) + 1

            valid, reason = is_valid_frame(metrics)

            if valid:
                weight = frame_weight(metrics)
                valid_samples.append(metrics)
                valid_weights.append(weight)
                status = f"valid_{frame_type}"
            else:
                weight = 0
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                status = reason

            if SAVE_FRAMES:
                saved_frames.append(save_frame(img, frame_folder, frame_index, status))

            print(
                f"Frame:{frame_index:03} | "
                f"{status} | "
                f"W:{weight:.2f} | "
                f"Avg:{metrics['avg_brightness']:.1f} | "
                f"Shadow:{metrics['shadow_pct']:.1f}% | "
                f"Mid:{metrics['midtone_pct']:.1f}% | "
                f"Highlight:{metrics['highlight_pct']:.1f}% | "
                f"Contrast:{metrics['contrast_spread']:.1f}"
            )

            frame_index += 1
            time.sleep(SAMPLE_INTERVAL_SECONDS)

    return {
        "frame_folder": frame_folder,
        "saved_frames": saved_frames,
        "captured_samples": captured_samples,
        "valid_samples": valid_samples,
        "valid_weights": valid_weights,
        "rejection_counts": rejection_counts,
        "frame_type_counts": frame_type_counts,
    }


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
    print("\nEnter what you actually used in RivaTuner.")
    print("Leave blank if unchanged.\n")

    return {
        "map": input("Map: ").strip(),
        "raid_notes": input("Raid notes: ").strip(),
        "actual_settings": {
            "brightness": input("RivaTuner Brightness: ").strip(),
            "contrast": input("RivaTuner Contrast: ").strip(),
            "gamma": input("RivaTuner Gamma: ").strip(),
        },
        "rating": input("Visibility rating 1 to 10: ").strip(),
        "notes": input("Notes: ").strip(),
    }


def main():
    result = sample_screen()

    captured_samples = result["captured_samples"]
    valid_samples = result["valid_samples"]
    valid_weights = result["valid_weights"]

    captured_count = len(captured_samples)
    valid_count = len(valid_samples)
    rejected_count = captured_count - valid_count

    print("\nSampling Summary")
    print("Captured frames:", captured_count)
    print("Valid frames:", valid_count)
    print("Rejected frames:", rejected_count)

    print("\nFrame Type Breakdown:")
    for frame_type, count in result["frame_type_counts"].items():
        print(f"{frame_type}: {count}")

    if result["rejection_counts"]:
        print("\nRejection Reasons:")
        for reason, count in result["rejection_counts"].items():
            print(f"{reason}: {count}")

    if valid_count < MIN_VALID_FRAMES:
        print("\nNot enough valid frames to build a reliable raid profile.")
        print("Try again once fully loaded into raid and looking at the environment.")

        failed_profile = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "status": "failed_not_enough_valid_frames",
            "sample_config": {
                "duration_seconds": SAMPLE_DURATION_SECONDS,
                "interval_seconds": SAMPLE_INTERVAL_SECONDS,
                "frames_captured": captured_count,
                "valid_frames": valid_count,
                "mss_monitor_index": MSS_MONITOR_INDEX,
            },
            "frame_folder": result["frame_folder"],
            "saved_frames": result["saved_frames"],
            "rejection_counts": result["rejection_counts"],
            "frame_type_counts": result["frame_type_counts"],
            "captured_average_metrics": average_metrics(captured_samples)
            if captured_samples
            else None,
        }

        if result["frame_folder"]:
            metadata_path = os.path.join(result["frame_folder"], "metadata.json")
            save_json(metadata_path, failed_profile)
            print(f"Saved failed metadata to: {metadata_path}")

        save_history(failed_profile)
        print(f"Saved failed run to {HISTORY_FILE}")
        return

    weighted_metrics = weighted_average(valid_samples, valid_weights)
    raw_valid_average = average_metrics(valid_samples)
    volatility = volatility_metrics(valid_samples)

    scene = classify_scene(weighted_metrics)
    confidence = confidence_score(volatility, valid_count, captured_count)
    visibility = visibility_score(weighted_metrics)
    recommendation = build_recommendation(weighted_metrics, confidence)

    raid_profile = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": "success",
        "sample_config": {
            "duration_seconds": SAMPLE_DURATION_SECONDS,
            "interval_seconds": SAMPLE_INTERVAL_SECONDS,
            "frames_captured": captured_count,
            "valid_frames": valid_count,
            "rejected_frames": rejected_count,
            "mss_monitor_index": MSS_MONITOR_INDEX,
            "save_frames": SAVE_FRAMES,
            "frame_size": [FRAME_WIDTH, FRAME_HEIGHT],
        },
        "frame_folder": result["frame_folder"],
        "saved_frames": result["saved_frames"],
        "rejection_counts": result["rejection_counts"],
        "frame_type_counts": result["frame_type_counts"],
        "scene": scene,
        "confidence_score": confidence,
        "visibility_score": visibility,
        "weighted_metrics": weighted_metrics,
        "raw_valid_average_metrics": raw_valid_average,
        "metrics_volatility": volatility,
        "recommendation": recommendation,
    }

    print("\nScene Analysis")
    print("Scene:", scene)
    print("Confidence Score:", confidence, "/ 100")
    print("Visibility Score:", visibility, "/ 100")
    print("Weighted Average Brightness:", round(weighted_metrics["avg_brightness"], 2))
    print("Weighted Deep Shadow %:", round(weighted_metrics["deep_shadow_pct"], 2))
    print("Weighted Shadow %:", round(weighted_metrics["shadow_pct"], 2))
    print("Weighted Midtone %:", round(weighted_metrics["midtone_pct"], 2))
    print("Weighted Bright %:", round(weighted_metrics["bright_pct"], 2))
    print("Weighted Highlight %:", round(weighted_metrics["highlight_pct"], 2))
    print("Weighted Contrast Spread:", round(weighted_metrics["contrast_spread"], 2))

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
        print(reason)

    if result["frame_folder"]:
        metadata_path = os.path.join(result["frame_folder"], "metadata.json")
        save_json(metadata_path, raid_profile)
        print(f"\nSaved frames to: {result['frame_folder']}")
        print(f"Saved metadata to: {metadata_path}")

    user_log = ask_user_log()
    raid_profile["user_log"] = user_log

    save_history(raid_profile)
    print(f"\nSaved raid log to {HISTORY_FILE}")


if __name__ == "__main__":
    main()