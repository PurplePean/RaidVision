import json
import os
from datetime import datetime, timezone


DISPLAY_INDEX = 1
OUTPUT_PATH = "control_profile.json"


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def convert_recommendation_to_profile(recommendation, display_index=DISPLAY_INDEX):
    gamma_adjustment = float(recommendation.get("gamma_adjustment", 0.0))
    brightness_adjustment = float(recommendation.get("brightness_adjustment", 0.0))
    contrast_adjustment = float(recommendation.get("contrast_adjustment", 0.0))

    brightness = 0.5 + (brightness_adjustment / 100.0)
    contrast = 0.5 + (contrast_adjustment / 100.0)
    gamma = 1.0 + gamma_adjustment

    profile = {
        "display_index": display_index,
        "brightness": round(clamp(brightness, 0.0, 1.0), 3),
        "contrast": round(clamp(contrast, 0.0, 1.0), 3),
        "gamma": round(clamp(gamma, 0.4, 2.8), 3),
        "reset": False,
        "heartbeat": datetime.now(timezone.utc).isoformat()
    }

    return profile


def atomic_write_json(data, output_path=OUTPUT_PATH):
    temp_path = output_path + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    os.replace(temp_path, output_path)


def write_control_profile_from_recommendation(recommendation):
    profile = convert_recommendation_to_profile(recommendation)
    atomic_write_json(profile)

    print("Wrote control profile:")
    print(json.dumps(profile, indent=2))

    return profile


if __name__ == "__main__":
    test_recommendation = {
        "gamma_adjustment": 0.35,
        "brightness_adjustment": 10,
        "contrast_adjustment": 3
    }

    write_control_profile_from_recommendation(test_recommendation)