from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


CONTROL_PROFILE_PATH = Path("control_profile.json")


def write_display_color_profile(
    profile: dict[str, Any],
    display_index: int,
    path: Path = CONTROL_PROFILE_PATH,
) -> dict[str, Any]:
    """
    Write the active display color stack profile.

    Brightness, contrast, and gamma are applied through the C# controller.
    Vibrance is logged only until NVIDIA vibrance control is wired.
    """
    output = {
        "mode": "display_color_stack",
        "display_index": display_index,
        "brightness": profile["brightness"],
        "contrast": profile["contrast"],
        "gamma": profile["gamma"],
        "vibrance": profile.get("vibrance", 50),
        "vibrance_apply_status": "logged_only",
        "reset": False,
        "heartbeat": datetime.now().isoformat(timespec="seconds"),
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    return output


def write_custom_lut_profile(
    recommendation: dict[str, Any],
    display_index: int,
    path: Path = CONTROL_PROFILE_PATH,
) -> dict[str, Any]:
    """
    Legacy Custom LUT writer.

    Kept for compatibility with the Custom LUT experiment branch.
    """
    profile = {
        "mode": "custom_lut",
        "display_index": display_index,
        "shadow_lift": recommendation["shadow_lift"],
        "midtone": recommendation["midtone"],
        "highlight_protect": recommendation["highlight_protect"],
        "reset": False,
        "heartbeat": datetime.now().isoformat(timespec="seconds"),
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2)

    return profile
