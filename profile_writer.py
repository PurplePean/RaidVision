from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


CONTROL_PROFILE_PATH = Path("control_profile.json")


def write_custom_lut_profile(
    recommendation: dict[str, Any],
    display_index: int,
    path: Path = CONTROL_PROFILE_PATH,
) -> dict[str, Any]:
    """
    Write a custom LUT profile to control_profile.json.
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
