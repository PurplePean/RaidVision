from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent

CONTROL_PROJECT = (
    REPO_ROOT
    / "control"
    / "raidvision_control"
    / "raidvision_control_test.csproj"
)


def run_control_command(args: list[str]) -> int:
    """
    Run the C# RaidVision control CLI.
    """
    command = [
        "dotnet",
        "run",
        "--project",
        str(CONTROL_PROJECT),
        "--",
        *args,
    ]

    print()
    print("Running control command:")
    print(" ".join(command))
    print()

    completed = subprocess.run(command, cwd=REPO_ROOT)
    return completed.returncode


def apply_display_color_stack(
    profile: dict[str, Any],
    display_index: int,
) -> int:
    """
    Apply brightness, contrast, and gamma through the C# controller.

    Vibrance is currently logged only and is not applied here yet.
    """
    return run_control_command(
        [
            "apply",
            "--display",
            str(display_index),
            "--brightness",
            str(profile["brightness"]),
            "--contrast",
            str(profile["contrast"]),
            "--gamma",
            str(profile["gamma"]),
        ]
    )


def apply_custom_lut(
    recommendation: dict[str, Any],
    display_index: int,
) -> int:
    """
    Legacy Custom LUT apply path.

    Kept for branch compatibility, but this is not the primary path on
    feature/display-color-stack.
    """
    return run_control_command(
        [
            "custom-lut",
            "--display",
            str(display_index),
            "--shadow-lift",
            str(recommendation["shadow_lift"]),
            "--midtone",
            str(recommendation["midtone"]),
            "--highlight-protect",
            str(recommendation["highlight_protect"]),
        ]
    )


def reset_display(display_index: int) -> int:
    """
    Reset the selected display back to neutral.
    """
    return run_control_command(
        [
            "reset",
            "--display",
            str(display_index),
        ]
    )
