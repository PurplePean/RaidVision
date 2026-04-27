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


def apply_custom_lut(
    recommendation: dict[str, Any],
    display_index: int,
) -> int:
    """
    Apply a custom LUT profile through the C# display controller.
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
