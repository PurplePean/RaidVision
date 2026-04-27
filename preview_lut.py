from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image


FRAME_ROOT = Path("frames")
OUTPUT_ROOT = Path("lut_previews")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

PRESETS = [
    {
        "name": "shadow035_mid020_high040",
        "shadow_lift": 0.35,
        "midtone_boost": 0.20,
        "highlight_protect": 0.40,
    },
    {
        "name": "shadow055_mid030_high050",
        "shadow_lift": 0.55,
        "midtone_boost": 0.30,
        "highlight_protect": 0.50,
    },
    {
        "name": "shadow075_mid040_high060",
        "shadow_lift": 0.75,
        "midtone_boost": 0.40,
        "highlight_protect": 0.60,
    },
]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def apply_visibility_curve(
    x: float,
    shadow_lift: float,
    midtone_boost: float,
    highlight_protect: float,
) -> float:
    y = x

    if x < 0.35:
        t = x / 0.35
        lifted = x + shadow_lift * 0.35 * (1.0 - (t ** 1.8))
        y = max(y, lifted)

    if 0.25 <= x <= 0.75:
        center_distance = abs(x - 0.5) / 0.25
        mid_weight = clamp(1.0 - center_distance, 0.0, 1.0)
        y += midtone_boost * 0.12 * mid_weight

    if x > 0.72:
        t = (x - 0.72) / 0.28
        compression = highlight_protect * 0.18 * t * t
        y -= compression

    return clamp(y, 0.0, 1.0)


def apply_color_bias(
    y: float,
    original_x: float,
    green_boost: float,
    blue_boost: float,
) -> float:
    if original_x > 0.45:
        return y

    shadow_weight = clamp((0.45 - original_x) / 0.45, 0.0, 1.0)
    boost = (green_boost + blue_boost) * shadow_weight

    return clamp(y + boost, 0.0, 1.0)


def build_lut(
    shadow_lift: float,
    midtone_boost: float,
    highlight_protect: float,
) -> list[tuple[int, int, int]]:
    lut: list[tuple[int, int, int]] = []

    for i in range(256):
        x = i / 255.0

        y = apply_visibility_curve(
            x=x,
            shadow_lift=shadow_lift,
            midtone_boost=midtone_boost,
            highlight_protect=highlight_protect,
        )

        red_y = y
        green_y = apply_color_bias(y, x, green_boost=0.025, blue_boost=0.0)
        blue_y = apply_color_bias(y, x, green_boost=0.0, blue_boost=0.035)

        lut.append(
            (
                round(clamp(red_y, 0.0, 1.0) * 255),
                round(clamp(green_y, 0.0, 1.0) * 255),
                round(clamp(blue_y, 0.0, 1.0) * 255),
            )
        )

    return lut


def find_frames() -> list[Path]:
    if not FRAME_ROOT.exists():
        return []

    frames = [
        path
        for path in FRAME_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(frames, key=lambda path: path.stat().st_mtime, reverse=True)


def apply_lut_to_image(image: Image.Image, lut: list[tuple[int, int, int]]) -> Image.Image:
    rgb = image.convert("RGB")
    pixels = rgb.load()

    width, height = rgb.size

    for y in range(height):
        for x in range(width):
            red, green, blue = pixels[x, y]
            pixels[x, y] = (
                lut[red][0],
                lut[green][1],
                lut[blue][2],
            )

    return rgb


def save_side_by_side(original: Image.Image, preview: Image.Image, output_path: Path) -> None:
    original_rgb = original.convert("RGB")
    preview_rgb = preview.convert("RGB")

    width, height = original_rgb.size

    combined = Image.new("RGB", (width * 2, height))
    combined.paste(original_rgb, (0, 0))
    combined.paste(preview_rgb, (width, 0))

    combined.save(output_path, quality=90)


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    frames = find_frames()

    if not frames:
        print("No saved frames found.")
        print(f"Expected images inside: {FRAME_ROOT.resolve()}")
        print("Run main.py with SAVE_FRAMES = True first.")
        return 1

    selected_frames = frames[:5]

    print("RaidVision LUT Preview")
    print(f"Found frames: {len(frames)}")
    print(f"Previewing latest: {len(selected_frames)}")
    print(f"Output folder: {OUTPUT_ROOT.resolve()}")
    print()

    for frame_path in selected_frames:
        original = Image.open(frame_path)

        frame_stem = frame_path.stem
        original_output = OUTPUT_ROOT / f"{frame_stem}_original.jpg"
        original.convert("RGB").save(original_output, quality=90)

        print(f"Frame: {frame_path}")

        for preset in PRESETS:
            lut = build_lut(
                shadow_lift=preset["shadow_lift"],
                midtone_boost=preset["midtone_boost"],
                highlight_protect=preset["highlight_protect"],
            )

            preview = apply_lut_to_image(original, lut)

            preview_output = OUTPUT_ROOT / f"{frame_stem}_{preset['name']}.jpg"
            comparison_output = OUTPUT_ROOT / f"{frame_stem}_{preset['name']}_compare.jpg"

            preview.save(preview_output, quality=90)
            save_side_by_side(original, preview, comparison_output)

            print(f"  saved: {comparison_output}")

        print()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())