from __future__ import annotations

import json
import os
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

import cv2
import mss
import numpy as np

from capture import (
    MSS_MONITOR_INDEX,
    SAMPLE_DURATION_SECONDS,
    SAMPLE_INTERVAL_SECONDS,
    analyze_basic_frame,
    classify_capture_frame,
    create_raid_folder,
    save_frame,
)
from control_runner import apply_display_color_stack, reset_display
from profile_writer import write_display_color_profile
from visibility_engine import (
    analyze_frames,
    find_frame_files,
    find_latest_frame_folder,
    save_masked_debug,
    save_zone_overlay,
)


DEFAULT_DISPLAY_INDEX = 1
DEBUG_ROOT = Path("debug_views")
FEEDBACK_ROOT = Path("feedback_logs")


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


class RaidVisionGUI:
    """
    RaidVision Display Color Stack GUI.

    Brightness, contrast, and gamma are applied through the C# controller.
    Vibrance is logged only until NVIDIA vibrance control is wired.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("RaidVision Display Color Stack")
        self.root.geometry("980x720")

        self.display_index = DEFAULT_DISPLAY_INDEX

        self.capture_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.is_sampling = False

        self.current_frame_folder: Path | None = None
        self.current_report: dict[str, Any] | None = None
        self.engine_recommendation: dict[str, Any] | None = None
        self.current_debug_folder: Path | None = None

        self.brightness_var = tk.DoubleVar(value=0.50)
        self.contrast_var = tk.DoubleVar(value=0.50)
        self.gamma_var = tk.DoubleVar(value=1.00)
        self.vibrance_var = tk.DoubleVar(value=50.0)

        self.slider_debounce_job = None
        self.is_live_apply_running = False
        self.pending_live_apply = False
        self.suppress_slider_apply = False

        self.map_var = tk.StringVar(value="Unknown")
        self.time_of_day_var = tk.StringVar(value="Unknown")
        self.weather_var = tk.StringVar(value="Unknown")
        self.night_vision_var = tk.StringVar(value="Off")
        self.thermal_var = tk.StringVar(value="Off")

        self.rating_var = tk.StringVar(value="")
        self.washed_out_var = tk.BooleanVar(value=False)
        self.too_dark_var = tk.BooleanVar(value=False)
        self.too_bright_var = tk.BooleanVar(value=False)

        self.status_var = tk.StringVar(value="Ready")
        self.recommendation_text = tk.StringVar(value="No recommendation yet.")
        self.pressure_labels: dict[str, tk.StringVar] = {}

        self.build_ui()

    def build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=2)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=12)
        middle = ttk.Frame(self.root, padding=12)
        right = ttk.Frame(self.root, padding=12)

        left.grid(row=0, column=0, sticky="nsew")
        middle.grid(row=0, column=1, sticky="nsew")
        right.grid(row=0, column=2, sticky="nsew")

        self.build_buttons(left)
        self.build_sliders(middle)
        self.build_feedback_panel(middle)
        self.build_pressure_panel(right)

        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor="w",
            padding=6,
        )
        status_bar.grid(row=1, column=0, columnspan=3, sticky="ew")

    def build_buttons(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Workflow", font=("Segoe UI", 14, "bold")).pack(anchor="w")

        ttk.Button(parent, text="Start RaidVision", command=self.start_sample).pack(fill="x", pady=5)
        ttk.Button(parent, text="Reset Display", command=self.reset).pack(fill="x", pady=5)
        ttk.Button(parent, text="Save Preferred", command=self.save_feedback).pack(fill="x", pady=5)

        ttk.Separator(parent).pack(fill="x", pady=10)

        ttk.Label(parent, text="Advanced", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Button(parent, text="Analyze Latest", command=self.analyze_latest).pack(fill="x", pady=5)
        ttk.Button(parent, text="Open Debug Folder", command=self.open_debug_folder).pack(fill="x", pady=5)

        ttk.Separator(parent).pack(fill="x", pady=10)

        ttk.Label(parent, text="Behavior", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(parent, text="Start samples, analyzes, and auto applies.").pack(anchor="w")
        ttk.Label(parent, text="Sliders live apply after a short pause.").pack(anchor="w")
        ttk.Label(parent, text="Vibrance is logged only for now.").pack(anchor="w")

    def build_sliders(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Display Color Stack", font=("Segoe UI", 14, "bold")).pack(anchor="w")

        ttk.Label(
            parent,
            textvariable=self.recommendation_text,
            justify="left",
            padding=8,
        ).pack(fill="x", pady=6)

        ttk.Separator(parent).pack(fill="x", pady=10)

        self.add_slider(parent, "Brightness", self.brightness_var, 0.00, 1.00)
        self.add_slider(parent, "Contrast", self.contrast_var, 0.00, 1.00)
        self.add_slider(parent, "Gamma", self.gamma_var, 0.40, 2.80)
        self.add_slider(parent, "Vibrance, logged only", self.vibrance_var, 0.0, 100.0)

    def add_slider(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.DoubleVar,
        minimum: float,
        maximum: float,
    ) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=8)

        value_label = ttk.Label(frame, width=8)
        value_label.pack(side="right")

        ttk.Label(frame, text=label).pack(anchor="w")

        slider = ttk.Scale(
            frame,
            from_=minimum,
            to=maximum,
            variable=variable,
            orient="horizontal",
            command=lambda _value: self.on_slider_changed(value_label, variable),
        )
        slider.pack(fill="x")

        value_label.config(text=f"{variable.get():.3f}")

    def build_feedback_panel(self, parent: ttk.Frame) -> None:
        ttk.Separator(parent).pack(fill="x", pady=16)

        ttk.Label(parent, text="Scene Context", font=("Segoe UI", 14, "bold")).pack(anchor="w")

        context_grid = ttk.Frame(parent)
        context_grid.pack(fill="x", pady=6)

        self.add_combo(context_grid, "Map:", self.map_var, [
            "Unknown", "Customs", "Woods", "Shoreline", "Interchange", "Reserve",
            "Lighthouse", "Streets", "Ground Zero", "Factory", "Labs", "Other",
        ], 0)

        self.add_combo(context_grid, "Time:", self.time_of_day_var, ["Unknown", "Day", "Dusk", "Night"], 1)
        self.add_combo(context_grid, "Weather:", self.weather_var, ["Unknown", "Clear", "Overcast", "Rain", "Fog", "Snow"], 2)
        self.add_combo(context_grid, "Night Vision:", self.night_vision_var, ["Off", "On"], 3)
        self.add_combo(context_grid, "Thermal:", self.thermal_var, ["Off", "On"], 4)

        ttk.Separator(parent).pack(fill="x", pady=12)

        ttk.Label(parent, text="Feedback", font=("Segoe UI", 14, "bold")).pack(anchor="w")

        rating_frame = ttk.Frame(parent)
        rating_frame.pack(fill="x", pady=5)

        ttk.Label(rating_frame, text="Rating 1 to 10:").pack(side="left")
        ttk.Entry(rating_frame, textvariable=self.rating_var, width=8).pack(side="left", padx=8)

        ttk.Checkbutton(parent, text="Washed out", variable=self.washed_out_var).pack(anchor="w")
        ttk.Checkbutton(parent, text="Too dark", variable=self.too_dark_var).pack(anchor="w")
        ttk.Checkbutton(parent, text="Too bright", variable=self.too_bright_var).pack(anchor="w")

        ttk.Label(parent, text="Notes:").pack(anchor="w", pady=(8, 0))
        self.notes_box = tk.Text(parent, height=5, wrap="word")
        self.notes_box.pack(fill="both", expand=True)

    def add_combo(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        values: list[str],
        row: int,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Combobox(
            parent,
            textvariable=variable,
            values=values,
            width=18,
        ).grid(row=row, column=1, sticky="w", pady=3)

    def build_pressure_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Pressure Readout", font=("Segoe UI", 14, "bold")).pack(anchor="w")

        for name in [
            "shadow_pressure",
            "deep_shadow_pressure",
            "midtone_pressure",
            "highlight_pressure",
            "low_contrast_pressure",
            "backlight_pressure",
            "night_pressure",
        ]:
            value_var = tk.StringVar(value=f"{name}: --")
            self.pressure_labels[name] = value_var
            ttk.Label(parent, textvariable=value_var).pack(anchor="w", pady=3)

    def start_sample(self) -> None:
        if self.is_sampling:
            self.set_status("Already sampling.")
            return

        self.stop_event.clear()
        self.is_sampling = True
        self.current_frame_folder = create_raid_folder()

        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()

        self.set_status("Sampling started. Auto analyze and auto apply will run when complete.")

    def capture_loop(self) -> None:
        assert self.current_frame_folder is not None

        saved_frames: list[str] = []
        valid_count = 0
        rejected_count = 0
        rejection_counts: dict[str, int] = {}

        try:
            with mss.MSS() as sct:
                monitor = sct.monitors[MSS_MONITOR_INDEX]
                start = time.time()
                frame_index = 1

                while not self.stop_event.is_set():
                    if time.time() - start >= SAMPLE_DURATION_SECONDS:
                        break

                    frame = np.array(sct.grab(monitor))
                    metrics = analyze_basic_frame(frame)
                    is_valid, status = classify_capture_frame(metrics)

                    if is_valid:
                        valid_count += 1
                    else:
                        rejected_count += 1
                        rejection_counts[status] = rejection_counts.get(status, 0) + 1

                    saved_path = save_frame(frame, self.current_frame_folder, frame_index, status)
                    saved_frames.append(str(saved_path))

                    self.set_status(
                        f"Sampling frame {frame_index:03} | {status} | "
                        f"valid {valid_count} | rejected {rejected_count}"
                    )

                    frame_index += 1
                    time.sleep(SAMPLE_INTERVAL_SECONDS)

            capture_metadata = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "frame_folder": str(self.current_frame_folder),
                "saved_frames": saved_frames,
                "captured_frames": len(saved_frames),
                "valid_frames": valid_count,
                "rejected_frames": rejected_count,
                "rejection_counts": rejection_counts,
                "sample_config": {
                    "mss_monitor_index": MSS_MONITOR_INDEX,
                    "duration_seconds": SAMPLE_DURATION_SECONDS,
                    "interval_seconds": SAMPLE_INTERVAL_SECONDS,
                },
            }

            self.save_json(self.current_frame_folder / "capture_metadata.json", capture_metadata)

        except Exception as error:
            self.root.after(0, lambda: messagebox.showerror("Capture Error", str(error)))

        finally:
            self.is_sampling = False
            self.root.after(0, self.analyze_current_folder)

    def analyze_current_folder(self) -> None:
        if self.current_frame_folder is None:
            self.set_status("No current frame folder to analyze.")
            return

        self.analyze_folder(self.current_frame_folder)

    def analyze_latest(self) -> None:
        latest = find_latest_frame_folder()

        if latest is None:
            messagebox.showwarning("No Frames", "No saved frame folder found.")
            return

        self.current_frame_folder = latest
        self.analyze_folder(latest)

    def analyze_folder(self, frame_folder: Path) -> None:
        try:
            frame_paths = find_frame_files(frame_folder)
            valid_paths = [path for path in frame_paths if "valid" in path.stem.lower()]
            selected_paths = valid_paths if valid_paths else frame_paths

            frames = []
            for path in selected_paths:
                frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if frame is not None:
                    frames.append(frame)

            if not frames:
                messagebox.showwarning("No Frames", f"No readable frames found in {frame_folder}.")
                return

            report = analyze_frames(frames)
            recommendation = self.build_display_recommendation(report)

            control_profile = write_display_color_profile(recommendation, self.display_index)

            report["frame_folder"] = str(frame_folder)
            report["frame_files_used"] = [str(path) for path in selected_paths]
            report["display_color_recommendation"] = recommendation
            report["control_profile"] = control_profile

            self.save_json(frame_folder / "visibility_report.json", report)

            debug_folder = DEBUG_ROOT / frame_folder.name
            debug_folder.mkdir(parents=True, exist_ok=True)
            self.current_debug_folder = debug_folder

            for index, path in enumerate(selected_paths[:10], start=1):
                frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if frame is None:
                    continue

                save_masked_debug(frame, debug_folder / f"{index:03}_{path.stem}_masked.jpg")
                save_zone_overlay(frame, debug_folder / f"{index:03}_{path.stem}_zones.jpg")

            self.current_report = report
            self.engine_recommendation = recommendation

            self.update_recommendation_ui(report, recommendation)
            self.set_status(f"Analysis complete: {frame_folder.name}. Auto applying display color stack.")

            self.apply_profile(recommendation)

        except Exception as error:
            messagebox.showerror("Analysis Error", str(error))

    def build_display_recommendation(self, report: dict[str, Any]) -> dict[str, Any]:
        pressures = report.get("average_pressures", {})

        shadow = float(pressures.get("shadow_pressure", 0.0))
        deep_shadow = float(pressures.get("deep_shadow_pressure", 0.0))
        highlight = float(pressures.get("highlight_pressure", 0.0))
        low_contrast = float(pressures.get("low_contrast_pressure", 0.0))
        backlight = float(pressures.get("backlight_pressure", 0.0))
        night = float(pressures.get("night_pressure", 0.0))

        brightness = clamp(0.50 + shadow * 0.10 + night * 0.08 + backlight * 0.04 - highlight * 0.06, 0.00, 1.00)
        contrast = clamp(0.50 + low_contrast * 0.22 + shadow * 0.08 + deep_shadow * 0.06 - highlight * 0.04, 0.00, 1.00)
        gamma = clamp(1.00 + shadow * 0.45 + deep_shadow * 0.25 + night * 0.30 - highlight * 0.12, 0.40, 2.80)
        vibrance = clamp(50.0 + shadow * 8.0 + low_contrast * 8.0 - highlight * 4.0, 0.0, 100.0)

        reasoning = []

        if shadow > 0.55:
            reasoning.append("Shadow pressure is elevated.")
        if low_contrast > 0.45:
            reasoning.append("Low contrast pressure is elevated.")
        if night > 0.45:
            reasoning.append("Sample trends dark or night like.")
        if highlight > 0.35:
            reasoning.append("Highlight pressure is elevated.")
        if not reasoning:
            reasoning.append("Scene is close to baseline.")

        return {
            "mode": "display_color_stack",
            "brightness": round(brightness, 3),
            "contrast": round(contrast, 3),
            "gamma": round(gamma, 3),
            "vibrance": round(vibrance, 1),
            "vibrance_apply_status": "logged_only",
            "reasoning": reasoning,
        }

    def update_recommendation_ui(self, report: dict[str, Any], recommendation: dict[str, Any]) -> None:
        pressures = report["average_pressures"]

        self.suppress_slider_apply = True
        self.brightness_var.set(float(recommendation["brightness"]))
        self.contrast_var.set(float(recommendation["contrast"]))
        self.gamma_var.set(float(recommendation["gamma"]))
        self.vibrance_var.set(float(recommendation["vibrance"]))
        self.suppress_slider_apply = False

        self.recommendation_text.set(
            f"Brightness: {recommendation['brightness']}\n"
            f"Contrast: {recommendation['contrast']}\n"
            f"Gamma: {recommendation['gamma']}\n"
            f"Vibrance: {recommendation['vibrance']} logged only"
        )

        for name, label_var in self.pressure_labels.items():
            value = pressures.get(name, "--")
            label_var.set(f"{name}: {value}")

    def on_slider_changed(self, value_label: ttk.Label, variable: tk.DoubleVar) -> None:
        value_label.config(text=f"{variable.get():.3f}")

        if self.suppress_slider_apply:
            return

        self.schedule_live_preview_apply()

    def schedule_live_preview_apply(self) -> None:
        if self.slider_debounce_job is not None:
            self.root.after_cancel(self.slider_debounce_job)

        self.slider_debounce_job = self.root.after(450, self.apply_live_preview)

    def apply_live_preview(self) -> None:
        self.slider_debounce_job = None

        if self.is_live_apply_running:
            self.pending_live_apply = True
            return

        profile = self.get_slider_profile()

        thread = threading.Thread(
            target=self.live_apply_worker,
            args=(profile,),
            daemon=True,
        )
        thread.start()

    def live_apply_worker(self, profile: dict[str, Any]) -> None:
        self.is_live_apply_running = True

        try:
            self.apply_profile(profile)
        except Exception as error:
            self.root.after(0, lambda: messagebox.showerror("Live Apply Error", str(error)))
        finally:
            self.is_live_apply_running = False

            if self.pending_live_apply:
                self.pending_live_apply = False
                self.root.after(50, self.apply_live_preview)

    def get_slider_profile(self) -> dict[str, Any]:
        return {
            "mode": "display_color_stack",
            "brightness": round(float(self.brightness_var.get()), 3),
            "contrast": round(float(self.contrast_var.get()), 3),
            "gamma": round(float(self.gamma_var.get()), 3),
            "vibrance": round(float(self.vibrance_var.get()), 1),
            "vibrance_apply_status": "logged_only",
            "reasoning": ["Manual display color stack sliders"],
        }

    def apply_profile(self, profile: dict[str, Any]) -> None:
        write_display_color_profile(profile, self.display_index)

        self.set_status(
            "Applying display stack: "
            f"B {profile['brightness']} | "
            f"C {profile['contrast']} | "
            f"G {profile['gamma']} | "
            f"V {profile['vibrance']} logged"
        )

        return_code = apply_display_color_stack(profile, self.display_index)
        self.set_status(f"Display stack applied. Exit code: {return_code}")

    def reset(self) -> None:
        self.set_status("Resetting display.")
        return_code = reset_display(self.display_index)
        self.set_status(f"Reset complete. Exit code: {return_code}")

    def save_feedback(self) -> None:
        FEEDBACK_ROOT.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = FEEDBACK_ROOT / f"feedback_{timestamp}.json"

        engine_recommendation = self.engine_recommendation
        user_profile = self.get_slider_profile()

        recommendation_delta = None
        if engine_recommendation is not None:
            recommendation_delta = {
                "brightness": round(user_profile["brightness"] - engine_recommendation["brightness"], 3),
                "contrast": round(user_profile["contrast"] - engine_recommendation["contrast"], 3),
                "gamma": round(user_profile["gamma"] - engine_recommendation["gamma"], 3),
                "vibrance": round(user_profile["vibrance"] - engine_recommendation["vibrance"], 1),
            }

        feedback = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "frame_folder": str(self.current_frame_folder) if self.current_frame_folder else None,
            "active_control_stack": "display_color_stack",
            "scene_context": {
                "map": self.map_var.get(),
                "time_of_day": self.time_of_day_var.get(),
                "weather": self.weather_var.get(),
                "night_vision": self.night_vision_var.get(),
                "thermal": self.thermal_var.get(),
            },
            "selected_profile_source": "user_preferred_display_color_stack",
            "user_marked_as_preferred": True,
            "engine_recommendation": engine_recommendation,
            "user_preferred_profile": user_profile,
            "recommendation_delta": recommendation_delta,
            "average_pressures": self.current_report.get("average_pressures") if self.current_report else None,
            "rating": self.rating_var.get().strip(),
            "washed_out": self.washed_out_var.get(),
            "too_dark": self.too_dark_var.get(),
            "too_bright": self.too_bright_var.get(),
            "notes": self.notes_box.get("1.0", "end").strip(),
        }

        self.save_json(path, feedback)
        self.set_status(f"Saved preferred display profile: {path}")

    def open_debug_folder(self) -> None:
        folder = self.current_debug_folder or DEBUG_ROOT
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(folder)

    def save_json(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    def set_status(self, message: str) -> None:
        self.root.after(0, lambda: self.status_var.set(message))


def main() -> int:
    root = tk.Tk()
    app = RaidVisionGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
