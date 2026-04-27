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
from control_runner import apply_custom_lut, reset_display
from profile_writer import write_custom_lut_profile
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


class RaidVisionGUI:
    """
    RaidVision local control panel.

    Responsibilities:
    - Start and stop frame sampling
    - Auto analyze after stopping
    - Show recommended custom LUT values
    - Let user tweak sliders
    - Apply recommended or manual values
    - Reset display
    - Save feedback logs
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("RaidVision Control Panel")
        self.root.geometry("980x720")

        self.display_index = DEFAULT_DISPLAY_INDEX

        self.capture_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.is_sampling = False

        self.current_frame_folder: Path | None = None
        self.current_report: dict[str, Any] | None = None
        self.current_recommendation: dict[str, Any] | None = None
        self.current_debug_folder: Path | None = None

        self.shadow_lift_var = tk.DoubleVar(value=0.35)
        self.midtone_var = tk.DoubleVar(value=0.20)
        self.highlight_protect_var = tk.DoubleVar(value=0.40)

        self.rating_var = tk.StringVar(value="")
        self.washed_out_var = tk.BooleanVar(value=False)
        self.too_dark_var = tk.BooleanVar(value=False)
        self.too_bright_var = tk.BooleanVar(value=False)

        self.map_var = tk.StringVar(value="Unknown")
        self.time_of_day_var = tk.StringVar(value="Unknown")
        self.weather_var = tk.StringVar(value="Unknown")
        self.night_vision_var = tk.StringVar(value="Off")
        self.thermal_var = tk.StringVar(value="Off")

        self.live_preview_var = tk.BooleanVar(value=False)
        self.slider_debounce_job = None
        self.is_live_apply_running = False
        self.pending_live_apply = False

        self.status_var = tk.StringVar(value="Ready")
        self.recommendation_text = tk.StringVar(value="No recommendation yet.")

        self.pressure_labels: dict[str, tk.StringVar] = {}

        self.build_ui()

    # ============================================================
    # UI Layout
    # ============================================================

    def build_ui(self) -> None:
        """
        Build the main GUI layout.
        """
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
        self.build_pressure_panel(right)
        self.build_feedback_panel(middle)

        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor="w",
            padding=6,
        )
        status_bar.grid(row=1, column=0, columnspan=3, sticky="ew")

    def build_buttons(self, parent: ttk.Frame) -> None:
        """
        Build left side workflow buttons.
        """
        ttk.Label(parent, text="Workflow", font=("Segoe UI", 14, "bold")).pack(anchor="w")

        ttk.Button(parent, text="Start Sample", command=self.start_sample).pack(fill="x", pady=5)
        ttk.Button(parent, text="Stop Sample + Auto Analyze", command=self.stop_sample).pack(fill="x", pady=5)

        ttk.Separator(parent).pack(fill="x", pady=10)

        ttk.Button(parent, text="Analyze Latest", command=self.analyze_latest).pack(fill="x", pady=5)
        ttk.Button(parent, text="Apply Recommended", command=self.apply_recommended).pack(fill="x", pady=5)
        ttk.Button(parent, text="Apply Current Sliders", command=self.apply_manual_sliders).pack(fill="x", pady=5)
        ttk.Button(parent, text="Reset Display", command=self.reset).pack(fill="x", pady=5)

        ttk.Separator(parent).pack(fill="x", pady=10)

        ttk.Button(parent, text="Open Debug Folder", command=self.open_debug_folder).pack(fill="x", pady=5)
        ttk.Button(parent, text="Save As Preferred", command=self.save_feedback).pack(fill="x", pady=5)

        ttk.Separator(parent).pack(fill="x", pady=10)

        ttk.Label(parent, text="Hotkey plan", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(parent, text="F8  Start Sample").pack(anchor="w")
        ttk.Label(parent, text="F9  Stop + Analyze").pack(anchor="w")
        ttk.Label(parent, text="F10 Apply Recommended").pack(anchor="w")
        ttk.Label(parent, text="F11 Reset").pack(anchor="w")
        ttk.Label(parent, text="F12 Save Feedback").pack(anchor="w")

    def build_sliders(self, parent: ttk.Frame) -> None:
        """
        Build middle panel recommendation and manual LUT sliders.
        """
        ttk.Label(parent, text="Recommended LUT", font=("Segoe UI", 14, "bold")).pack(anchor="w")

        recommendation_label = ttk.Label(
            parent,
            textvariable=self.recommendation_text,
            justify="left",
            padding=8,
        )
        recommendation_label.pack(fill="x", pady=6)

        ttk.Separator(parent).pack(fill="x", pady=10)

        ttk.Label(parent, text="Manual Custom LUT Sliders", font=("Segoe UI", 14, "bold")).pack(anchor="w")

        ttk.Checkbutton(
            parent,
            text="Live Preview Sliders",
            variable=self.live_preview_var,
        ).pack(anchor="w", pady=(4, 10))

        self.add_slider(parent, "Shadow Lift", self.shadow_lift_var, 0.0, 1.0)
        self.add_slider(parent, "Midtone", self.midtone_var, 0.0, 0.5)
        self.add_slider(parent, "Highlight Protect", self.highlight_protect_var, 0.0, 1.0)

    def add_slider(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.DoubleVar,
        minimum: float,
        maximum: float,
    ) -> None:
        """
        Add a labeled slider with numeric value.
        """
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

    def build_pressure_panel(self, parent: ttk.Frame) -> None:
        """
        Build right side pressure readout.
        """
        ttk.Label(parent, text="Pressure Readout", font=("Segoe UI", 14, "bold")).pack(anchor="w")

        pressure_names = [
            "shadow_pressure",
            "deep_shadow_pressure",
            "midtone_pressure",
            "highlight_pressure",
            "low_contrast_pressure",
            "backlight_pressure",
            "night_pressure",
        ]

        for name in pressure_names:
            value_var = tk.StringVar(value=f"{name}: --")
            self.pressure_labels[name] = value_var
            ttk.Label(parent, textvariable=value_var).pack(anchor="w", pady=3)

    def build_feedback_panel(self, parent: ttk.Frame) -> None:
        """
        Build feedback inputs used for tuning logs.
        """
        ttk.Separator(parent).pack(fill="x", pady=16)

        ttk.Label(parent, text="Scene Context", font=("Segoe UI", 14, "bold")).pack(anchor="w")

        context_grid = ttk.Frame(parent)
        context_grid.pack(fill="x", pady=6)

        ttk.Label(context_grid, text="Map:").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Combobox(
            context_grid,
            textvariable=self.map_var,
            values=[
                "Unknown",
                "Customs",
                "Woods",
                "Shoreline",
                "Interchange",
                "Reserve",
                "Lighthouse",
                "Streets",
                "Ground Zero",
                "Factory",
                "Labs",
                "Other",
            ],
            width=18,
        ).grid(row=0, column=1, sticky="w", pady=3)

        ttk.Label(context_grid, text="Time:").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Combobox(
            context_grid,
            textvariable=self.time_of_day_var,
            values=["Unknown", "Day", "Dusk", "Night"],
            width=18,
        ).grid(row=1, column=1, sticky="w", pady=3)

        ttk.Label(context_grid, text="Weather:").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Combobox(
            context_grid,
            textvariable=self.weather_var,
            values=["Unknown", "Clear", "Overcast", "Rain", "Fog", "Snow"],
            width=18,
        ).grid(row=2, column=1, sticky="w", pady=3)

        ttk.Label(context_grid, text="Night Vision:").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Combobox(
            context_grid,
            textvariable=self.night_vision_var,
            values=["Off", "On"],
            width=18,
        ).grid(row=3, column=1, sticky="w", pady=3)

        ttk.Label(context_grid, text="Thermal:").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Combobox(
            context_grid,
            textvariable=self.thermal_var,
            values=["Off", "On"],
            width=18,
        ).grid(row=4, column=1, sticky="w", pady=3)

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

    # ============================================================
    # Capture Workflow
    # ============================================================

    def start_sample(self) -> None:
        """
        Start a background sample capture.
        """
        if self.is_sampling:
            self.set_status("Already sampling.")
            return

        self.stop_event.clear()
        self.is_sampling = True
        self.current_frame_folder = create_raid_folder()

        self.capture_thread = threading.Thread(
            target=self.capture_loop,
            daemon=True,
        )
        self.capture_thread.start()

        self.set_status("Sampling started. Stop manually or wait for auto finish.")

    def stop_sample(self) -> None:
        """
        Stop capture and trigger analysis.
        """
        if not self.is_sampling:
            self.set_status("Not currently sampling.")
            return

        self.stop_event.set()
        self.set_status("Stopping sample. Analysis will run automatically.")

    def capture_loop(self) -> None:
        """
        Capture frames until stopped or sample duration ends.
        """
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

                    saved_path = save_frame(
                        frame=frame,
                        folder=self.current_frame_folder,
                        index=frame_index,
                        status=status,
                    )

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

    # ============================================================
    # Analysis
    # ============================================================

    def analyze_current_folder(self) -> None:
        """
        Analyze the current frame folder.
        """
        if self.current_frame_folder is None:
            self.set_status("No current frame folder to analyze.")
            return

        self.analyze_folder(self.current_frame_folder)

    def analyze_latest(self) -> None:
        """
        Analyze the latest saved frame folder.
        """
        latest = find_latest_frame_folder()

        if latest is None:
            messagebox.showwarning("No Frames", "No saved frame folder found.")
            return

        self.current_frame_folder = latest
        self.analyze_folder(latest)

    def analyze_folder(self, frame_folder: Path) -> None:
        """
        Run the visibility engine against a folder of frames.
        """
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
            recommendation = report["lut_recommendation"]

            profile = write_custom_lut_profile(
                recommendation=recommendation,
                display_index=self.display_index,
            )

            report["frame_folder"] = str(frame_folder)
            report["frame_files_used"] = [str(path) for path in selected_paths]
            report["control_profile"] = profile

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
            self.current_recommendation = recommendation

            self.update_recommendation_ui(report)
            self.set_status(f"Analysis complete: {frame_folder.name}")

        except Exception as error:
            messagebox.showerror("Analysis Error", str(error))

    def update_recommendation_ui(self, report: dict[str, Any]) -> None:
        """
        Update GUI labels and sliders from the latest recommendation.
        """
        recommendation = report["lut_recommendation"]
        pressures = report["average_pressures"]

        self.shadow_lift_var.set(float(recommendation["shadow_lift"]))
        self.midtone_var.set(float(recommendation["midtone"]))
        self.highlight_protect_var.set(float(recommendation["highlight_protect"]))

        self.recommendation_text.set(
            f"Shadow Lift: {recommendation['shadow_lift']}\n"
            f"Midtone: {recommendation['midtone']}\n"
            f"Highlight Protect: {recommendation['highlight_protect']}"
        )

        for name, label_var in self.pressure_labels.items():
            value = pressures.get(name, "--")
            label_var.set(f"{name}: {value}")

    # ============================================================
    # Live Slider Preview
    # ============================================================

    def on_slider_changed(
        self,
        value_label: ttk.Label,
        variable: tk.DoubleVar,
    ) -> None:
        """
        Update slider value text and optionally schedule a live preview apply.
        """
        value_label.config(text=f"{variable.get():.3f}")

        if self.live_preview_var.get():
            self.schedule_live_preview_apply()

    def schedule_live_preview_apply(self) -> None:
        """
        Debounce live slider updates so we only apply after movement pauses briefly.
        """
        if self.slider_debounce_job is not None:
            self.root.after_cancel(self.slider_debounce_job)

        self.slider_debounce_job = self.root.after(350, self.apply_live_preview)

    def apply_live_preview(self) -> None:
        """
        Apply the current slider values automatically when live preview is enabled.
        """
        self.slider_debounce_job = None

        if not self.live_preview_var.get():
            return

        if self.is_live_apply_running:
            self.pending_live_apply = True
            return

        recommendation = self.get_slider_recommendation()

        thread = threading.Thread(
            target=self.live_apply_worker,
            args=(recommendation,),
            daemon=True,
        )
        thread.start()

    def live_apply_worker(self, recommendation: dict[str, Any]) -> None:
        """
        Run live preview apply outside the Tkinter UI thread.
        """
        self.is_live_apply_running = True

        try:
            write_custom_lut_profile(
                recommendation=recommendation,
                display_index=self.display_index,
            )

            self.set_status(
                "Live applying sliders: "
                f"{recommendation['shadow_lift']} / "
                f"{recommendation['midtone']} / "
                f"{recommendation['highlight_protect']}"
            )

            return_code = apply_custom_lut(recommendation, self.display_index)

            self.set_status(f"Live preview applied. Exit code: {return_code}")

        except Exception as error:
            self.root.after(0, lambda: messagebox.showerror("Live Preview Error", str(error)))

        finally:
            self.is_live_apply_running = False

            if self.pending_live_apply and self.live_preview_var.get():
                self.pending_live_apply = False
                self.root.after(50, self.apply_live_preview)

    # ============================================================
    # Apply and Reset
    # ============================================================

    def get_slider_recommendation(self) -> dict[str, Any]:
        """
        Build a recommendation object from current slider values.
        """
        return {
            "mode": "custom_lut",
            "shadow_lift": round(float(self.shadow_lift_var.get()), 3),
            "midtone": round(float(self.midtone_var.get()), 3),
            "highlight_protect": round(float(self.highlight_protect_var.get()), 3),
            "reasoning": ["Manual slider profile"],
        }

    def apply_recommended(self) -> None:
        """
        Apply the latest recommended custom LUT.
        """
        if self.current_recommendation is None:
            messagebox.showwarning("No Recommendation", "Run analysis first.")
            return

        self.set_status("Applying recommended LUT.")
        return_code = apply_custom_lut(self.current_recommendation, self.display_index)
        self.set_status(f"Apply recommended complete. Exit code: {return_code}")

    def apply_manual_sliders(self) -> None:
        """
        Apply the current manual slider values.
        """
        recommendation = self.get_slider_recommendation()

        write_custom_lut_profile(
            recommendation=recommendation,
            display_index=self.display_index,
        )

        self.set_status("Applying manual slider LUT.")
        return_code = apply_custom_lut(recommendation, self.display_index)
        self.set_status(f"Apply manual complete. Exit code: {return_code}")

    def reset(self) -> None:
        """
        Reset display to neutral.
        """
        self.set_status("Resetting display.")
        return_code = reset_display(self.display_index)
        self.set_status(f"Reset complete. Exit code: {return_code}")

    # ============================================================
    # Feedback
    # ============================================================

    def save_feedback(self) -> None:
        """
        Save the current slider values as the user's preferred profile.

        This separates what RaidVision recommended from what the user actually preferred.
        """
        FEEDBACK_ROOT.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = FEEDBACK_ROOT / f"feedback_{timestamp}.json"

        engine_recommendation = self.current_recommendation
        user_recommendation = self.get_slider_recommendation()

        recommendation_delta = None

        if engine_recommendation is not None:
            recommendation_delta = {
                "shadow_lift": round(
                    user_recommendation["shadow_lift"] - engine_recommendation["shadow_lift"],
                    3,
                ),
                "midtone": round(
                    user_recommendation["midtone"] - engine_recommendation["midtone"],
                    3,
                ),
                "highlight_protect": round(
                    user_recommendation["highlight_protect"] - engine_recommendation["highlight_protect"],
                    3,
                ),
            }

        feedback = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "frame_folder": str(self.current_frame_folder) if self.current_frame_folder else None,

            "scene_context": {
                "map": self.map_var.get(),
                "time_of_day": self.time_of_day_var.get(),
                "weather": self.weather_var.get(),
                "night_vision": self.night_vision_var.get(),
                "thermal": self.thermal_var.get(),
            },

            "selected_profile_source": "user_recommendation",
            "user_marked_as_preferred": True,

            "engine_recommendation": engine_recommendation,
            "user_recommendation": {
                **user_recommendation,
                "source": "manual_slider_tuning",
            },
            "recommendation_delta": recommendation_delta,

            "average_pressures": (
                self.current_report.get("average_pressures")
                if self.current_report
                else None
            ),

            "rating": self.rating_var.get().strip(),
            "washed_out": self.washed_out_var.get(),
            "too_dark": self.too_dark_var.get(),
            "too_bright": self.too_bright_var.get(),
            "notes": self.notes_box.get("1.0", "end").strip(),
        }

        self.save_json(path, feedback)
        self.set_status(f"Saved preferred profile: {path}")


    # ============================================================
    # Utilities
    # ============================================================

    def open_debug_folder(self) -> None:
        """
        Open the current debug folder in Windows Explorer.
        """
        folder = self.current_debug_folder or DEBUG_ROOT

        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(folder)

    def save_json(self, path: Path, data: dict[str, Any]) -> None:
        """
        Save JSON safely from the GUI.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    def set_status(self, message: str) -> None:
        """
        Update the status bar from any thread.
        """
        self.root.after(0, lambda: self.status_var.set(message))


def main() -> int:
    root = tk.Tk()
    app = RaidVisionGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())