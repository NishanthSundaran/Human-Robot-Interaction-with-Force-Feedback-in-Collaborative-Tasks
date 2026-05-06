#!/usr/bin/env python3
"""
hybrid_tuning_gui.py

Tkinter GUI for live-tuning hybrid admittance controller parameters.
Sends ros2 param set commands to /hybrid_control_node in real time.

Usage:
    ros2 run my_thesis_controller hybrid_tuning_gui
    # or just: python3 hybrid_tuning_gui.py
"""

import subprocess
import threading
import tkinter as tk
from tkinter import ttk

NODE = "/hybrid_control_node"

# (param_name, label, min, max, default, resolution, category)
PARAMS = [
    # ── Filtering (most impactful for vibration) ──
    ("alpha_wrench",            "Wrench LP Filter (alpha)",  0.01, 0.50, 0.15, 0.01, "Filtering"),
    ("admittance_output_alpha", "Output LP Filter (alpha)",  0.01, 0.50, 0.25, 0.01, "Filtering"),
    ("force_deadband",          "Force Deadband (N)",        0.5,  10.0, 3.0,  0.1,  "Filtering"),
    ("torque_deadband",         "Torque Deadband (Nm)",      0.05, 2.0,  0.2,  0.05, "Filtering"),

    # ── Admittance Dynamics ──
    ("mass_linear",             "Virtual Mass Linear (kg)",    0.5, 10.0, 3.0,  0.1,  "Dynamics"),
    ("mass_angular",            "Virtual Mass Angular (kg.m2)", 0.1, 3.0, 0.5,  0.1,  "Dynamics"),
    ("damping_linear",          "Damping Linear (Ns/m)",      10.0, 100.0, 40.0, 1.0, "Dynamics"),
    ("damping_angular",         "Damping Angular (Nms/rad)",   1.0, 30.0, 8.0,  0.5,  "Dynamics"),
    ("guiding_damping_linear",  "Guiding Damping Lin (Ns/m)", 5.0, 80.0, 30.0, 1.0,  "Dynamics"),
    ("guiding_damping_angular", "Guiding Damping Ang",         1.0, 20.0, 5.0,  0.5,  "Dynamics"),

    # ── Adaptive Damping ──
    ("adaptive_damping_k_lin",  "Adaptive Damp k_lin",        0.0, 10.0, 2.0,  0.1,  "Adaptive"),
    ("adaptive_damping_k_ang",  "Adaptive Damp k_ang",        0.0, 5.0,  1.0,  0.1,  "Adaptive"),

    # ── Stop / Contact Detection ──
    ("stop_force_epsilon",      "Stop Force Eps (N)",         0.1, 5.0,  0.5,  0.1,  "Contact"),
    ("stop_torque_epsilon",     "Stop Torque Eps (Nm)",       0.01, 1.0, 0.05, 0.01, "Contact"),
    ("no_contact_decay_tau",    "No-Contact Decay Tau (s)",   0.05, 1.0, 0.15, 0.01, "Contact"),

    # ── Safety Limits ──
    ("max_linear_speed",        "Max Linear Speed (m/s)",     0.05, 0.5, 0.25, 0.01, "Safety"),
    ("max_angular_speed",       "Max Angular Speed (rad/s)",  0.1,  3.0, 1.0,  0.1,  "Safety"),
    ("max_linear_accel",        "Max Linear Accel (m/s2)",    0.1,  3.0, 1.0,  0.1,  "Safety"),
    ("max_angular_accel",       "Max Angular Accel (rad/s2)", 0.5,  10.0, 3.0, 0.5,  "Safety"),

    # ── Intent Amplification ──
    ("intent_saturation_force", "Intent Sat Force (N)",       5.0, 30.0, 15.0, 0.5,  "Intent"),
    ("intent_ref_force",        "Intent Ref Force (N)",       1.0, 15.0, 5.0,  0.5,  "Intent"),
    ("intent_saturation_torque","Intent Sat Torque (Nm)",     0.5, 5.0,  2.0,  0.1,  "Intent"),
    ("intent_ref_torque",       "Intent Ref Torque (Nm)",     0.1, 2.0,  0.5,  0.1,  "Intent"),

    # ── Position Control Gains ──
    ("kp_pos",                  "Kp Position [x,y,z]",        0.0, 10.0, 2.0,  0.1,  "PosCtrl"),
    ("kp_rot",                  "Kp Rotation [x,y,z]",        0.0, 5.0,  1.0,  0.1,  "PosCtrl"),
]

# Parameters that are arrays (need special handling)
ARRAY_PARAMS = {"kp_pos", "kp_rot"}


def set_param(name, value):
    """Set a ROS 2 parameter on the hybrid_control_node."""
    if name in ARRAY_PARAMS:
        # Set as array of 3 identical values
        val_str = f"[{value}, {value}, {value}]"
    else:
        val_str = str(value)

    cmd = ["ros2", "param", "set", NODE, name, val_str]
    try:
        subprocess.run(cmd, capture_output=True, timeout=3)
    except Exception:
        pass


class HybridTuningGUI:
    def __init__(self, root):
        self.root = root
        root.title("Hybrid Controller Tuning")
        root.resizable(True, True)

        # Style
        style = ttk.Style()
        style.configure("Header.TLabel", font=("Helvetica", 10, "bold"))
        style.configure("Val.TLabel", font=("Courier", 9), width=8)

        # Organize by category
        categories = {}
        for p in PARAMS:
            cat = p[6]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(p)

        self.sliders = {}
        self.value_labels = {}

        # Create notebook with tabs
        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=5, pady=5)

        for cat_name, params in categories.items():
            frame = ttk.Frame(notebook, padding=10)
            notebook.add(frame, text=cat_name)

            for i, (name, label, lo, hi, default, res, _) in enumerate(params):
                ttk.Label(frame, text=label, style="Header.TLabel").grid(
                    row=i, column=0, sticky="w", padx=(0, 10), pady=2)

                var = tk.DoubleVar(value=default)
                slider = ttk.Scale(
                    frame, from_=lo, to=hi, variable=var,
                    orient="horizontal", length=300,
                    command=lambda val, n=name, v=var, r=res: self._on_slide(n, v, r))
                slider.grid(row=i, column=1, sticky="ew", pady=2)

                val_label = ttk.Label(frame, text=f"{default:.3f}", style="Val.TLabel")
                val_label.grid(row=i, column=2, padx=(5, 0), pady=2)

                # Reset button
                reset_btn = ttk.Button(
                    frame, text="Reset", width=5,
                    command=lambda n=name, v=var, d=default, r=res: self._reset(n, v, d, r))
                reset_btn.grid(row=i, column=3, padx=(5, 0), pady=2)

                frame.columnconfigure(1, weight=1)
                self.sliders[name] = (var, res)
                self.value_labels[name] = val_label

        # Bottom buttons
        btn_frame = ttk.Frame(root, padding=5)
        btn_frame.pack(fill="x")

        ttk.Button(btn_frame, text="Reset ALL to Defaults",
                    command=self._reset_all).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Print Current Values",
                    command=self._print_values).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Apply Anti-Vibration Preset",
                    command=self._preset_anti_vibration).pack(side="left", padx=5)

        # Status bar
        self.status = ttk.Label(root, text="Ready", relief="sunken", anchor="w")
        self.status.pack(fill="x", side="bottom")

    def _on_slide(self, name, var, resolution):
        """Called on slider move — snap to resolution and send param."""
        raw = var.get()
        snapped = round(raw / resolution) * resolution
        var.set(snapped)
        self.value_labels[name].config(text=f"{snapped:.3f}")
        # Send in background thread to avoid blocking GUI
        threading.Thread(target=set_param, args=(name, snapped), daemon=True).start()
        self.status.config(text=f"Set {name} = {snapped}")

    def _reset(self, name, var, default, resolution):
        var.set(default)
        self.value_labels[name].config(text=f"{default:.3f}")
        threading.Thread(target=set_param, args=(name, default), daemon=True).start()
        self.status.config(text=f"Reset {name} = {default}")

    def _reset_all(self):
        for name, label, lo, hi, default, res, _ in PARAMS:
            var, _ = self.sliders[name]
            var.set(default)
            self.value_labels[name].config(text=f"{default:.3f}")
            threading.Thread(target=set_param, args=(name, default), daemon=True).start()
        self.status.config(text="All parameters reset to defaults")

    def _preset_anti_vibration(self):
        """Apply a preset that reduces vibration during idle waiting."""
        preset = {
            "alpha_wrench": 0.08,
            "force_deadband": 4.5,
            "torque_deadband": 0.4,
            "damping_linear": 55.0,
            "damping_angular": 12.0,
            "stop_force_epsilon": 1.2,
            "stop_torque_epsilon": 0.1,
            "no_contact_decay_tau": 0.3,
            "admittance_output_alpha": 0.15,
        }
        for name, value in preset.items():
            if name in self.sliders:
                var, res = self.sliders[name]
                snapped = round(value / res) * res
                var.set(snapped)
                self.value_labels[name].config(text=f"{snapped:.3f}")
                threading.Thread(target=set_param, args=(name, snapped), daemon=True).start()
        self.status.config(text="Anti-vibration preset applied")

    def _print_values(self):
        print("\n# Current hybrid controller parameters:")
        for name, label, lo, hi, default, res, cat in PARAMS:
            var, _ = self.sliders[name]
            val = var.get()
            marker = " *" if abs(val - default) > res * 0.5 else ""
            print(f"  {name}: {val:.3f}{marker}")
        print()
        self.status.config(text="Values printed to terminal")


def main():
    root = tk.Tk()
    HybridTuningGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
