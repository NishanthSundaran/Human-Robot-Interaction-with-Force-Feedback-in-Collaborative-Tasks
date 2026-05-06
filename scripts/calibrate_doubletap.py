#!/usr/bin/env python3
"""
Double-tap calibration tool.

Subscribes to /wrench_zeroed, asks you to perform several double-taps,
records force spikes, and outputs tuned parameters for interaction_classifier
and assistive_lift_v4_node.

Usage (while hardware launch is running):
  python3 scripts/calibrate_doubletap.py
"""

import math
import time
import sys
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import WrenchStamped
from collections import deque

import numpy as np


class DoubleTapCalibrator(Node):
    def __init__(self):
        super().__init__('doubletap_calibrator')

        self.samples = []           # (time_s, force_mag)
        self.recording = False
        self.start_time = None

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            WrenchStamped, '/wrench_zeroed', self._wrench_cb, qos)

        self.get_logger().info('Subscribed to /wrench_zeroed')

    def _wrench_cb(self, msg):
        if not self.recording:
            return
        fx = msg.wrench.force.x
        fy = msg.wrench.force.y
        fz = msg.wrench.force.z
        mag = math.sqrt(fx*fx + fy*fy + fz*fz)
        t = time.monotonic() - self.start_time
        self.samples.append((t, mag))


def detect_spikes(times, forces, threshold):
    """Find spike peaks above threshold with simple state machine."""
    spikes = []  # list of (peak_time, peak_force)
    in_spike = False
    peak_t = 0.0
    peak_f = 0.0

    for t, f in zip(times, forces):
        if not in_spike:
            if f > threshold:
                in_spike = True
                peak_t = t
                peak_f = f
        else:
            if f > peak_f:
                peak_t = t
                peak_f = f
            if f < threshold * 0.5:
                spikes.append((peak_t, peak_f))
                in_spike = False
                peak_f = 0.0

    if in_spike:
        spikes.append((peak_t, peak_f))

    return spikes


def analyze_round(samples):
    """Analyze one recording round, return spike info."""
    if len(samples) < 10:
        return None

    times = np.array([s[0] for s in samples])
    forces = np.array([s[1] for s in samples])

    # Noise floor: use first 0.5s as baseline
    baseline_mask = times < 0.5
    if baseline_mask.sum() < 5:
        baseline_mask = np.ones(len(times), dtype=bool)
        baseline_mask[len(times)//2:] = False

    noise_mean = float(np.mean(forces[baseline_mask]))
    noise_std = float(np.std(forces[baseline_mask]))
    noise_max = float(np.max(forces[baseline_mask]))

    # Detect spikes above noise_mean + 2*noise_std (or 1.5 N min)
    spike_thresh = max(noise_mean + 2.0 * noise_std, 1.5)
    spikes = detect_spikes(times, forces, spike_thresh)

    return {
        'noise_mean': noise_mean,
        'noise_std': noise_std,
        'noise_max': noise_max,
        'spike_thresh_used': spike_thresh,
        'spikes': spikes,
        'times': times,
        'forces': forces,
    }


def main():
    rclpy.init()
    node = DoubleTapCalibrator()

    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()

    time.sleep(1.0)

    print("\n" + "="*60)
    print("  DOUBLE-TAP CALIBRATION")
    print("="*60)
    print()
    print("Make sure the hardware launch is running and the robot")
    print("is at home position (not being touched).")
    print()
    print("You'll do 5 rounds. Each round:")
    print("  1. Don't touch the robot (2s baseline)")
    print("  2. Do ONE double-tap (two quick taps)")
    print("  3. Stop touching (1s cooldown)")
    print()

    all_tap_forces = []
    all_tap_gaps = []
    all_noise_means = []
    all_noise_stds = []
    all_noise_maxs = []

    num_rounds = 5

    for i in range(num_rounds):
        input(f"\n--- Round {i+1}/{num_rounds}: Press ENTER, wait 2s, then double-tap ---")

        node.samples = []
        node.start_time = time.monotonic()
        node.recording = True

        # Record for 6 seconds total
        print("  Recording... (don't touch for 2s, then double-tap, then wait)")
        time.sleep(6.0)
        node.recording = False

        result = analyze_round(node.samples)
        if result is None:
            print("  ERROR: Not enough data. Is /wrench_zeroed publishing?")
            continue

        spikes = result['spikes']
        print(f"  Noise: mean={result['noise_mean']:.2f}N, "
              f"std={result['noise_std']:.2f}N, max={result['noise_max']:.2f}N")
        print(f"  Detected {len(spikes)} spike(s):")

        for j, (st, sf) in enumerate(spikes):
            print(f"    Spike {j+1}: t={st:.3f}s, force={sf:.2f}N")

        all_noise_means.append(result['noise_mean'])
        all_noise_stds.append(result['noise_std'])
        all_noise_maxs.append(result['noise_max'])

        if len(spikes) >= 2:
            # Take first two spikes as the double tap
            tap1_t, tap1_f = spikes[0]
            tap2_t, tap2_f = spikes[1]
            gap = tap2_t - tap1_t
            all_tap_forces.extend([tap1_f, tap2_f])
            all_tap_gaps.append(gap)
            print(f"  Double-tap gap: {gap:.3f}s")
            print(f"  Tap forces: {tap1_f:.2f}N, {tap2_f:.2f}N")
        elif len(spikes) == 1:
            print("  Only 1 spike detected — tapped too softly or too fast")
            all_tap_forces.append(spikes[0][1])
        else:
            print("  No spikes detected — tap harder next time")

    # ======================== Analysis ========================
    print("\n" + "="*60)
    print("  ANALYSIS")
    print("="*60)

    if len(all_tap_forces) == 0:
        print("No taps detected at all. Check /wrench_zeroed topic.")
        node.destroy_node()
        rclpy.shutdown()
        return

    noise_mean = np.mean(all_noise_means)
    noise_std = np.mean(all_noise_stds)
    noise_max = np.max(all_noise_maxs)
    tap_forces = np.array(all_tap_forces)
    tap_min = float(np.min(tap_forces))
    tap_max = float(np.max(tap_forces))
    tap_mean = float(np.mean(tap_forces))

    print(f"\nNoise floor:")
    print(f"  Mean force:  {noise_mean:.2f} N")
    print(f"  Std dev:     {noise_std:.2f} N")
    print(f"  Max noise:   {noise_max:.2f} N")

    print(f"\nYour tap forces ({len(tap_forces)} taps):")
    print(f"  Min:  {tap_min:.2f} N")
    print(f"  Max:  {tap_max:.2f} N")
    print(f"  Mean: {tap_mean:.2f} N")

    if len(all_tap_gaps) > 0:
        gaps = np.array(all_tap_gaps)
        gap_min = float(np.min(gaps))
        gap_max = float(np.max(gaps))
        gap_mean = float(np.mean(gaps))
        print(f"\nYour double-tap timing ({len(gaps)} pairs):")
        print(f"  Fastest gap: {gap_min:.3f}s")
        print(f"  Slowest gap: {gap_max:.3f}s")
        print(f"  Average gap: {gap_mean:.3f}s")
    else:
        gap_max = 1.0
        gap_mean = 0.5
        print("\nNo double-tap pairs detected — using defaults for timing.")

    # ======================== Recommendations ========================
    print("\n" + "="*60)
    print("  RECOMMENDED PARAMETERS")
    print("="*60)

    # nudge_threshold: midpoint between noise max and tap min, with safety margin
    rec_nudge_thresh = max((noise_max + tap_min) / 2.0, noise_max + 0.5)
    rec_nudge_thresh = round(rec_nudge_thresh, 1)

    # z_score_thresh: how many std-devs the weakest tap is above noise
    weakest_z = (tap_min - noise_mean) / max(noise_std, 0.01)
    rec_z_thresh = max(round(weakest_z * 0.6, 1), 1.5)  # 60% of weakest tap's z-score

    # nudge_cooldown: half of the fastest gap (so both taps register)
    rec_cooldown = round(min(gap_min * 0.4 if len(all_tap_gaps) > 0 else 0.3, 0.5), 2)

    # double_tap_window: 1.5x slowest gap (generous)
    rec_window = round(gap_max * 1.5 if len(all_tap_gaps) > 0 else 2.5, 1)
    rec_window = max(rec_window, 1.5)  # floor at 1.5s

    print(f"\n  nudge_threshold:     {rec_nudge_thresh} N  "
          f"(noise_max={noise_max:.1f}, weakest_tap={tap_min:.1f})")
    print(f"  z_score_thresh:      {rec_z_thresh}  "
          f"(weakest tap z-score={weakest_z:.1f})")
    print(f"  nudge_cooldown_s:    {rec_cooldown} s  "
          f"(fastest gap={gap_min:.3f}s)" if len(all_tap_gaps) > 0 else
          f"  nudge_cooldown_s:    {rec_cooldown} s  (default)")
    print(f"  DOUBLE_TAP_WINDOW_S: {rec_window} s  "
          f"(slowest gap={gap_max:.3f}s)" if len(all_tap_gaps) > 0 else
          f"  DOUBLE_TAP_WINDOW_S: {rec_window} s  (default)")

    print(f"\n  Code changes needed:")
    print(f"    interaction_classifier (assistive_lift_v4_node.py line ~2034):")
    print(f'      "-p", "nudge_threshold:={rec_nudge_thresh}",')
    print(f'      "-p", "z_score_thresh:={rec_z_thresh}",')
    print(f'      "-p", "nudge_cooldown_s:={rec_cooldown}",')
    print(f"    assistive_lift_v4_node.py line ~107:")
    print(f"      DOUBLE_TAP_WINDOW_S = {rec_window}")

    # Also flag the GUIDING guard issue
    print(f"\n  IMPORTANT: Currently nudge is REJECTED during GUIDING state.")
    print(f"  If you tap while holding the robot, the tap won't register")
    print(f"  until 3s after you release (guide_exit_count=1500 @ 500Hz).")
    print(f"  Consider removing 'self.state != \"GUIDING\"' guard in")
    print(f"  human_interaction_classifier.py line 179.")

    print("\n" + "="*60)
    print("  Want me to apply these values? (check the main session)")
    print("="*60 + "\n")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
