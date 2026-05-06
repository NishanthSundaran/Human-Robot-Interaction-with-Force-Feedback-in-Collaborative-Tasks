#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import String, Bool, Float32

import numpy as np
from collections import deque


class HumanInteractionClassifier(Node):
    """
    Human interaction classifier for force/torque-based HRI.

    States:
      - IDLE: no meaningful contact
      - GUIDING: sustained contact (continuous force)
      - NUDGE: short impulse-like contact (spike)

    Outputs:
      - /interaction_state (std_msgs/String): "IDLE" | "GUIDING" | "NUDGE"
      - /interaction/is_guiding (std_msgs/Bool)
      - /interaction/is_nudge (std_msgs/Bool)  (one-shot pulse)
      - /interaction/force_mag (std_msgs/Float32) filtered magnitude
      - /interaction/force_avg (std_msgs/Float32) window average
      - /interaction/z_score (std_msgs/Float32) z-score of current sample
    """

    def __init__(self):
        super().__init__('interaction_classifier')

        # ---------------- Parameters ----------------
        self.declare_parameter('ft_topic', '/wrench_zeroed')
        self.declare_parameter('rate_hz', 100.0)

        # Windowing / features
        self.declare_parameter('window_size', 100)         # ~0.2 s at 500 Hz wrench rate
        self.declare_parameter('ema_alpha', 0.2)           # low-pass on magnitude (0..1) — was 0.5, more filtering to reject FT noise

        # Thresholds — raised to avoid false triggers from UR3e FT noise
        self.declare_parameter('guide_threshold', 5.0)     # N (avg force after EMA) — was 2.0, too close to noise floor
        self.declare_parameter('nudge_threshold', 10.0)    # N (raw magnitude, bypasses EMA) — was 8.0
        self.declare_parameter('z_score_thresh', 3.0)      # spike score (on raw buffer) — was 2.5

        # Hysteresis / debounce
        self.declare_parameter('guide_enter_count', 25)    # consecutive samples to enter GUIDING (50ms at 500Hz wrench rate)
        self.declare_parameter('guide_exit_count', 1500)   # consecutive samples to exit GUIDING (3s at 500Hz) — long enough for grab-release-grab
        self.declare_parameter('idle_threshold', 2.0)      # N (avg force) to consider idle candidate — was 1.0

        # Nudge cooldown so one hit doesn't spam events
        self.declare_parameter('nudge_cooldown_s', 0.75)

        # Baseline auto-zero — tracks slow FT sensor drift (matches hybrid controller's auto_bias)
        self.declare_parameter('baseline_adapt_rate', 0.002)   # blend rate per sample (very slow)
        self.declare_parameter('baseline_std_gate', 1.0)       # N — only adapt when raw_std < this (stable signal)
        self.declare_parameter('baseline_max', 3.0)            # N — cap prevents gravity-shift from eating threshold
        self.declare_parameter('baseline_settle_s', 3.0)       # seconds after GUIDING exit before adapting baseline

        # Cached params
        self.ft_topic = self.get_parameter('ft_topic').value
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.window_size = int(self.get_parameter('window_size').value)
        self.ema_alpha = float(self.get_parameter('ema_alpha').value)

        self.guide_threshold = float(self.get_parameter('guide_threshold').value)
        self.idle_threshold = float(self.get_parameter('idle_threshold').value)
        self.nudge_threshold = float(self.get_parameter('nudge_threshold').value)
        self.z_score_thresh = float(self.get_parameter('z_score_thresh').value)

        self.guide_enter_count = int(self.get_parameter('guide_enter_count').value)
        self.guide_exit_count = int(self.get_parameter('guide_exit_count').value)
        self.nudge_cooldown_s = float(self.get_parameter('nudge_cooldown_s').value)

        self.baseline_adapt_rate = float(self.get_parameter('baseline_adapt_rate').value)
        self.baseline_std_gate = float(self.get_parameter('baseline_std_gate').value)
        self.baseline_max = float(self.get_parameter('baseline_max').value)
        self.baseline_settle_s = float(self.get_parameter('baseline_settle_s').value)

        # ---------------- State / buffers ----------------
        self.force_buffer = deque(maxlen=self.window_size)
        self.raw_buffer = deque(maxlen=self.window_size)  # raw (un-EMA'd) for nudge detection
        self.ema_force = 0.0
        self.have_ema = False

        self.state = "IDLE"
        self._guiding_enter_hits = 0
        self._guiding_exit_hits = 0
        self._last_nudge_time = -1e9  # seconds
        self._last_guiding_exit_time = -1e9  # seconds — settle gate for baseline
        self.force_baseline = 0.0    # slow-adapting baseline to reject FT drift

        # ---------------- Interfaces ----------------
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub_wrench = self.create_subscription(
            WrenchStamped,
            self.ft_topic,
            self.process_force,
            sensor_qos
        )

        # Reset baseline when hybrid controller is re-enabled (after homing / FT re-tare)
        self.create_subscription(
            Bool, '/hybrid_controller/enabled', self._hybrid_enabled_cb, 10)

        self.pub_state = self.create_publisher(String, '/interaction_state', 10)
        self.pub_guiding = self.create_publisher(Bool, '/interaction/is_guiding', 10)
        self.pub_nudge = self.create_publisher(Bool, '/interaction/is_nudge', 10)

        self.pub_force = self.create_publisher(Float32, '/interaction/force_mag', 10)
        self.pub_avg = self.create_publisher(Float32, '/interaction/force_avg', 10)
        self.pub_z = self.create_publisher(Float32, '/interaction/z_score', 10)

        # Publish outputs at a steady rate (useful for plotting/rosbag)
        self.timer = self.create_timer(1.0 / self.rate_hz, self.publish_outputs)

        self.get_logger().info(
            f"InteractionClassifier running. ft_topic={self.ft_topic}, "
            f"window={self.window_size}, rate={self.rate_hz}Hz"
        )

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def process_force(self, msg: WrenchStamped):
        fx = msg.wrench.force.x
        fy = msg.wrench.force.y
        fz = msg.wrench.force.z
        f_mag = float(np.linalg.norm([fx, fy, fz]))

        # EMA low-pass on magnitude (for GUIDING detection)
        if not self.have_ema:
            self.ema_force = f_mag
            self.have_ema = True
        else:
            a = self.ema_alpha
            self.ema_force = a * f_mag + (1.0 - a) * self.ema_force

        self.force_buffer.append(self.ema_force)
        self.raw_buffer.append(f_mag)

        if len(self.force_buffer) < self.window_size:
            return

        # EMA'd stats for GUIDING detection
        avg_force = float(np.mean(self.force_buffer))

        # Raw stats for NUDGE detection (z-score on unfiltered signal)
        raw_avg = float(np.mean(self.raw_buffer))
        raw_std = float(np.std(self.raw_buffer))
        if raw_std < 1e-3:
            raw_std = 1e-3
        z_score = float((f_mag - raw_avg) / raw_std)

        t = self.now_s()

        # Baseline auto-zero: slowly track FT drift during stable IDLE periods
        # Gate: only adapt after settling from GUIDING (EMA/forces need time to decay)
        settled = (t - self._last_guiding_exit_time) > self.baseline_settle_s
        if self.state == "IDLE" and raw_std < self.baseline_std_gate and settled:
            b = self.baseline_adapt_rate
            self.force_baseline = (1.0 - b) * self.force_baseline + b * avg_force
            self.force_baseline = min(self.force_baseline, self.baseline_max)

        # Subtract baseline from force metrics used for classification
        avg_force = max(0.0, avg_force - self.force_baseline)

        # --------- Classification (with hysteresis) ---------
        new_state = self.state

        # NUDGE: spike in raw signal + above threshold + cooldown
        nudge_ready = (t - self._last_nudge_time) > self.nudge_cooldown_s
        is_nudge = (z_score > self.z_score_thresh) and (f_mag > self.nudge_threshold) and nudge_ready and (self.state != "GUIDING")

        if is_nudge:
            new_state = "NUDGE"
            self._last_nudge_time = t
            # After publishing NUDGE once, next cycles will fall back to IDLE/GUIDING rules.

        else:
            # GUIDING enter/exit hysteresis using consecutive-window counters
            if self.state != "GUIDING":
                if avg_force > self.guide_threshold:
                    self._guiding_enter_hits += 1
                else:
                    self._guiding_enter_hits = 0

                if self._guiding_enter_hits >= self.guide_enter_count:
                    new_state = "GUIDING"
                    self._guiding_exit_hits = 0

            else:
                # currently GUIDING
                if avg_force < self.idle_threshold:
                    self._guiding_exit_hits += 1
                else:
                    self._guiding_exit_hits = 0

                if self._guiding_exit_hits >= self.guide_exit_count:
                    new_state = "IDLE"
                    self._guiding_enter_hits = 0
                    self._last_guiding_exit_time = t

            if self.state != "GUIDING" and new_state != "GUIDING" and new_state != "NUDGE":
                new_state = "IDLE"

        # State transition
        if new_state != self.state:
            self.state = new_state
            self.get_logger().info(
                f"Interaction state: {self.state}  "
                f"(avg={avg_force:.2f}N, baseline={self.force_baseline:.2f}N)"
            )

        # Cache latest computed features for publishing
        self._last_force_mag = f_mag
        self._last_force_avg = avg_force
        self._last_z = z_score

        # One-shot nudge pulse topic
        if is_nudge:
            self.pub_nudge.publish(Bool(data=True))

    def _hybrid_enabled_cb(self, msg: Bool):
        """Reset classifier state when hybrid controller is (re-)enabled."""
        if msg.data:
            self.force_baseline = 0.0
            self.force_buffer.clear()
            self.raw_buffer.clear()
            self.ema_force = 0.0
            self.have_ema = False
            self._guiding_enter_hits = 0
            self._guiding_exit_hits = 0
            self.state = "IDLE"
            self.get_logger().info("Hybrid controller enabled — classifier reset")

    def publish_outputs(self):
        # Publish continuous outputs if we have data
        if not hasattr(self, "_last_force_mag"):
            return

        self.pub_state.publish(String(data=self.state))
        self.pub_guiding.publish(Bool(data=(self.state == "GUIDING")))

        self.pub_force.publish(Float32(data=float(self._last_force_mag)))
        self.pub_avg.publish(Float32(data=float(self._last_force_avg)))
        self.pub_z.publish(Float32(data=float(self._last_z)))


def main():
    rclpy.init()
    node = HumanInteractionClassifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
