#!/usr/bin/env python3
"""
Double-tap test tool.

Listens to /interaction/is_nudge and /effective_interaction_state,
detects double-taps using the same logic as assistive_lift_v4_node,
and prints clear feedback.

Usage (while hardware launch is running + interaction_classifier is up):
  python3 scripts/test_doubletap.py

Or launch the classifier manually first:
  ros2 run my_thesis_controller interaction_classifier --ros-args \
    -p nudge_threshold:=2.5 -p z_score_thresh:=2.0 -p nudge_cooldown_s:=0.3

Then run this test.
"""

import time
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String


# These will be overridden by command-line or you can edit them here
DOUBLE_TAP_WINDOW_S = 1.5


class DoubleTapTester(Node):
    def __init__(self):
        super().__init__('doubletap_tester')

        self.declare_parameter('window', DOUBLE_TAP_WINDOW_S)
        self.window = float(self.get_parameter('window').value)

        self.first_tap_time = None
        self.tap_count = 0
        self.double_tap_count = 0
        self.interaction_state = "UNKNOWN"

        self.create_subscription(
            Bool, '/interaction/is_nudge', self._nudge_cb, 10)
        self.create_subscription(
            String, '/effective_interaction_state', self._state_cb, 10)

        self.create_timer(0.1, self._timeout_check)

        self.get_logger().info(
            f'Double-tap tester ready (window={self.window}s). '
            f'Tap the robot!')

    def _state_cb(self, msg):
        old = self.interaction_state
        self.interaction_state = msg.data
        if old != msg.data:
            print(f"  [state] {old} -> {msg.data}")

    def _nudge_cb(self, msg):
        if not msg.data:
            return

        self.tap_count += 1
        now = time.monotonic()

        if self.first_tap_time is None:
            self.first_tap_time = now
            print(f"\n  TAP #{self.tap_count}  (first tap — waiting for second...)")
        else:
            elapsed = now - self.first_tap_time
            if elapsed <= self.window:
                self.double_tap_count += 1
                print(f"\n  TAP #{self.tap_count}  -> DOUBLE TAP #{self.double_tap_count}! "
                      f"(gap={elapsed:.3f}s)")
                print(f"  ============================")
                self.first_tap_time = None
            else:
                print(f"\n  TAP #{self.tap_count}  (too slow: {elapsed:.3f}s > {self.window}s window)")
                print(f"  Resetting — this is now the new first tap.")
                self.first_tap_time = now

    def _timeout_check(self):
        if self.first_tap_time is not None:
            elapsed = time.monotonic() - self.first_tap_time
            if elapsed > self.window:
                print(f"\n  [timeout] First tap expired ({elapsed:.1f}s). Tap again to start fresh.")
                self.first_tap_time = None


def main():
    rclpy.init()
    node = DoubleTapTester()

    print("\n" + "="*50)
    print("  DOUBLE-TAP TESTER")
    print("="*50)
    print(f"  Window: {node.window}s")
    print(f"  Listening to: /interaction/is_nudge")
    print(f"  State from:   /effective_interaction_state")
    print()
    print("  Tap the robot to test. Ctrl+C to stop.")
    print("="*50 + "\n")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print(f"\n\nResults: {node.tap_count} taps, "
              f"{node.double_tap_count} double-taps detected.\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
