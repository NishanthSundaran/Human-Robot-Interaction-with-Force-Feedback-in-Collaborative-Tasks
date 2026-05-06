#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import Trigger

import numpy as np


class FTZeroer(Node):
    """
    Subscribes to a WrenchStamped topic, estimates a bias (tare) over N samples,
    and publishes a bias-corrected (zeroed) WrenchStamped.
    """

    def __init__(self):
        super().__init__("ft_zeroer")

        # Parameters
        self.declare_parameter("input_topic", "/force_torque_sensor_broadcaster/wrench")
        self.declare_parameter("output_topic", "/wrench_zeroed")
        self.declare_parameter("tare_samples", 250)          # ~0.5s at 500 Hz
        self.declare_parameter("auto_tare_on_start", True)
        self.declare_parameter("use_mean_filter", True)      # mean over window
        self.declare_parameter("stamp_output_with_input", True)

        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.tare_samples = int(self.get_parameter("tare_samples").value)
        self.auto_tare_on_start = bool(self.get_parameter("auto_tare_on_start").value)
        self.use_mean_filter = bool(self.get_parameter("use_mean_filter").value)
        self.stamp_output_with_input = bool(self.get_parameter("stamp_output_with_input").value)

        # QoS suitable for high-rate sensor data
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
        )

        self.sub = self.create_subscription(WrenchStamped, self.input_topic, self.cb, sensor_qos)
        self.pub = self.create_publisher(WrenchStamped, self.output_topic, sensor_qos)

        # Service to trigger re-tare
        self.srv = self.create_service(Trigger, "~/tare", self.handle_tare)

        # State for bias estimation
        self._buffer = []
        self._bias = np.zeros(6, dtype=float)
        self._have_bias = False
        self._taring = False

        if self.auto_tare_on_start:
            self.start_tare()

        self.get_logger().info(
            f"FTZeroer running.\n"
            f"  input_topic:  {self.input_topic}\n"
            f"  output_topic: {self.output_topic}\n"
            f"  tare_samples: {self.tare_samples}\n"
            f"  auto_tare_on_start: {self.auto_tare_on_start}\n"
            f"Service: {self.get_name()}/tare (std_srvs/Trigger)"
        )

    def start_tare(self):
        self._buffer = []
        self._taring = True
        self._have_bias = False
        self.get_logger().info("Tare requested: collecting samples...")

    def handle_tare(self, request, response):
        self.start_tare()
        response.success = True
        response.message = f"Tare started. Collecting {self.tare_samples} samples."
        return response

    def cb(self, msg: WrenchStamped):
        w = np.array(
            [
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
                msg.wrench.torque.x,
                msg.wrench.torque.y,
                msg.wrench.torque.z,
            ],
            dtype=float,
        )

        # Collect tare samples if requested
        if self._taring:
            self._buffer.append(w)
            if len(self._buffer) >= self.tare_samples:
                buf = np.vstack(self._buffer)
                self._bias = buf.mean(axis=0) if self.use_mean_filter else buf[-1].copy()
                self._have_bias = True
                self._taring = False
                self.get_logger().info(
                    "Tare complete. Bias set to:\n"
                    f"  F: [{self._bias[0]: .4f}, {self._bias[1]: .4f}, {self._bias[2]: .4f}] N\n"
                    f"  T: [{self._bias[3]: .4f}, {self._bias[4]: .4f}, {self._bias[5]: .4f}] Nm"
                )

        # Publish zeroed wrench once bias exists; otherwise publish raw (optional behavior)
        w_zero = w - self._bias if self._have_bias else w

        out = WrenchStamped()
        out.header = msg.header if self.stamp_output_with_input else self.get_clock().now().to_msg()
        out.header.frame_id = msg.header.frame_id  # keep sensor frame
        out.wrench.force.x = float(w_zero[0])
        out.wrench.force.y = float(w_zero[1])
        out.wrench.force.z = float(w_zero[2])
        out.wrench.torque.x = float(w_zero[3])
        out.wrench.torque.y = float(w_zero[4])
        out.wrench.torque.z = float(w_zero[5])

        self.pub.publish(out)


def main():
    rclpy.init()
    node = FTZeroer()
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


if __name__ == "__main__":
    main()
