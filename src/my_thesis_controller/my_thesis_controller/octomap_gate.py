#!/usr/bin/env python3
"""
octomap_gate.py

Gates the point cloud feed to the OctoMap based on interaction state.

- During IDLE/NUDGE: passes point cloud through → OctoMap builds for collision avoidance
- During GUIDING: blocks point cloud + clears OctoMap → human is not treated as obstacle

Subscribes:
  /cloud_filtered             (sensor_msgs/PointCloud2)
  /effective_interaction_state (std_msgs/String)

Publishes:
  /cloud_gated                (sensor_msgs/PointCloud2)

Calls:
  /move_group/clear_octomap   (std_srvs/Empty) on entering GUIDING
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from std_srvs.srv import Empty


class OctomapGate(Node):
    def __init__(self):
        super().__init__("octomap_gate")

        self._guiding = False
        self._was_guiding = False

        # QoS to match robot_self_filter output
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._pub = self.create_publisher(PointCloud2, "/cloud_gated", sensor_qos)

        self.create_subscription(
            PointCloud2, "/cloud_filtered", self._cloud_cb, sensor_qos)
        self.create_subscription(
            String, "/effective_interaction_state", self._state_cb, 10)

        self._clear_cli = self.create_client(Empty, "/move_group/clear_octomap")

        self.get_logger().info(
            "octomap_gate: point cloud gated by interaction state. "
            "GUIDING → blocked + OctoMap cleared. IDLE/NUDGE → passed through.")

    def _state_cb(self, msg: String):
        state = msg.data.strip().upper()
        self._guiding = (state == "GUIDING")

        # On transition into GUIDING: clear OctoMap once
        if self._guiding and not self._was_guiding:
            self._clear_octomap()
        self._was_guiding = self._guiding

    def _cloud_cb(self, msg: PointCloud2):
        if not self._guiding:
            self._pub.publish(msg)

    def _clear_octomap(self):
        if not self._clear_cli.service_is_ready():
            self.get_logger().warn("octomap_gate: clear_octomap service not ready.")
            return
        self._clear_cli.call_async(Empty.Request())
        self.get_logger().info("octomap_gate: GUIDING started — OctoMap cleared.")


def main():
    rclpy.init()
    node = OctomapGate()
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
