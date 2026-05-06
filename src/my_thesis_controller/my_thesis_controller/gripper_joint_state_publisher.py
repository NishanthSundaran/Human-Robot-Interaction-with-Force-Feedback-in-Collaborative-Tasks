#!/usr/bin/env python3
"""Publish default joint states for the Robotiq 140 gripper (all positions = 0).

When no physical gripper is connected, robot_state_publisher still needs
joint-state messages for every non-fixed joint in the URDF so it can
broadcast the corresponding TF frames.  This node publishes all six
revolute / mimic gripper joints at 50 Hz.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

GRIPPER_JOINTS = [
    "robotiq_140_finger_joint",
    "robotiq_140_left_inner_knuckle_joint",
    "robotiq_140_right_inner_knuckle_joint",
    "robotiq_140_left_inner_finger_joint",
    "robotiq_140_right_inner_finger_joint",
    "robotiq_140_right_outer_knuckle_joint",
]


class GripperJointStatePublisher(Node):
    def __init__(self):
        super().__init__("gripper_joint_state_publisher")
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self.timer = self.create_timer(1.0 / 50.0, self.publish)
        self.get_logger().info(
            f"Publishing default joint states for {len(GRIPPER_JOINTS)} gripper joints at 50 Hz"
        )

    def publish(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(GRIPPER_JOINTS)
        msg.position = [0.0] * len(GRIPPER_JOINTS)
        msg.velocity = [0.0] * len(GRIPPER_JOINTS)
        msg.effort = [0.0] * len(GRIPPER_JOINTS)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = GripperJointStatePublisher()
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
