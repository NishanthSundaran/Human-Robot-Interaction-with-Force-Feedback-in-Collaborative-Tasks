# target_pose_integrator.py
# CORRECT approach: seed from TF, then update from TF every cycle
# No twist integration needed - just re-seed from current EE pose

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped
import tf2_ros

class TargetPoseIntegrator(Node):

    def __init__(self):
        super().__init__('target_pose_integrator')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.seeded = False
        self.target_pose = None   # holds the fixed target once seeded

        pose_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        self.pose_pub = self.create_publisher(
            PoseStamped, '/hybrid_controller/target_pose', pose_qos)

        self.create_subscription(
            PoseStamped,
            '/force_adaptive_replanner/new_goal',
            self._replanner_goal_cb,
            10,
        )

        self.create_subscription(
            PoseStamped,
            '/hybrid_controller/update_target',
            self._hybrid_update_cb,
            10,
        )

        self.timer = self.create_timer(0.002, self.publish_pose)  # 500 Hz
        self.get_logger().info('TargetPoseIntegrator running')

    def _replanner_goal_cb(self, msg: PoseStamped):
        self.target_pose = msg
        self.seeded = True
        self.get_logger().info(
            f'Target updated by replanner: [{msg.pose.position.x:.3f}, '
            f'{msg.pose.position.y:.3f}, {msg.pose.position.z:.3f}]'
        )

    def _hybrid_update_cb(self, msg: PoseStamped):
        self.target_pose = msg
        self.seeded = True
        self.get_logger().info(
            f'Target updated by hybrid controller: [{msg.pose.position.x:.3f}, '
            f'{msg.pose.position.y:.3f}, {msg.pose.position.z:.3f}]'
        )

    def publish_pose(self):
        if not self.seeded:
            # Seed once from actual robot TF position
            try:
                t = self.tf_buffer.lookup_transform(
                    'base_link', 'tool0', rclpy.time.Time())
                self.target_pose = PoseStamped()
                self.target_pose.header.frame_id = 'base_link'
                self.target_pose.pose.position.x = t.transform.translation.x
                self.target_pose.pose.position.y = t.transform.translation.y
                self.target_pose.pose.position.z = t.transform.translation.z
                self.target_pose.pose.orientation = t.transform.rotation
                self.seeded = True
                self.get_logger().info(
                    f'Target pose seeded: [{self.target_pose.pose.position.x:.3f}, '
                    f'{self.target_pose.pose.position.y:.3f}, '
                    f'{self.target_pose.pose.position.z:.3f}]')
            except Exception:
                return

        self.target_pose.header.stamp = self.get_clock().now().to_msg()
        self.pose_pub.publish(self.target_pose)

def main():
    rclpy.init()
    rclpy.spin(TargetPoseIntegrator())

if __name__ == '__main__':
    main()
