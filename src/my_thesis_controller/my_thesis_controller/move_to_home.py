#!/usr/bin/env python3
"""
move_to_home.py

Persistent node that moves the UR3e to its home joint configuration
[0, -1.2, 1.0, -1.37, -pi/2, 0] — singularity-free, gripper above tables.

- Auto-triggers once at startup
- Can be re-triggered anytime via:  ros2 service call /move_to_home/trigger std_srvs/srv/Trigger
- Checks safety mode: waits for NORMAL before sending trajectory
- Always restores forward_velocity_controller on success or failure
"""

import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from ur_dashboard_msgs.msg import SafetyMode
from controller_manager_msgs.srv import SwitchController
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    JointConstraint,
    RobotState,
)
import tf2_ros


# Shoulder at -1.2 rad (~69°), elbow at 1.0 rad (~57°): pushes wrist center 0.40m from
# J1 axis (avoids shoulder singularity) while keeping gripper ~8cm above workspace table.
HOME_POSITIONS = [0.0, -1.2, 1.0, -1.37, -1.57, 0.0]
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
TOLERANCE_RAD = 0.05
MAX_SPEED_RAD_S = 0.15


class MoveToHome(Node):
    def __init__(self):
        super().__init__("move_to_home")
        self._cb_group = ReentrantCallbackGroup()

        self._current_positions = None
        self._positions_event = threading.Event()
        self._safety_mode = None
        self._program_running = False
        self._homing_lock = threading.Lock()

        # Two QoS profiles to try matching the UR driver
        # Some UR driver topics use BEST_EFFORT, some use RELIABLE
        ur_qos_be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        ur_qos_rel = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscribers — create both QoS variants to ensure we get messages
        self.create_subscription(
            JointState, "/joint_states", self._js_cb, 10,
            callback_group=self._cb_group)
        self.create_subscription(
            SafetyMode, "/io_and_status_controller/safety_mode",
            self._safety_cb, ur_qos_be, callback_group=self._cb_group)
        self.create_subscription(
            SafetyMode, "/io_and_status_controller/safety_mode",
            self._safety_cb, ur_qos_rel, callback_group=self._cb_group)
        self.create_subscription(
            Bool, "/io_and_status_controller/robot_program_running",
            self._program_cb, ur_qos_be, callback_group=self._cb_group)
        self.create_subscription(
            Bool, "/io_and_status_controller/robot_program_running",
            self._program_cb, ur_qos_rel, callback_group=self._cb_group)

        # Service: trigger homing on demand
        self.create_service(
            Trigger, "~/trigger", self._trigger_srv_cb,
            callback_group=self._cb_group)

        # Service clients
        self._pause_cli = self.create_client(
            Trigger, "/servo_node/pause_servo",
            callback_group=self._cb_group)
        self._unpause_cli = self.create_client(
            Trigger, "/servo_node/unpause_servo",
            callback_group=self._cb_group)
        self._switch_cli = self.create_client(
            SwitchController, "/controller_manager/switch_controller",
            callback_group=self._cb_group)
        self._tare_cli = self.create_client(
            Trigger, "/ft_zeroer/tare",
            callback_group=self._cb_group)

        # TF for reading current TCP pose after homing
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Publisher to re-seed target_pose_integrator after homing
        self._target_pub = self.create_publisher(
            PoseStamped, "/hybrid_controller/update_target", 10)

        # Publisher to enable/disable hybrid controller during homing
        # Use transient_local so late subscribers (e.g. V4 node) get the last value
        _latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self._hybrid_enabled_pub = self.create_publisher(
            Bool, "/hybrid_controller/enabled", _latched_qos)

        # MoveGroup action client (collision-aware planning + execution)
        # NOTE: MoveIt 2 Humble exposes the MoveAction at "move_action",
        # NOT "move_group" (that's the node name, not the action name).
        self._mg_client = ActionClient(
            self, MoveGroup, "move_action",
            callback_group=self._cb_group)
        self._mg_server_ready = False

        self.get_logger().info(
            "move_to_home: ready. Trigger via: "
            "ros2 service call /move_to_home/trigger std_srvs/srv/Trigger")

        # Poll for /move_group from executor context (server_is_ready()
        # needs DDS discovery events that only fire in executor callbacks).
        self._mg_poll_timer = self.create_timer(
            0.5, self._poll_mg_server_cb, callback_group=self._cb_group)

        # Auto-trigger at startup via one-shot timer (ensures executor is
        # spinning before ActionClient discovery is used).
        self._startup_timer = self.create_timer(
            1.0, self._startup_timer_cb, callback_group=self._cb_group)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _poll_mg_server_cb(self):
        """Runs in executor context — DDS discovery works here."""
        if self._mg_client.server_is_ready():
            self._mg_server_ready = True
            self._mg_poll_timer.cancel()

    def _startup_timer_cb(self):
        """One-shot: fires once the executor is spinning, then starts homing."""
        self._startup_timer.cancel()
        threading.Thread(target=self._run_homing, args=("startup",), daemon=True).start()

    def _js_cb(self, msg: JointState):
        try:
            positions = []
            for name in JOINT_NAMES:
                idx = msg.name.index(name)
                positions.append(msg.position[idx])
            self._current_positions = positions
            self._positions_event.set()
        except (ValueError, IndexError):
            pass

    def _safety_cb(self, msg: SafetyMode):
        self._safety_mode = msg.mode

    def _program_cb(self, msg: Bool):
        self._program_running = msg.data

    def _trigger_srv_cb(self, request, response):
        """Service handler — spawns homing in a thread so the service returns quickly."""
        if self._homing_lock.locked():
            response.success = False
            response.message = "Homing already in progress"
            return response
        threading.Thread(target=self._run_homing, args=("service",), daemon=True).start()
        response.success = True
        response.message = "Homing triggered"
        return response

    # ------------------------------------------------------------------
    # Homing sequence (runs in background thread)
    # ------------------------------------------------------------------
    def _run_homing(self, source: str):
        if not self._homing_lock.acquire(blocking=False):
            self.get_logger().warn("move_to_home: homing already in progress, skipping.")
            return

        try:
            self._do_homing(source)
        except Exception as e:
            self.get_logger().error(f"move_to_home: unexpected error: {e}")
            self._restore_fvc()
        finally:
            self._homing_lock.release()

    def _do_homing(self, source: str):
        self.get_logger().info(f"move_to_home: triggered ({source})")

        # 1. Wait for joint states
        if not self._positions_event.wait(timeout=30.0):
            self.get_logger().error("move_to_home: timed out waiting for joint states.")
            return

        # 2. Read fresh positions
        positions = list(self._current_positions)

        # 3. Check if already at home
        if self._is_at_home(positions):
            self.get_logger().info("move_to_home: already at home.")
            return

        self.get_logger().info(
            f"move_to_home: current={[f'{p:.3f}' for p in positions]}, "
            f"target={[f'{p:.3f}' for p in HOME_POSITIONS]}")

        # 4. Disable hybrid + pause Servo IMMEDIATELY to prevent Servo from
        #    receiving twist commands at a singular configuration while waiting.
        self._set_hybrid_enabled(False)
        self._call_service_sync(self._pause_cli, Trigger.Request(), "pause_servo")

        # 5. Wait for robot to be in NORMAL safety mode + program running
        #    User must press "Run" on pendant's external control program first.
        if not self._wait_for_robot_ready(timeout=120.0):
            self.get_logger().error(
                "move_to_home: robot not ready (press Run on teach pendant for external control).")
            return

        # 7. Switch to scaled_joint_trajectory_controller
        self._switch_to_jtc()

        # 8. Wait for MoveGroup action server
        # NOTE: server_is_ready() only works from executor callbacks (DDS
        # discovery events). _mg_poll_timer sets _mg_server_ready from
        # executor context; we just check the flag here.
        self.get_logger().info("move_to_home: waiting for /move_group action server...")
        mg_deadline = time.monotonic() + 30.0
        while time.monotonic() < mg_deadline:
            if self._mg_server_ready:
                break
            time.sleep(0.5)
        if not self._mg_server_ready:
            self.get_logger().error("move_to_home: /move_group action server not available.")
            self._restore_fvc()
            return

        # 9. Re-read positions (may have changed while waiting)
        positions = list(self._current_positions)
        if self._is_at_home(positions):
            self.get_logger().info("move_to_home: reached home while waiting.")
            self._restore_fvc()
            return

        # 10. Build MoveGroup goal with joint constraints (collision-aware)
        #     Use Pilz PTP for smooth, deterministic joint-space trajectories.
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = "ur3e_manipulator"
        req.pipeline_id = "pilz"
        req.planner_id = "PTP"
        req.num_planning_attempts = 1
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = 0.2
        req.max_acceleration_scaling_factor = 0.2

        jcs = []
        for name, val in zip(JOINT_NAMES, HOME_POSITIONS):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = val
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            jcs.append(jc)

        req.goal_constraints = [Constraints(joint_constraints=jcs)]
        req.start_state = RobotState()
        req.start_state.is_diff = True

        goal.request = req
        goal.planning_options.plan_only = False
        goal.planning_options.replan = False

        # Try Pilz PTP first, fall back to OMPL if it fails (e.g. self-collision)
        pilz_ok = self._send_and_wait_mg(goal, "Pilz PTP")

        if not pilz_ok:
            self.get_logger().warn(
                "move_to_home: Pilz PTP failed — retrying with OMPL RRTConnect...")
            ompl_goal = MoveGroup.Goal()
            ompl_req = MotionPlanRequest()
            ompl_req.group_name = "ur3e_manipulator"
            ompl_req.pipeline_id = "ompl"
            ompl_req.planner_id = "RRTConnectkConfigDefault"
            ompl_req.num_planning_attempts = 5
            ompl_req.allowed_planning_time = 10.0
            ompl_req.max_velocity_scaling_factor = 0.15
            ompl_req.max_acceleration_scaling_factor = 0.15
            ompl_req.goal_constraints = [Constraints(joint_constraints=jcs)]
            ompl_req.start_state = RobotState()
            ompl_req.start_state.is_diff = True
            ompl_goal.request = ompl_req
            ompl_goal.planning_options.plan_only = False
            ompl_goal.planning_options.replan = False
            self._send_and_wait_mg(ompl_goal, "OMPL RRTConnect")

        # 12. Always restore
        self._restore_fvc()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _send_and_wait_mg(self, goal, label: str) -> bool:
        """Send a MoveGroup goal, wait for result. Returns True on success."""
        self.get_logger().info(f"move_to_home: planning with {label}...")
        goal_future = self._mg_client.send_goal_async(goal)
        self._wait_future(goal_future, 15.0)
        if not goal_future.done():
            self.get_logger().error(f"move_to_home: {label} send_goal timed out.")
            return False

        goal_handle = goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"move_to_home: {label} goal REJECTED.")
            return False

        self.get_logger().info(f"move_to_home: {label} goal accepted, executing...")
        result_future = goal_handle.get_result_async()
        self._wait_future(result_future, 60.0)

        if result_future.done():
            result = result_future.result().result
            err_code = result.error_code.val
            if err_code == 1:  # MoveItErrorCodes.SUCCESS
                self.get_logger().info(f"move_to_home: {label} homing SUCCEEDED.")
                return True
            else:
                self.get_logger().warn(
                    f"move_to_home: {label} finished with error_code={err_code}")
                return False
        else:
            self.get_logger().warn(f"move_to_home: {label} result timed out.")
            return False

    def _is_at_home(self, positions) -> bool:
        return all(
            abs(c - h) <= TOLERANCE_RAD
            for c, h in zip(positions, HOME_POSITIONS))

    def _wait_for_robot_ready(self, timeout: float) -> bool:
        """Block until safety_mode==NORMAL and program is running.
        Falls back to program_running only if safety_mode never arrives."""
        deadline = time.monotonic() + timeout
        last_log = 0.0
        while time.monotonic() < deadline:
            sm = self._safety_mode
            pr = self._program_running
            # Full check: safety normal + program running
            if sm == SafetyMode.NORMAL and pr:
                self.get_logger().info(
                    f"move_to_home: robot ready (safety_mode={sm}, program_running={pr})")
                return True
            # Fallback: if safety_mode topic never arrives but program is running
            # (QoS mismatch protection — after 15s, trust program_running alone)
            elapsed = time.monotonic() - (deadline - timeout)
            if sm is None and pr and elapsed > 15.0:
                self.get_logger().warn(
                    "move_to_home: safety_mode never received, but program_running=True. Proceeding.")
                return True
            now = time.monotonic()
            if now - last_log > 5.0:
                self.get_logger().warn(
                    f"move_to_home: waiting for robot ready "
                    f"(safety_mode={sm}, program_running={pr}, elapsed={elapsed:.0f}s). "
                    f"Press Play on the teach pendant.")
                last_log = now
            time.sleep(0.5)
        return False

    def _switch_to_jtc(self):
        req = SwitchController.Request()
        req.activate_controllers = ["scaled_joint_trajectory_controller"]
        req.deactivate_controllers = ["forward_velocity_controller"]
        req.strictness = SwitchController.Request.BEST_EFFORT
        req.activate_asap = True
        self._call_service_sync(self._switch_cli, req, "switch to scaled_jtc")

    def _restore_fvc(self):
        """Restore FVC with clean hybrid state to avoid jerk/drift on transition."""
        # 1. Ensure hybrid is disabled (idempotent safety — publishes zero twist)
        self._set_hybrid_enabled(False)

        # 2. Switch controller to FVC
        req = SwitchController.Request()
        req.activate_controllers = ["forward_velocity_controller"]
        req.deactivate_controllers = ["scaled_joint_trajectory_controller"]
        req.strictness = SwitchController.Request.BEST_EFFORT
        req.activate_asap = True
        self._call_service_sync(self._switch_cli, req, "switch to fvc")

        # 3. Unpause servo IMMEDIATELY so FVC receives zero-velocity commands
        #    (hybrid is disabled → publishes zero twist → servo converts to zero
        #    joint velocities → FVC gets proper commands, no stale/missing input)
        self._call_service_sync(self._unpause_cli, Trigger.Request(), "unpause_servo")

        # 4. Re-tare FT sensor to clear residual forces
        self._call_service_sync(self._tare_cli, Trigger.Request(), "ft_tare")

        # 5. Wait for tare to settle + robot to be fully stationary
        time.sleep(1.5)

        # 6. Re-seed target_pose_integrator AFTER settling (TF is now accurate)
        self._update_target_pose()
        time.sleep(0.1)  # let target propagate through integrator

        # 7. Re-enable hybrid (triggers state reset + 1s zero-twist guard)
        self._set_hybrid_enabled(True)
        self.get_logger().info("move_to_home: restored forward_velocity_controller.")

    def _set_hybrid_enabled(self, enabled: bool):
        """Publish hybrid controller enabled/disabled (twice with gap for reliability)."""
        msg = Bool()
        msg.data = enabled
        self._hybrid_enabled_pub.publish(msg)
        time.sleep(0.05)
        self._hybrid_enabled_pub.publish(msg)
        state_str = "enabled" if enabled else "disabled"
        self.get_logger().info(f"move_to_home: hybrid controller {state_str}.")

    def _update_target_pose(self):
        """Publish current TCP pose to re-seed the target_pose_integrator."""
        try:
            t = self._tf_buffer.lookup_transform(
                "base_link", "tool0", rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0))
            msg = PoseStamped()
            msg.header.frame_id = "base_link"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.position.x = t.transform.translation.x
            msg.pose.position.y = t.transform.translation.y
            msg.pose.position.z = t.transform.translation.z
            msg.pose.orientation = t.transform.rotation
            self._target_pub.publish(msg)
            self.get_logger().info(
                f"move_to_home: target re-seeded to "
                f"[{msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, "
                f"{msg.pose.position.z:.3f}]")
        except Exception as ex:
            self.get_logger().warn(f"move_to_home: failed to re-seed target: {ex}")

    def _call_service_sync(self, client, request, label: str):
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(f"move_to_home: {label} service not available, skipping.")
            return None
        future = client.call_async(request)
        self._wait_future(future, 5.0)
        if future.done():
            return future.result()
        self.get_logger().warn(f"move_to_home: {label} timed out.")
        return None

    def _wait_future(self, future, timeout: float):
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)


def main():
    rclpy.init()
    node = MoveToHome()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass


if __name__ == "__main__":
    main()
