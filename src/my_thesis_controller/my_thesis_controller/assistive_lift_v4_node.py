#!/usr/bin/env python3
"""
assistive_lift_v4_node.py

Thesis validation: Human-Guided Placement — V4.

Differences from v2/v3:
  - Robot picks object and lifts, then enters servo/hybrid mode WITH object
  - Human guides the robot (with object) to the desired placement location
  - On double tap (NUDGE), robot computes safe placement height from object dims
  - If current height is unsafe (too high), robot descends to safe height
  - Robot places the object and retracts

FSM:
  INIT → PERCEIVE → PLAN_GRASP → MOVE_TO_PREGRASP → MOVE_TO_GRASP →
  GRASP → LIFT → ENABLE_SERVO → AWAIT_INTERACTION →
  PREP_PLACE → [DESCEND_TO_PLACE →] PLACE → RETRACT → DONE
"""

import csv
import datetime
import math
import os
import subprocess
import threading

import numpy as np
from scipy.optimize import minimize as scipy_minimize

try:
    import cv2
    from cv_bridge import CvBridge
    _HAS_CV = True
except ImportError:
    _HAS_CV = False

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from std_msgs.msg import Bool, Float64MultiArray, String
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped, Pose
from std_srvs.srv import Trigger
from controller_manager_msgs.srv import SwitchController
from moveit_msgs.msg import (
    CollisionObject,
    AttachedCollisionObject,
    PlanningScene,
    MotionPlanRequest,
    Constraints,
    JointConstraint,
    PositionConstraint,
    OrientationConstraint,
    MoveItErrorCodes,
    RobotState,
    BoundingVolume,
)
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from shape_msgs.msg import SolidPrimitive
import tf2_ros


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PLANNING_GROUP = "ur3e_manipulator"
EE_LINK = "tool0"
BASE_FRAME = "base_link"
CAMERA_OPTICAL_FRAME = "camera_depth_optical_frame"

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

PREGRASP_OFFSET = 0.10   # metres above grasp for pre-grasp / lift
TOOL0_TO_FINGERTIP_DEFAULT = 0.245  # tool0 flange → fingertip (measured)
MIN_TABLE_CLEARANCE = 0.02  # minimum gap between fingertip bottom and table (m)

FALLBACK_GRASP_JOINTS = [0.9458, -1.0842, 0.8970, -1.3379, -1.5634, -0.6528]

# Safe placement
WORKSPACE_TABLE_Z = 0.06  # workspace table surface height in base_link (m)
PLACE_MARGIN_M = 0.005  # 5 mm above surface for gentle placement
DESCENT_THRESHOLD_M = 0.005  # descend if more than 5 mm away from safe height

# Workspace table bounds in base_link (reject detections outside this ROI)
# Table collision box: centre (-0.645, 0, -0.26), dims [0.59, 0.79, 0.74]
# Surface at z=0.11.  Add generous margin for camera noise.
WS_X_MIN, WS_X_MAX = -0.30,  0.35   # base_link x: shifted to -30cm (left in camera)
WS_Y_MIN, WS_Y_MAX =  0.00,  0.80   # base_link y: obj at ~0.39, table ~79cm wide
WS_Z_MIN, WS_Z_MAX = -0.05,  0.30   # surface ~0.09, objects up to ~20cm above
MAX_OBJECT_HEIGHT_M = 0.15  # nothing graspable is taller than 15cm above table

LIFT_HEIGHT = 0.10  # metres to lift after grasping (matches PREGRASP_OFFSET)

# Double-tap detection
DOUBLE_TAP_WINDOW_S = 1.5  # max time between two taps to count as double-tap

# Gripper
GRIPPER_OPEN_SIM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
GRIPPER_OPEN_HW = [0.0]
GRIPPER_MAX_OPENING_M = 0.140
GRIPPER_MAX_JOINT_VAL = 0.7
GRIPPER_MARGIN_M = 0.005  # 5 mm clearance each side
DEFAULT_GRIPPED_WIDTH = 0.03

BOX_ID = "grasp_box"
DEFAULT_BOX_DIMS = [0.065, 0.03, 0.06]

ROBOTIQ_TOUCH_LINKS = [
    "robotiq_140_robotiq_140_base_link",
    "robotiq_140_left_outer_knuckle",
    "robotiq_140_left_outer_finger",
    "robotiq_140_left_inner_finger",
    "robotiq_140_left_inner_finger_pad",
    "robotiq_140_left_inner_knuckle",
    "robotiq_140_right_outer_knuckle",
    "robotiq_140_right_outer_finger",
    "robotiq_140_right_inner_finger",
    "robotiq_140_right_inner_finger_pad",
    "robotiq_140_right_inner_knuckle",
]

# Perception
RANSAC_ITERATIONS = 100
RANSAC_THRESHOLD_M = 0.005
BOX_MIN_HEIGHT_M = 0.01
YOLO_CONF_THRESH = 0.3
HSV_RED_MIN_FRAC = 0.25
MAX_PERCEPTION_ATTEMPTS = 30

# Timing
INTERACTION_TIMEOUT_S = 120.0
GUARD_PERIOD_S = 1.5
TARE_SETTLE_S = 2.0
FORCE_THRESHOLD_N = 5.0
VELOCITY_THRESHOLD_MPS = 0.01


# ---------------------------------------------------------------------------
# UR3e FK / IK (DH parameters, 180° Z correction for base_link)
# ---------------------------------------------------------------------------
_DH_D = [0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921]
_DH_A = [0.0, -0.24365, -0.21325, 0.0, 0.0, 0.0]
_DH_ALPHA = [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]
_RZ_PI = np.diag([-1.0, -1.0, 1.0, 1.0])
_UR3E_MAX_REACH = 0.50


def _dh_matrix(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,     d],
        [0.0,     0.0,      0.0,   1.0],
    ])


def _fk(joints):
    """UR3e FK → 4×4 transform in base_link frame."""
    T = np.eye(4)
    for i in range(6):
        T = T @ _dh_matrix(joints[i], _DH_D[i], _DH_A[i], _DH_ALPHA[i])
    return _RZ_PI @ T


def _solve_ik(target_pos, seed_joints, max_iter=30000):
    """Numerical IK: find joints placing tool0 at target_pos, tool pointing down.

    Returns (joints, pos_error_mm, ori_error_deg) or None.
    """
    dist = np.sqrt(target_pos[0] ** 2 + target_pos[1] ** 2)
    if dist > _UR3E_MAX_REACH:
        return None

    target_rot = np.array([[-1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, -1.0]])

    def cost(q):
        T = _fk(q)
        dp = T[:3, 3] - target_pos
        dR = T[:3, :3] - target_rot
        return 1e4 * np.sum(dp ** 2) + 1e2 * np.sum(dR ** 2)

    seed = np.array(seed_joints, dtype=float)
    r = scipy_minimize(cost, seed, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-10,
                                'fatol': 1e-14, 'adaptive': True})
    r2 = scipy_minimize(cost, r.x, method='Powell',
                        options={'maxiter': max_iter, 'ftol': 1e-16,
                                 'xtol': 1e-12})

    q_sol = (r2.x + np.pi) % (2 * np.pi) - np.pi
    T_sol = _fk(q_sol)
    pos_err = np.linalg.norm(T_sol[:3, 3] - target_pos) * 1000
    R_err = T_sol[:3, :3].T @ target_rot
    cos_a = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    ori_err = np.degrees(np.arccos(cos_a))

    if pos_err > 5.0 or ori_err > 15.0:
        return None
    return q_sol.tolist(), pos_err, ori_err


# ---------------------------------------------------------------------------
# Perception helpers
# ---------------------------------------------------------------------------
def _quat_to_rot(x, y, z, w):
    """Quaternion (x,y,z,w) → 3×3 rotation matrix."""
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def _ransac_plane(points, n_iter=RANSAC_ITERATIONS, thresh=RANSAC_THRESHOLD_M):
    """RANSAC plane fit.  Returns (unit_normal, d) where n·p + d ≈ 0, or None."""
    n = len(points)
    if n < 10:
        return None
    best_inliers = 0
    best_plane = None
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        if normal[2] < 0:
            normal = -normal
        d = -np.dot(normal, p0)
        dists = np.abs(points @ normal + d)
        inliers = int(np.sum(dists < thresh))
        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = (normal.copy(), float(d))
    return best_plane


def _compute_finger_joint(gripped_width_m):
    """finger_joint value for a given gripped width.

    Close *tighter* than the box so the position controller pushes
    against the surfaces and generates grip force in simulation.
    """
    desired = max(0.0, gripped_width_m - 2 * GRIPPER_MARGIN_M)
    fj = GRIPPER_MAX_JOINT_VAL * (1.0 - desired / GRIPPER_MAX_OPENING_M)
    return max(0.0, min(GRIPPER_MAX_JOINT_VAL, fj))


def _make_gripper_cmd(finger_joint):
    """6-element Robotiq 2F-140 command from a single finger_joint value."""
    fj = finger_joint
    return [fj, -fj, fj, -fj, -fj, fj]


# ---------------------------------------------------------------------------
# FSM states
# ---------------------------------------------------------------------------
class Phase:
    INIT = "INIT"
    PERCEIVE = "PERCEIVE"
    PLAN_GRASP = "PLAN_GRASP"
    MOVE_TO_PREGRASP = "MOVE_TO_PREGRASP"
    MOVE_TO_GRASP = "MOVE_TO_GRASP"
    GRASP = "GRASP"
    LIFT = "LIFT"
    ENABLE_SERVO = "ENABLE_SERVO"
    AWAIT_INTERACTION = "AWAIT_INTERACTION"
    PREP_PLACE = "PREP_PLACE"
    DESCEND_TO_PLACE = "DESCEND_TO_PLACE"
    PLACE = "PLACE"
    RETRACT = "RETRACT"
    DONE = "DONE"


# ===================================================================
# Node
# ===================================================================
class AssistiveLiftV4Node(Node):
    def __init__(self):
        super().__init__("assistive_lift_v4")

        self.declare_parameter("use_gripper", False)
        self.declare_parameter("jtc_name", "joint_trajectory_controller")
        self.declare_parameter("rgb_topic", "/realsense/image")
        self.declare_parameter("depth_topic", "/realsense/depth_image")
        self.declare_parameter("camera_info_topic", "/realsense/camera_info")
        self.declare_parameter("camera_optical_frame", "camera_depth_optical_frame")
        self.declare_parameter(
            "gripper_topic",
            "/robotiq_group_position_controller/commands")
        self.declare_parameter("hardware_gripper", False)
        self.declare_parameter("tool0_to_fingertip", TOOL0_TO_FINGERTIP_DEFAULT)
        self._use_gripper = self.get_parameter("use_gripper").value
        self._use_sim = self.get_parameter("use_sim_time").value
        self._jtc_name = self.get_parameter("jtc_name").value
        self._rgb_topic = self.get_parameter("rgb_topic").value
        self._depth_topic = self.get_parameter("depth_topic").value
        self._camera_info_topic = self.get_parameter("camera_info_topic").value
        self._camera_optical_frame = self.get_parameter("camera_optical_frame").value
        self._gripper_topic = self.get_parameter("gripper_topic").value
        self._hardware_gripper = self.get_parameter("hardware_gripper").value
        self._tool0_to_fingertip = self.get_parameter("tool0_to_fingertip").value

        self._phase = Phase.INIT
        self._phase_start = self.get_clock().now()
        self._timestamps = {}

        # MoveGroup
        self._mg_server_ready = False
        self._goal_handle = None
        self._move_result = None

        # Controller switch
        self._switch_done = False

        # Homing
        self._hybrid_enabled = False

        # Tare
        self._tare_done = False

        # TF
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Camera
        self._bridge = CvBridge() if _HAS_CV else None
        self._K = None
        self._rgb = None
        self._depth = None

        # YOLO
        self._yolo = None
        if _HAS_YOLO:
            try:
                self._yolo = YOLO("yolov8n.pt")
                self.get_logger().info("YOLOv8n model loaded.")
            except Exception as e:
                self.get_logger().warn(f"YOLO load failed: {e}")

        # Perception results
        self._box_center_bl = None   # [x,y,z] geometric centre in base_link
        self._box_dims = None        # [lx, ly, lz] metres
        self._box_yaw = 0.0          # degrees on table plane
        self._table_z = WORKSPACE_TABLE_Z  # detected table surface Z
        self._perception_done = False
        self._perception_attempts = 0

        # IK / grasp planning results
        self._grasp_joints = None
        self._pregrasp_pos = None       # [x,y,z] for MoveGroup pose goal
        self._grasp_tool0_target = None
        self._descent_distance = PREGRASP_OFFSET  # actual pregrasp-to-grasp Z
        self._gripper_cmd = None
        self._ik_solved = False
        self._ik_future = None
        self._pregrasp_ik_joints = None

        # Phase-entry flags
        self._plan_grasp_started = False
        self._grasp_started = False
        self._grasp_switch_started = False
        self._prep_place_started = False
        self._retract_lift_sent = False

        # Current joint state (for IK seed during placement)
        self._current_joints = None

        # Recorded positions
        self._lift_position = None
        self._guided_position = None  # EE pos when NUDGE received
        self._place_position = None   # EE pos after descent (actual placement)
        self._descent_needed = False
        self._descent_amount = 0.0

        # Input thread (no-gripper mode)
        self._input_received = False

        # Interaction classifier process (started after lift)
        self._classifier_proc = None

        # Interaction metrics
        self._interaction_state = "IDLE"
        self._prev_interaction_state = "IDLE"
        self._nudge_received = False  # single tap flag (internal)
        self._double_tap_received = False
        self._first_tap_time = None  # timestamp of first tap
        self._guide_cycles = []
        self._current_cycle = None
        self._guide_cycle_count = 0
        self._force_samples = []
        self._velocity_samples = []
        self._state_log = []
        self._experiment_dir = os.path.expanduser("~/ros2_ws/experiment_data")
        os.makedirs(self._experiment_dir, exist_ok=True)
        self._trial_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._double_tap_time_ms = None
        self._force_above_thresh_time = None
        self._velocity_above_thresh_time = None
        self._await_start_time = None
        self._last_interaction_time = None

        # ── Action clients ──
        self._move_client = ActionClient(self, MoveGroup, "move_action")
        self._exec_client = ActionClient(
            self, ExecuteTrajectory, "execute_trajectory")

        # ── Service client for Cartesian path ──
        self._cartesian_srv = self.create_client(
            GetCartesianPath, "/compute_cartesian_path")

        # ── Service client for IK (Pilz pre-solve) ──
        self._ik_srv = self.create_client(GetPositionIK, "/compute_ik")

        # ── Publishers ──
        self._pub_planning_scene = self.create_publisher(
            PlanningScene, "/planning_scene", 10)
        self._pub_gripper = self.create_publisher(
            Float64MultiArray, self._gripper_topic, 10)
        _latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self._pub_enabled = self.create_publisher(
            Bool, "/hybrid_controller/enabled", _latched_qos)
        self._pub_update_target = self.create_publisher(
            PoseStamped, "/hybrid_controller/update_target", 10)
        self._pub_servo_twist = self.create_publisher(
            TwistStamped, "/servo_node/delta_twist_cmds", 10)

        # ── Debug image publisher (perception overlay with ROI) ──
        if _HAS_CV:
            self._pub_debug_img = self.create_publisher(
                Image, "/assistive_lift_v4/debug_image", 10)

        # ── Camera subscribers ──
        if _HAS_CV:
            self.create_subscription(
                CameraInfo, self._camera_info_topic, self._cam_info_cb, 10)
            self.create_subscription(
                Image, self._rgb_topic, self._rgb_cb, 10)
            self.create_subscription(
                Image, self._depth_topic, self._depth_cb, 10)

        # ── Observation subscribers ──
        self.create_subscription(
            Bool, "/hybrid_controller/enabled", self._enabled_cb, _latched_qos)
        self.create_subscription(
            String, "/interaction_state", self._interaction_state_cb, 10)
        self.create_subscription(
            Bool, "/interaction/is_nudge", self._nudge_cb, 10)

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(
            WrenchStamped, "/wrench_zeroed", self._wrench_cb, sensor_qos)
        self.create_subscription(
            TwistStamped, "/servo_node/delta_twist_cmds",
            self._twist_obs_cb, sensor_qos)

        # ── Joint state subscriber (for IK seed) ──
        self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 10)

        # ── Service clients ──
        self._pause_cli = self.create_client(
            Trigger, "/servo_node/pause_servo")
        self._unpause_cli = self.create_client(
            Trigger, "/servo_node/unpause_servo")
        self._switch_cli = self.create_client(
            SwitchController, "/controller_manager/switch_controller")
        self._tare_cli = self.create_client(Trigger, "/ft_zeroer/tare")

        # ── FSM timer 20 Hz ──
        self._timer = self.create_timer(0.05, self._tick)

        self.get_logger().info("AssistiveLiftV4Node started.")

    # ==================================================================
    # Camera callbacks
    # ==================================================================
    def _cam_info_cb(self, msg: CameraInfo):
        self._K = np.array(msg.k).reshape(3, 3)

    def _rgb_cb(self, msg: Image):
        try:
            self._rgb = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def _depth_cb(self, msg: Image):
        try:
            self._depth = self._bridge.imgmsg_to_cv2(msg, "passthrough")
        except Exception:
            pass

    # ==================================================================
    # Observation callbacks
    # ==================================================================
    def _enabled_cb(self, msg: Bool):
        self._hybrid_enabled = msg.data

    def _interaction_state_cb(self, msg: String):
        old = self._interaction_state
        new = msg.data
        if old != new:
            self._prev_interaction_state = old
            self._interaction_state = new
            if self._phase == Phase.AWAIT_INTERACTION:
                now_s = self.get_clock().now().nanoseconds / 1e9
                self._state_log.append((now_s, old, new))

    def _nudge_cb(self, msg: Bool):
        if not msg.data or self._phase != Phase.AWAIT_INTERACTION:
            return
        now_s = self.get_clock().now().nanoseconds / 1e9
        if self._first_tap_time is None:
            # First tap
            self._first_tap_time = now_s
            self.get_logger().info("First tap detected — waiting for second tap...")
        else:
            elapsed = now_s - self._first_tap_time
            if elapsed <= DOUBLE_TAP_WINDOW_S:
                # Second tap within window → double tap!
                self._double_tap_received = True
                self._double_tap_time_ms = elapsed * 1000.0
                self.get_logger().info(
                    f"Double tap detected! ({elapsed:.2f}s between taps)")
            else:
                # Too slow — treat this as a new first tap
                self._first_tap_time = now_s
                self.get_logger().info(
                    "Tap too late — resetting. Waiting for second tap...")

    def _wrench_cb(self, msg: WrenchStamped):
        if self._phase != Phase.AWAIT_INTERACTION:
            return
        if self._interaction_state != "GUIDING":
            return
        t = self.get_clock().now().nanoseconds / 1e9
        fx, fy, fz = msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z
        self._force_samples.append((t, fx, fy, fz))
        f_mag = math.sqrt(fx * fx + fy * fy + fz * fz)
        if f_mag > FORCE_THRESHOLD_N and self._force_above_thresh_time is None:
            self._force_above_thresh_time = t

    def _twist_obs_cb(self, msg: TwistStamped):
        if self._phase != Phase.AWAIT_INTERACTION:
            return
        if self._interaction_state != "GUIDING":
            return
        t = self.get_clock().now().nanoseconds / 1e9
        vx, vy, vz = msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z
        self._velocity_samples.append((t, vx, vy, vz))
        v_mag = math.sqrt(vx * vx + vy * vy + vz * vz)
        if v_mag > VELOCITY_THRESHOLD_MPS and self._velocity_above_thresh_time is None:
            self._velocity_above_thresh_time = t

    def _joint_state_cb(self, msg: JointState):
        joint_map = dict(zip(msg.name, msg.position))
        try:
            self._current_joints = [joint_map[n] for n in JOINT_NAMES]
        except KeyError:
            pass

    # ==================================================================
    # FSM tick
    # ==================================================================
    def _tick(self):
        p = self._phase
        if p == Phase.INIT:
            self._do_init()
        elif p == Phase.PERCEIVE:
            self._do_perceive()
        elif p == Phase.PLAN_GRASP:
            self._do_plan_grasp()
        elif p == Phase.MOVE_TO_PREGRASP:
            self._wait_move_result()
        elif p == Phase.MOVE_TO_GRASP:
            self._wait_move_result()
        elif p == Phase.GRASP:
            self._do_grasp()
        elif p == Phase.LIFT:
            self._wait_move_result()
        elif p == Phase.ENABLE_SERVO:
            self._do_enable_servo()
        elif p == Phase.AWAIT_INTERACTION:
            self._do_await_interaction()
        elif p == Phase.PREP_PLACE:
            self._do_prep_place()
        elif p == Phase.DESCEND_TO_PLACE:
            self._wait_move_result()
        elif p == Phase.PLACE:
            self._do_place()
        elif p == Phase.RETRACT:
            self._do_retract()
        elif p == Phase.DONE:
            self._do_done()

    # ==================================================================
    # Phase helpers
    # ==================================================================
    def _set_phase(self, new_phase: str):
        now = self.get_clock().now()
        elapsed = (now - self._phase_start).nanoseconds / 1e9
        self.get_logger().info(
            f"[{self._phase}] -> [{new_phase}] ({elapsed:.2f}s)")
        self._timestamps[self._phase] = elapsed
        self._phase = new_phase
        self._phase_start = now
        if new_phase == Phase.DONE:
            self._log_summary()

    def _elapsed(self) -> float:
        return (self.get_clock().now() - self._phase_start).nanoseconds / 1e9

    # ==================================================================
    # INIT — wait for homing, tare FT sensor
    # ==================================================================
    def _do_init(self):
        if not self._mg_server_ready:
            if self._move_client.server_is_ready():
                self._mg_server_ready = True
                self.get_logger().info("MoveGroup action server ready.")
            else:
                return

        if not self._switch_cli.service_is_ready():
            return

        if not self._hybrid_enabled:
            self.get_logger().info(
                "Waiting for homing (hybrid_controller/enabled)...",
                throttle_duration_sec=5.0)
            return

        if not self._tare_done:
            if self._tare_cli.service_is_ready():
                self.get_logger().info("Calling /ft_zeroer/tare ...")
                future = self._tare_cli.call_async(Trigger.Request())
                future.add_done_callback(self._on_tare_done)
                self._tare_done = True
            else:
                self.get_logger().info(
                    "Waiting for /ft_zeroer/tare service...",
                    throttle_duration_sec=5.0)
            return

        if self._elapsed() < 2.0:
            return

        msg = Bool()
        msg.data = False
        self._pub_enabled.publish(msg)
        self._call_pause(True)
        self._set_phase(Phase.PERCEIVE)

    def _on_tare_done(self, future):
        try:
            result = future.result()
            self.get_logger().info(f"Tare result: {result.message}")
        except Exception as e:
            self.get_logger().warn(f"Tare call failed: {e}")
        self._phase_start = self.get_clock().now()

    # ==================================================================
    # PERCEIVE — Depth-only (RANSAC table plane + object segmentation)
    # ==================================================================
    def _do_perceive(self):
        if not _HAS_CV:
            self.get_logger().error(
                "cv2/cv_bridge not available — cannot perceive.")
            self._set_phase(Phase.DONE)
            return

        if self._depth is None or self._K is None or self._rgb is None:
            self.get_logger().info(
                "Waiting for camera data...", throttle_duration_sec=3.0)
            return

        # Throttle perception to ~2 Hz
        expected_time = 0.5 * self._perception_attempts + 2.0
        if self._elapsed() < expected_time:
            return

        self._perception_attempts += 1

        # ── HSV red mask + depth → 3D position in base_link ──
        result = self._process_hsv_depth()
        if result is None:
            if self._perception_attempts >= MAX_PERCEPTION_ATTEMPTS:
                self.get_logger().error(
                    f"No red object detected after {MAX_PERCEPTION_ATTEMPTS} "
                    f"attempts. Falling back to depth-only.")
                # Last resort: try depth-only
                result = self._process_depth_full()
                if result is None:
                    self._set_phase(Phase.DONE)
                    return
            else:
                self.get_logger().info(
                    f"No red object detected (attempt "
                    f"{self._perception_attempts}).",
                    throttle_duration_sec=2.0)
                return

        self._box_center_bl = result["center"]
        self._box_dims = result["dims"]
        self._box_yaw = result["yaw"]
        self._table_z = result["table_z"]
        self._perception_done = True

        cx, cy, cz = self._box_center_bl
        lx, ly, lz = self._box_dims
        self.get_logger().info(
            f"Object detected: "
            f"pose=({cx:.3f}, {cy:.3f}, {cz:.3f}, yaw={self._box_yaw:.1f}°), "
            f"size=({lx*100:.1f}x{ly*100:.1f}x{lz*100:.1f} cm), "
            f"table_z={self._table_z:.3f}m")

        self._set_phase(Phase.PLAN_GRASP)

    # ==================================================================
    # HSV + Depth perception (primary method for colored objects)
    # ==================================================================
    def _process_hsv_depth(self):
        """Detect red object via HSV color mask, then use aligned depth
        to compute 3D position and dimensions in base_link.

        Returns: {center, dims, yaw, table_z} or None.
        """
        ih, iw = self._rgb.shape[:2]
        fx, fy = self._K[0, 0], self._K[1, 1]
        cx_k, cy_k = self._K[0, 2], self._K[1, 2]

        # ── HSV red mask ──
        hsv = cv2.cvtColor(self._rgb, cv2.COLOR_BGR2HSV)
        red_mask = (
            cv2.inRange(hsv, np.array([0, 120, 80]), np.array([10, 255, 255]))
            | cv2.inRange(hsv, np.array([170, 120, 80]), np.array([180, 255, 255]))
        )

        # Morphological cleanup
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kern)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kern)

        # Find contours — pick largest red blob
        contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.get_logger().info("No red pixels found.")
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 100:  # too small
            self.get_logger().info(f"Red blob too small: {area} px")
            return None

        # Bounding rect and mask for the object
        x1, y1, w, h = cv2.boundingRect(largest)
        x2, y2 = x1 + w, y1 + h

        # Create a mask for just this contour
        obj_mask = np.zeros((ih, iw), dtype=np.uint8)
        cv2.drawContours(obj_mask, [largest], -1, 255, -1)

        # ── Get depth at red pixels ──
        depth_m = self._depth.astype(np.float64)
        if self._depth.dtype == np.uint16:
            depth_m /= 1000.0

        # Pixels inside the red contour with valid depth
        obj_pixels = np.where(obj_mask > 0)
        vs_obj = obj_pixels[0]
        us_obj = obj_pixels[1]
        ds_obj = depth_m[vs_obj, us_obj]

        valid = (ds_obj > 0.1) & (ds_obj < 2.0) & np.isfinite(ds_obj)
        if np.sum(valid) < 10:
            self.get_logger().info("Not enough valid depth at red pixels.")
            return None

        us_v = us_obj[valid].astype(np.float64)
        vs_v = vs_obj[valid].astype(np.float64)
        ds_v = ds_obj[valid]

        # ── Back-project to 3D in camera optical frame ──
        x_cam = (us_v - cx_k) * ds_v / fx
        y_cam = (vs_v - cy_k) * ds_v / fy
        pts_cam = np.column_stack([x_cam, y_cam, ds_v])

        # ── Remove outliers (median depth ± 5cm) ──
        med_depth = np.median(ds_v)
        inlier = np.abs(ds_v - med_depth) < 0.05
        if np.sum(inlier) < 10:
            return None
        pts_cam = pts_cam[inlier]

        # ── Dimensions from camera-frame 3D points ──
        lx, ly = DEFAULT_BOX_DIMS[0], DEFAULT_BOX_DIMS[1]
        yaw_deg = 0.0
        if len(pts_cam) >= 5:
            xy_cam = pts_cam[:, :2]
            pts_um = (xy_cam * 1e6).astype(np.int32).reshape(-1, 1, 2)
            rect = cv2.minAreaRect(pts_um)
            d1 = rect[1][0] / 1e6
            d2 = rect[1][1] / 1e6
            yaw_deg = float(rect[2])
            if 0.005 < d1 < 0.30 and 0.005 < d2 < 0.30:
                lx, ly = max(d1, d2), min(d1, d2)
                if d1 < d2:
                    yaw_deg += 90.0

        # Height: use depth spread along the view direction (rough)
        # Better: estimate from bbox height in pixels * depth / fy
        obj_height = DEFAULT_BOX_DIMS[2]  # fallback
        depth_range = np.percentile(ds_v[inlier], 5) - np.percentile(ds_v[inlier], 95)
        if abs(depth_range) > 0.01:
            obj_height = abs(depth_range)
        # Also try bbox pixel height
        pixel_h = y2 - y1
        estimated_h = pixel_h * med_depth / fy
        if 0.01 < estimated_h < 0.15:
            obj_height = max(obj_height, estimated_h * 0.7)  # top-view underestimates

        if obj_height < 0.01:
            obj_height = DEFAULT_BOX_DIMS[2]

        self.get_logger().info(
            f"HSV red: {np.sum(inlier)} pts, "
            f"dims={lx*100:.1f}x{ly*100:.1f}x{obj_height*100:.1f}cm, "
            f"depth={med_depth:.3f}m")

        # ── Transform centroid to base_link ──
        centroid_cam = np.mean(pts_cam, axis=0)
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                BASE_FRAME, self._camera_optical_frame, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return None

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        R = _quat_to_rot(q.x, q.y, q.z, q.w)
        t_vec = np.array([t.x, t.y, t.z])
        centroid_bl = R @ centroid_cam + t_vec

        # Table Z: estimate from depth below the object
        # The bottom of the object is roughly at centroid_z - obj_height/2
        # Table surface is just below that
        table_z = float(centroid_bl[2]) - obj_height / 2.0
        center_z = float(centroid_bl[2])
        center = [float(centroid_bl[0]), float(centroid_bl[1]), center_z]

        in_roi = (WS_X_MIN <= center[0] <= WS_X_MAX and
                  WS_Y_MIN <= center[1] <= WS_Y_MAX and
                  WS_Z_MIN <= center[2] <= WS_Z_MAX)

        self.get_logger().info(
            f"Red object in base_link: ({center[0]:.3f}, {center[1]:.3f}, "
            f"{center[2]:.3f}), table_z≈{table_z:.3f}m"
            f"{'' if in_roi else ' [OUTSIDE ROI]'}")

        # ── Always publish debug image (even for rejected detections) ──
        try:
            dbg = self._rgb.copy()

            # ── Project workspace ROI onto image ──
            R_inv = R.T
            t_inv = -R_inv @ t_vec
            roi_corners_bl = [
                [WS_X_MIN, WS_Y_MIN, WS_Z_MIN],
                [WS_X_MAX, WS_Y_MIN, WS_Z_MIN],
                [WS_X_MAX, WS_Y_MAX, WS_Z_MIN],
                [WS_X_MIN, WS_Y_MAX, WS_Z_MIN],
                [WS_X_MIN, WS_Y_MIN, WS_Z_MAX],
                [WS_X_MAX, WS_Y_MIN, WS_Z_MAX],
                [WS_X_MAX, WS_Y_MAX, WS_Z_MAX],
                [WS_X_MIN, WS_Y_MAX, WS_Z_MAX],
            ]
            roi_px = {}  # index -> (u, v)
            h, w = dbg.shape[:2]
            for idx, pt_bl in enumerate(roi_corners_bl):
                pt_cam = R_inv @ np.array(pt_bl) + t_inv
                if pt_cam[2] > 0.01:
                    u = int(fx * pt_cam[0] / pt_cam[2] + cx_k)
                    v = int(fy * pt_cam[1] / pt_cam[2] + cy_k)
                    # Clamp to image bounds for drawing
                    u_c = max(-2000, min(u, w + 2000))
                    v_c = max(-2000, min(v, h + 2000))
                    roi_px[idx] = (u_c, v_c)

            # Draw whatever edges are visible (both endpoints must exist)
            # Bottom face edges: 0-1, 1-2, 2-3, 3-0
            # Top face edges: 4-5, 5-6, 6-7, 7-4
            # Vertical edges: 0-4, 1-5, 2-6, 3-7
            roi_edges = [(0,1),(1,2),(2,3),(3,0),
                         (4,5),(5,6),(6,7),(7,4),
                         (0,4),(1,5),(2,6),(3,7)]
            for a, b in roi_edges:
                if a in roi_px and b in roi_px:
                    thickness = 2 if a < 4 and b < 4 else 1
                    cv2.line(dbg, roi_px[a], roi_px[b],
                             (255, 255, 0), thickness)
            if 0 in roi_px:
                cv2.putText(dbg, "ROI", roi_px[0],
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            self.get_logger().debug(
                f"ROI projection: {len(roi_px)}/8 corners visible")

            # ── Draw detection (green=in ROI, red=outside) ──
            det_color = (0, 255, 0) if in_roi else (0, 0, 255)
            cv2.drawContours(dbg, [largest], -1, det_color, 2)
            cv2.rectangle(dbg, (x1, y1), (x2, y2), det_color, 2)
            label = (f"{lx*100:.0f}x{ly*100:.0f}x{obj_height*100:.0f}cm "
                     f"d={med_depth:.2f}m")
            if not in_roi:
                label += " [OUTSIDE ROI]"
            cv2.putText(dbg, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, det_color, 1)

            cv2.imwrite("/tmp/hsv_detect_debug.png", dbg)
            self._pub_debug_img.publish(
                self._bridge.cv2_to_imgmsg(dbg, "bgr8"))
        except Exception as e:
            self.get_logger().warn(f"Debug image error: {e}")

        # ── Reject if outside workspace table ROI ──
        if not in_roi:
            self.get_logger().warn(
                f"Object at ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) "
                f"is OUTSIDE workspace ROI "
                f"[x:{WS_X_MIN}..{WS_X_MAX}, y:{WS_Y_MIN}..{WS_Y_MAX}, "
                f"z:{WS_Z_MIN}..{WS_Z_MAX}] — rejected.")
            return None

        return {
            "center": center,
            "dims": [lx, ly, obj_height],
            "yaw": yaw_deg,
            "table_z": table_z,
        }

    # ── YOLO detection (box identification + rough region) ────────────
    def _detect_yolo_box(self):
        """Use YOLO to find which object is a box. Returns (x1,y1,x2,y2)."""
        if self._yolo is None:
            return self._detect_hsv_fallback()

        results = self._yolo(self._rgb, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return self._detect_hsv_fallback()

        ih, iw = self._rgb.shape[:2]
        best_bbox = None
        best_conf = 0.0

        best_cls = -1
        for box in results[0].boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            if conf < YOLO_CONF_THRESH:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(iw, x2), min(ih, y2)
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue

            cls_name = self._yolo.names.get(cls_id, str(cls_id))
            self.get_logger().info(
                f"  YOLO det: cls={cls_name}({cls_id}), "
                f"conf={conf:.2f}, bbox=({x1},{y1},{x2},{y2})")

            if conf > best_conf:
                best_bbox = (x1, y1, x2, y2)
                best_conf = conf
                best_cls = cls_id

        if best_bbox is not None:
            cls_name = self._yolo.names.get(best_cls, str(best_cls))
            self.get_logger().info(
                f"YOLO detection: cls={cls_name}, conf={best_conf:.2f}, "
                f"bbox={best_bbox}")

            # Save annotated debug image
            try:
                dbg = self._rgb.copy()
                cv2.rectangle(dbg,
                              (best_bbox[0], best_bbox[1]),
                              (best_bbox[2], best_bbox[3]),
                              (0, 255, 0), 2)
                cv2.putText(dbg, f"{cls_name} {best_conf:.2f}",
                            (best_bbox[0], max(15, best_bbox[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imwrite("/tmp/yolo_debug.png", dbg)
                self.get_logger().info("Debug image saved to /tmp/yolo_debug.png")
            except Exception:
                pass
        else:
            return self._detect_hsv_fallback()
        return best_bbox

    def _detect_hsv_fallback(self):
        """HSV red-colour fallback when YOLO misses."""
        hsv = cv2.cvtColor(self._rgb, cv2.COLOR_BGR2HSV)
        red_mask = (
            cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
            | cv2.inRange(hsv, np.array([170, 80, 80]),
                          np.array([180, 255, 255]))
        )
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kern)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kern)
        contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 50:
            return None
        x, y, w, h = cv2.boundingRect(largest)
        return (x, y, x + w, y + h)

    # ── Depth processing: camera-frame for accuracy ─────────────────
    def _process_depth_full(self):
        """
        Depth-only perception — all geometry in CAMERA OPTICAL FRAME
        for accurate dimensions (intrinsics + depth are correct).
        Only the centroid is transformed to base_link at the end.

        Steps:
          1. Back-project depth to 3D in camera frame
          2. RANSAC table plane in camera frame
          3. Object = points 1-15cm above table plane
          4. 2D mask → largest connected component → object pixels
          5. Dims from camera-frame 3D points (accurate)
          6. Transform centroid to base_link
        Returns: {center, dims, yaw, table_z} or None.
        """
        ih, iw = self._depth.shape[:2]
        fx, fy = self._K[0, 0], self._K[1, 1]
        cx_k, cy_k = self._K[0, 2], self._K[1, 2]

        # ── Build full-image depth in metres ──
        depth_m = self._depth.astype(np.float64)
        if self._depth.dtype == np.uint16:
            depth_m /= 1000.0

        # ── Sample every 2nd pixel (finer than before for small objects) ──
        step = 2
        vs, us = np.mgrid[0:ih:step, 0:iw:step]
        vs_f, us_f = vs.ravel(), us.ravel()
        ds = depth_m[vs_f, us_f]

        valid = (ds > 0.1) & (ds < 1.5) & np.isfinite(ds)
        if np.sum(valid) < 100:
            return None

        us_v = us_f[valid].astype(np.float64)
        vs_v = vs_f[valid].astype(np.float64)
        ds_v = ds[valid]

        # ── Back-project to 3D in camera optical frame ──
        x_cam = (us_v - cx_k) * ds_v / fx
        y_cam = (vs_v - cy_k) * ds_v / fy
        pts_cam = np.column_stack([x_cam, y_cam, ds_v])

        # ── Depth range filter: keep points near the table distance ──
        # Table is roughly 0.4-1.2m from camera — filter out background
        depth_mask = (ds_v > 0.3) & (ds_v < 1.2)
        if np.sum(depth_mask) < 100:
            return None
        pts_cam = pts_cam[depth_mask]
        us_v = us_v[depth_mask]
        vs_v = vs_v[depth_mask]

        # ── RANSAC table plane in camera frame ──
        plane = _ransac_plane(pts_cam)
        if plane is None:
            self.get_logger().warn("RANSAC plane fit failed.")
            return None

        normal, d_plane = plane
        signed_dists = pts_cam @ normal + d_plane

        # Ensure normal points AWAY from camera (toward table surface)
        # In camera optical frame, Z points forward. Table normal should
        # have a positive component pointing "up" from the table.
        # If most table inliers are in front of the plane, flip.
        table_inliers = np.abs(signed_dists) < RANSAC_THRESHOLD_M * 2
        n_table = np.sum(table_inliers)

        if n_table < 50:
            self.get_logger().info("Not enough table inliers.")
            return None

        # ── Object points: above table by 1cm to 15cm ──
        above = ((signed_dists > BOX_MIN_HEIGHT_M) &
                 (signed_dists < MAX_OBJECT_HEIGHT_M))
        n_above = np.sum(above)

        self.get_logger().info(
            f"Depth (cam frame): {len(pts_cam)} pts, "
            f"{n_table} table inliers, {n_above} above-table pts")

        if n_above < 5:
            self.get_logger().info("No object points above table.")
            return None

        # ── Build 2D mask of "above table" pixels → find object blob ──
        obj_mask_2d = np.zeros((ih, iw), dtype=np.uint8)
        obj_us = us_v[above].astype(int)
        obj_vs = vs_v[above].astype(int)
        obj_us = np.clip(obj_us, 0, iw - 1)
        obj_vs = np.clip(obj_vs, 0, ih - 1)
        obj_mask_2d[obj_vs, obj_us] = 255

        # Dilate to connect nearby pixels, then find contours
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        obj_mask_2d = cv2.dilate(obj_mask_2d, kern, iterations=2)
        obj_mask_2d = cv2.erode(obj_mask_2d, kern, iterations=1)

        contours, _ = cv2.findContours(
            obj_mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.get_logger().info("No object contours found.")
            return None

        # Find the contour that is most likely the box:
        # - Must have reasonable area (not too small, not too large)
        # - Pick the one with the most "above table" points inside
        best_contour = None
        best_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:  # too small (noise)
                continue
            # Count how many original above-table pixels fall inside
            count = 0
            for u, v in zip(obj_us, obj_vs):
                if cv2.pointPolygonTest(cnt, (float(u), float(v)), False) >= 0:
                    count += 1
            if count > best_count:
                best_count = count
                best_contour = cnt

        if best_contour is None or best_count < 3:
            self.get_logger().info("No valid object cluster found.")
            return None

        # ── Get 3D points for the best cluster ──
        cluster_mask = np.zeros(n_above, dtype=bool)
        for k in range(n_above):
            u, v = int(obj_us[k]), int(obj_vs[k])
            if cv2.pointPolygonTest(
                    best_contour, (float(u), float(v)), False) >= 0:
                cluster_mask[k] = True

        if np.sum(cluster_mask) < 3:
            return None

        # Get the camera-frame 3D points and distances for the cluster
        obj_pts_cam = pts_cam[above][cluster_mask]
        obj_dists_cluster = signed_dists[above][cluster_mask]

        # ── Height (95th percentile of distance above table) ──
        obj_height = float(np.percentile(obj_dists_cluster, 95))
        if obj_height < 0.01:
            obj_height = DEFAULT_BOX_DIMS[2]

        # ── Dimensions from camera-frame 3D points ──
        # Use minAreaRect on XY projection in camera frame
        # In camera optical frame: X=right, Y=down, Z=forward
        # The "XY" of the box on the table maps to camera X and Y
        lx, ly = DEFAULT_BOX_DIMS[0], DEFAULT_BOX_DIMS[1]
        yaw_deg = 0.0
        if len(obj_pts_cam) >= 5:
            xy_cam = obj_pts_cam[:, :2]  # camera X, Y
            pts_um = (xy_cam * 1e6).astype(np.int32).reshape(-1, 1, 2)
            rect = cv2.minAreaRect(pts_um)
            d1 = rect[1][0] / 1e6
            d2 = rect[1][1] / 1e6
            yaw_deg = float(rect[2])
            if 0.005 < d1 < 0.30 and 0.005 < d2 < 0.30:
                lx, ly = max(d1, d2), min(d1, d2)
                if d1 < d2:
                    yaw_deg += 90.0

        self.get_logger().info(
            f"Object (cam frame): {np.sum(cluster_mask)} pts, "
            f"dims={lx*100:.1f}x{ly*100:.1f}x{obj_height*100:.1f}cm")

        # ── Transform centroid to base_link ──
        centroid_cam = np.mean(obj_pts_cam, axis=0)
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                BASE_FRAME, self._camera_optical_frame, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return None

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        R = _quat_to_rot(q.x, q.y, q.z, q.w)
        t_vec = np.array([t.x, t.y, t.z])
        centroid_bl = R @ centroid_cam + t_vec

        # Table Z in base_link (from table inlier centroids)
        table_pts_cam = pts_cam[table_inliers]
        table_centroid_cam = np.mean(table_pts_cam, axis=0)
        table_centroid_bl = R @ table_centroid_cam + t_vec
        table_z = float(table_centroid_bl[2])

        center_z = table_z + obj_height / 2.0
        center = [float(centroid_bl[0]), float(centroid_bl[1]), float(center_z)]

        self.get_logger().info(
            f"Object in base_link: pos=({center[0]:.3f}, {center[1]:.3f}, "
            f"{center[2]:.3f}), table_z={table_z:.3f}m")

        return {
            "center": center,
            "dims": [lx, ly, obj_height],
            "yaw": yaw_deg,
            "table_z": table_z,
        }

    # ==================================================================
    # PLAN_GRASP — compute IK, add collision
    # ==================================================================
    def _do_plan_grasp(self):
        if not self._plan_grasp_started:
            self._plan_grasp_started = True

            bx, by, bz = self._box_center_bl
            bw, bd, bh = self._box_dims

            dist_xy = math.sqrt(bx * bx + by * by)
            use_fallback = False

            if dist_xy > _UR3E_MAX_REACH:
                self.get_logger().warn(
                    f"Box at XY={dist_xy:.3f}m — beyond IK reach "
                    f"(camera TF may be off). Using fallback grasp joints.")
                use_fallback = True

            # Compute grasp position: fingertips at the geometric centre
            # of the object (mid-height).  MIN_TABLE_CLEARANCE prevents
            # fingers from touching the table on very small objects.
            grasp_fingertip_z = max(
                self._table_z + bh / 2.0,
                self._table_z + MIN_TABLE_CLEARANCE)
            grasp_tool0_z = grasp_fingertip_z + self._tool0_to_fingertip
            self._grasp_tool0_target = [bx, by, grasp_tool0_z]
            self.get_logger().info(
                f"Grasp tool0 target: ({self._grasp_tool0_target[0]:.3f}, "
                f"{self._grasp_tool0_target[1]:.3f}, "
                f"{self._grasp_tool0_target[2]:.3f}), "
                f"fingertip_z={grasp_fingertip_z:.3f}, "
                f"table_z={self._table_z:.3f}, obj_h={bh*100:.1f}cm")

            gripped = min(bw, bd)
            fj = _compute_finger_joint(gripped)
            self._gripper_cmd = (
                [fj] if self._hardware_gripper else _make_gripper_cmd(fj))
            self.get_logger().info(
                f"Gripper: gripped_width={gripped*100:.1f}cm, "
                f"finger_joint={fj:.3f}")

            self._ik_solved = True

            # Don't add box collision — camera TF is inaccurate, wrong position
            # Workspace table stays in planning scene for collision avoidance

            # Reset adaptive pre-grasp offset state
            self._pregrasp_offsets_to_try = [PREGRASP_OFFSET, 0.05, 0.03]
            self._pregrasp_offset_idx = 0

            # Disable hybrid, pause Servo, switch to JTC for MoveGroup
            msg = Bool()
            msg.data = False
            self._pub_enabled.publish(msg)
            self._call_pause(True)
            self._switch_done = False
            self._switch_controllers(
                activate=[self._jtc_name],
                deactivate=["forward_velocity_controller"],
                done_cb=self._on_switch_done)
            return

        if not self._switch_done:
            return

        if self._elapsed() < 1.0:
            return

        # Step 1: Pre-solve IK with current joints as seed.
        # Pilz's internal IK has no seed and fails at workspace edges.
        # MoveIt /compute_ik seeded with current joints finds nearest solution.
        if self._ik_future is None and self._pregrasp_ik_joints is None:
            # Remove workspace table collision — arm links pass through
            # the table box when reaching over to grasp.  Re-added after lift.
            self._remove_collision_object("workspace_table")

            # Pre-grasp = grasp tool0 position + offset in Z.
            # Try progressively shorter offsets if IK fails (near reach limit).
            current_offset = self._pregrasp_offsets_to_try[self._pregrasp_offset_idx]
            pregrasp_pos = np.array(self._grasp_tool0_target, dtype=float)
            pregrasp_pos[2] += current_offset

            self.get_logger().info(
                f"Pre-grasp target: ({pregrasp_pos[0]:.3f}, "
                f"{pregrasp_pos[1]:.3f}, {pregrasp_pos[2]:.3f})")

            self._ik_future = self._solve_ik_moveit(
                pregrasp_pos.tolist(),
                target_orientation_quat=[0.0, 1.0, 0.0, 0.0])
            if self._ik_future is None:
                self.get_logger().error("IK service not ready — cannot plan pre-grasp")
                self._move_result = False
                self._set_phase(Phase.MOVE_TO_PREGRASP)
            return

        # Step 2: Wait for IK result.
        if self._ik_future is not None and not self._ik_future.done():
            return

        # Step 3: Extract joints or retry with shorter offset.
        if self._ik_future is not None:
            resp = self._ik_future.result()
            self._ik_future = None
            current_offset = self._pregrasp_offsets_to_try[self._pregrasp_offset_idx]
            if resp.error_code.val == 1:  # SUCCESS
                js = resp.solution.joint_state
                joints = []
                for jn in JOINT_NAMES:
                    idx = list(js.name).index(jn)
                    joints.append(js.position[idx])
                self._pregrasp_ik_joints = joints
                self._descent_distance = current_offset
                self.get_logger().info(
                    f"IK solved (offset={current_offset:.2f}m): "
                    f"[{', '.join(f'{j:.3f}' for j in joints)}]")
            else:
                # Try next shorter offset
                self._pregrasp_offset_idx += 1
                if self._pregrasp_offset_idx < len(self._pregrasp_offsets_to_try):
                    next_off = self._pregrasp_offsets_to_try[self._pregrasp_offset_idx]
                    self.get_logger().warn(
                        f"IK failed with offset={current_offset:.2f}m "
                        f"— retrying with {next_off:.2f}m")
                    return  # will retry on next tick
                else:
                    self.get_logger().error(
                        f"IK failed (code={resp.error_code.val}) at all offsets — "
                        "pose unreachable")
                    self._move_result = False
                    self._set_phase(Phase.MOVE_TO_PREGRASP)
                    return

        # Step 4: Send Pilz PTP with pre-solved joint values.
        self._set_phase(Phase.MOVE_TO_PREGRASP)
        self._send_pilz_ptp_joint_goal(
            self._pregrasp_ik_joints, vel_scale=0.3, acc_scale=0.3)

    # ==================================================================
    # MoveGroup helpers
    # ==================================================================
    def _send_joint_goal(self, joint_values):
        goal = self._build_joint_mg_goal(joint_values)
        self._move_result = None
        self._goal_handle = None
        self._move_client.send_goal_async(goal).add_done_callback(
            self._mg_goal_response)

    def _send_pose_goal(self, position, orientation_quat=None,
                        pos_tolerance=0.01, ori_tolerance=0.2):
        """Send MoveGroup goal with Cartesian position + orientation.
        MoveGroup's IK solver finds the joint configuration.
        position: [x, y, z] in base_link
        orientation_quat: [x, y, z, w] or None for tool-down default
        pos_tolerance: radius of goal sphere in metres (default 1cm)
        ori_tolerance: X/Y orientation tolerance in radians (default 0.2)
        """
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = PLANNING_GROUP
        req.pipeline_id = "ompl"
        req.planner_id = "RRTConnectkConfigDefault"
        req.num_planning_attempts = 20
        req.allowed_planning_time = 15.0
        req.max_velocity_scaling_factor = 0.3
        req.max_acceleration_scaling_factor = 0.3

        # Position constraint
        pc = PositionConstraint()
        pc.header.frame_id = BASE_FRAME
        pc.link_name = EE_LINK
        pc.weight = 1.0
        bv = BoundingVolume()
        sp = SolidPrimitive()
        sp.type = SolidPrimitive.SPHERE
        sp.dimensions = [pos_tolerance]
        bv.primitives = [sp]
        p = Pose()
        p.position.x = float(position[0])
        p.position.y = float(position[1])
        p.position.z = float(position[2])
        p.orientation.w = 1.0
        bv.primitive_poses = [p]
        pc.constraint_region = bv

        # Orientation constraint (tool pointing down)
        oc = OrientationConstraint()
        oc.header.frame_id = BASE_FRAME
        oc.link_name = EE_LINK
        oc.weight = 1.0
        if orientation_quat is None:
            # Tool pointing down: 180° rotation around Y
            oc.orientation.x = 0.0
            oc.orientation.y = 1.0
            oc.orientation.z = 0.0
            oc.orientation.w = 0.0
        else:
            oc.orientation.x = float(orientation_quat[0])
            oc.orientation.y = float(orientation_quat[1])
            oc.orientation.z = float(orientation_quat[2])
            oc.orientation.w = float(orientation_quat[3])
        oc.absolute_x_axis_tolerance = ori_tolerance
        oc.absolute_y_axis_tolerance = ori_tolerance
        oc.absolute_z_axis_tolerance = ori_tolerance

        constraints = Constraints()
        constraints.position_constraints = [pc]
        constraints.orientation_constraints = [oc]

        # Constrain shoulder_pan_joint to stay near its current value.
        # Prevents OMPL from finding paths with 360° base rotation.
        if self._current_joints is not None:
            jc = JointConstraint()
            jc.joint_name = "shoulder_pan_joint"
            jc.position = self._current_joints[0]
            jc.tolerance_above = 1.57   # ±90° from current
            jc.tolerance_below = 1.57
            jc.weight = 1.0
            constraints.joint_constraints = [jc]

        req.goal_constraints = [constraints]

        req.start_state = RobotState()
        req.start_state.is_diff = True

        goal.request = req
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 5

        self._move_result = None
        self._goal_handle = None
        self._move_client.send_goal_async(goal).add_done_callback(
            self._mg_goal_response)

        self.get_logger().info(
            f"MoveGroup pose goal: ({position[0]:.3f}, {position[1]:.3f}, "
            f"{position[2]:.3f})")

    # ── Pilz Industrial Motion Planner methods ────────────────────
    def _send_pilz_ptp_pose_goal(self, position, target_orientation_quat=None,
                                  vel_scale=0.3, acc_scale=0.3):
        """Pilz PTP with Cartesian pose constraints.
        Pilz handles IK internally — orientation is properly constrained."""
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = PLANNING_GROUP
        req.pipeline_id = "pilz"
        req.planner_id = "PTP"
        req.num_planning_attempts = 1
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = vel_scale
        req.max_acceleration_scaling_factor = acc_scale
        # Position constraint
        pc = PositionConstraint()
        pc.header.frame_id = BASE_FRAME
        pc.link_name = EE_LINK
        pc.weight = 1.0
        bv = BoundingVolume()
        sp = SolidPrimitive()
        sp.type = SolidPrimitive.SPHERE
        sp.dimensions = [0.01]
        bv.primitives = [sp]
        p = Pose()
        p.position.x = float(position[0])
        p.position.y = float(position[1])
        p.position.z = float(position[2])
        p.orientation.w = 1.0
        bv.primitive_poses = [p]
        pc.constraint_region = bv
        # Orientation constraint
        oc = OrientationConstraint()
        oc.header.frame_id = BASE_FRAME
        oc.link_name = EE_LINK
        oc.weight = 1.0
        if target_orientation_quat is None:
            oc.orientation.y = 1.0  # tool-down
        else:
            oc.orientation.x = float(target_orientation_quat[0])
            oc.orientation.y = float(target_orientation_quat[1])
            oc.orientation.z = float(target_orientation_quat[2])
            oc.orientation.w = float(target_orientation_quat[3])
        oc.absolute_x_axis_tolerance = 0.1
        oc.absolute_y_axis_tolerance = 0.1
        oc.absolute_z_axis_tolerance = 0.1
        constraints = Constraints()
        constraints.position_constraints = [pc]
        constraints.orientation_constraints = [oc]
        req.goal_constraints = [constraints]
        req.start_state = RobotState()
        req.start_state.is_diff = True
        goal.request = req
        goal.planning_options.plan_only = False
        goal.planning_options.replan = False
        self._move_result = None
        self._goal_handle = None
        self._move_client.send_goal_async(goal).add_done_callback(self._mg_goal_response)
        self.get_logger().info(
            f"Pilz PTP pose goal: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")

    def _solve_ik_moveit(self, target_position, target_orientation_quat=None):
        """Call MoveIt /compute_ik for collision-aware IK seeded with current joints.
        Returns a Future, or None if service not ready."""
        if not self._ik_srv.service_is_ready():
            return None
        req = GetPositionIK.Request()
        ik = req.ik_request
        ik.group_name = PLANNING_GROUP
        ik.avoid_collisions = True
        ik.ik_link_name = EE_LINK
        if self._current_joints is not None:
            ik.robot_state.joint_state.name = list(JOINT_NAMES)
            ik.robot_state.joint_state.position = list(self._current_joints)
        ik.robot_state.is_diff = False
        ik.pose_stamped.header.frame_id = BASE_FRAME
        ik.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        ik.pose_stamped.pose.position.x = float(target_position[0])
        ik.pose_stamped.pose.position.y = float(target_position[1])
        ik.pose_stamped.pose.position.z = float(target_position[2])
        if target_orientation_quat is None:
            ik.pose_stamped.pose.orientation.y = 1.0  # tool-down
        else:
            ik.pose_stamped.pose.orientation.x = float(target_orientation_quat[0])
            ik.pose_stamped.pose.orientation.y = float(target_orientation_quat[1])
            ik.pose_stamped.pose.orientation.z = float(target_orientation_quat[2])
            ik.pose_stamped.pose.orientation.w = float(target_orientation_quat[3])
        ik.timeout.sec = 1
        return self._ik_srv.call_async(req)

    def _send_pilz_ptp_joint_goal(self, joint_values, vel_scale=0.3, acc_scale=0.3):
        """Pilz PTP: deterministic shortest joint-space path."""
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = PLANNING_GROUP
        req.pipeline_id = "pilz"
        req.planner_id = "PTP"
        req.num_planning_attempts = 1
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = vel_scale
        req.max_acceleration_scaling_factor = acc_scale
        jcs = []
        for name, val in zip(JOINT_NAMES, joint_values):
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
        self._move_result = None
        self._goal_handle = None
        self._move_client.send_goal_async(goal).add_done_callback(self._mg_goal_response)
        self.get_logger().info(f"Pilz PTP joint goal sent (vel={vel_scale})")

    def _send_pilz_lin_pose_goal(self, target_position, target_orientation_quat=None,
                                  vel_scale=0.1, acc_scale=0.1):
        """Pilz LIN: deterministic Cartesian straight-line path."""
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = PLANNING_GROUP
        req.pipeline_id = "pilz"
        req.planner_id = "LIN"
        req.num_planning_attempts = 1
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = vel_scale
        req.max_acceleration_scaling_factor = acc_scale
        # Position constraint
        pc = PositionConstraint()
        pc.header.frame_id = BASE_FRAME
        pc.link_name = EE_LINK
        pc.weight = 1.0
        bv = BoundingVolume()
        sp = SolidPrimitive()
        sp.type = SolidPrimitive.SPHERE
        sp.dimensions = [0.001]
        bv.primitives = [sp]
        p = Pose()
        p.position.x = float(target_position[0])
        p.position.y = float(target_position[1])
        p.position.z = float(target_position[2])
        p.orientation.w = 1.0
        bv.primitive_poses = [p]
        pc.constraint_region = bv
        # Orientation constraint (preserve current or tool-down)
        oc = OrientationConstraint()
        oc.header.frame_id = BASE_FRAME
        oc.link_name = EE_LINK
        oc.weight = 1.0
        if target_orientation_quat is None:
            try:
                tf = self._tf_buffer.lookup_transform(BASE_FRAME, EE_LINK, rclpy.time.Time())
                oc.orientation = tf.transform.rotation
            except Exception:
                oc.orientation.y = 1.0  # fallback: tool-down
        else:
            oc.orientation.x = float(target_orientation_quat[0])
            oc.orientation.y = float(target_orientation_quat[1])
            oc.orientation.z = float(target_orientation_quat[2])
            oc.orientation.w = float(target_orientation_quat[3])
        oc.absolute_x_axis_tolerance = 0.01
        oc.absolute_y_axis_tolerance = 0.01
        oc.absolute_z_axis_tolerance = 0.01
        constraints = Constraints()
        constraints.position_constraints = [pc]
        constraints.orientation_constraints = [oc]
        req.goal_constraints = [constraints]
        req.start_state = RobotState()
        req.start_state.is_diff = True
        goal.request = req
        goal.planning_options.plan_only = False
        goal.planning_options.replan = False
        self._move_result = None
        self._goal_handle = None
        self._move_client.send_goal_async(goal).add_done_callback(self._mg_goal_response)
        self.get_logger().info(
            f"Pilz LIN goal: ({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f})")

    def _build_joint_mg_goal(self, joint_values):
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = PLANNING_GROUP
        req.pipeline_id = "ompl"
        req.planner_id = "RRTConnectkConfigDefault"
        req.num_planning_attempts = 10
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor = 0.3
        req.max_acceleration_scaling_factor = 0.3

        jcs = []
        for name, val in zip(JOINT_NAMES, joint_values):
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
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3
        return goal

    # ── Cartesian straight-line Z move ──────────────────────────────
    def _send_cartesian_z_move(self, dz, vel_scale=0.15):
        """Plan and execute a straight-line move in Z by dz metres.
        Always checks self-collision and static collision objects."""
        try:
            tf = self._tf_buffer.lookup_transform(
                BASE_FRAME, EE_LINK, rclpy.time.Time())
        except Exception as e:
            self.get_logger().error(f"TF lookup failed for Cartesian move: {e}")
            self._move_result = False
            return

        # Target pose: same XY/orientation, shifted Z
        target = Pose()
        target.position.x = tf.transform.translation.x
        target.position.y = tf.transform.translation.y
        target.position.z = tf.transform.translation.z + dz
        target.orientation = tf.transform.rotation

        req = GetCartesianPath.Request()
        req.header.frame_id = BASE_FRAME
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = PLANNING_GROUP
        req.link_name = EE_LINK
        req.waypoints = [target]
        req.max_step = 0.005        # 5mm resolution
        req.avoid_collisions = True
        req.start_state = RobotState()
        req.start_state.is_diff = True

        self.get_logger().info(
            f"Computing Cartesian path: dz={dz:+.3f}m "
            f"(target Z={target.position.z:.3f})")

        self._cartesian_vel_scale = vel_scale
        self._move_result = None
        future = self._cartesian_srv.call_async(req)
        future.add_done_callback(self._cartesian_path_response)

    def _send_cartesian_to_pose(self, target_xyz, vel_scale=0.1):
        """Cartesian straight-line move to an absolute [x, y, z] position.
        Preserves current tool orientation. Used for grasp approach."""
        try:
            tf = self._tf_buffer.lookup_transform(
                BASE_FRAME, EE_LINK, rclpy.time.Time())
        except Exception as e:
            self.get_logger().error(f"TF lookup failed for Cartesian move: {e}")
            self._move_result = False
            return

        target = Pose()
        target.position.x = float(target_xyz[0])
        target.position.y = float(target_xyz[1])
        target.position.z = float(target_xyz[2])
        target.orientation = tf.transform.rotation  # preserve orientation

        req = GetCartesianPath.Request()
        req.header.frame_id = BASE_FRAME
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = PLANNING_GROUP
        req.link_name = EE_LINK
        req.waypoints = [target]
        req.max_step = 0.005
        req.avoid_collisions = True
        req.start_state = RobotState()
        req.start_state.is_diff = True

        self.get_logger().info(
            f"Cartesian to pose: ({target_xyz[0]:.3f}, "
            f"{target_xyz[1]:.3f}, {target_xyz[2]:.3f})")

        self._cartesian_vel_scale = vel_scale
        self._move_result = None
        future = self._cartesian_srv.call_async(req)
        future.add_done_callback(self._cartesian_path_response)

    def _scale_trajectory(self, trajectory, vel_scale):
        """Scale trajectory speed for ISO-compliant velocity limiting.
        vel_scale < 1.0 slows down the trajectory proportionally."""
        if vel_scale >= 1.0:
            return
        for pt in trajectory.joint_trajectory.points:
            # Stretch time stamps (slower = longer duration)
            t_ns = pt.time_from_start.sec * 1_000_000_000 + pt.time_from_start.nanosec
            new_ns = int(t_ns / vel_scale)
            pt.time_from_start.sec = new_ns // 1_000_000_000
            pt.time_from_start.nanosec = new_ns % 1_000_000_000
            # Scale velocities and accelerations accordingly
            pt.velocities = [v * vel_scale for v in pt.velocities]
            pt.accelerations = [a * vel_scale * vel_scale for a in pt.accelerations]

    def _cartesian_path_response(self, future):
        try:
            resp = future.result()
        except Exception as e:
            self.get_logger().error(f"Cartesian path service failed: {e}")
            self._move_result = False
            return

        if resp.fraction < 0.90:
            self.get_logger().error(
                f"Cartesian path only {resp.fraction*100:.0f}% complete.")
            self._move_result = False
            return

        # Apply velocity scaling for ISO/TS 15066 compliance
        vel_scale = getattr(self, '_cartesian_vel_scale', 1.0)
        if vel_scale < 1.0:
            self._scale_trajectory(resp.solution, vel_scale)

        self.get_logger().info(
            f"Cartesian path computed ({resp.fraction*100:.0f}%), "
            f"executing (vel_scale={vel_scale})...")

        # Execute the trajectory
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = resp.solution
        self._exec_client.send_goal_async(goal).add_done_callback(
            self._exec_goal_response)

    def _exec_goal_response(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error("ExecuteTrajectory goal REJECTED!")
            self._move_result = False
            return
        gh.get_result_async().add_done_callback(self._exec_result)

    def _exec_result(self, future):
        result = future.result().result
        ok = result.error_code.val == MoveItErrorCodes.SUCCESS
        if ok:
            self.get_logger().info("Cartesian execution SUCCESS.")
        else:
            self.get_logger().error(
                f"Cartesian execution FAILED (code={result.error_code.val}).")
        self._move_result = ok

    def _mg_goal_response(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error("MoveGroup goal REJECTED!")
            self._move_result = False
            return
        self._goal_handle = gh
        gh.get_result_async().add_done_callback(self._mg_result)

    def _mg_result(self, future):
        result = future.result().result
        ok = result.error_code.val == MoveItErrorCodes.SUCCESS
        if ok:
            self.get_logger().info("MoveGroup execution SUCCESS.")
        else:
            self.get_logger().error(
                f"MoveGroup execution FAILED (code={result.error_code.val}).")
        self._move_result = ok

    def _wait_move_result(self):
        if self._move_result is None:
            return

        if not self._move_result:
            self.get_logger().error(
                f"Move failed in {self._phase} — aborting.")
            self._switch_controllers(
                activate=["forward_velocity_controller"],
                deactivate=[self._jtc_name],
                done_cb=self._on_switch_done)
            self._set_phase(Phase.DONE)
            return

        if self._phase == Phase.MOVE_TO_PREGRASP:
            # Cartesian LIN descent to exact grasp tool0 target.
            # Uses absolute target (not relative dz) so any OMPL
            # arrival error doesn't propagate to the grasp position.
            grasp_z = self._grasp_tool0_target[2]
            self.get_logger().info(
                f"At pre-grasp — Cartesian descent to grasp "
                f"(target z={grasp_z:.3f}).")
            self._set_phase(Phase.MOVE_TO_GRASP)
            self._send_cartesian_to_pose(self._grasp_tool0_target)

        elif self._phase == Phase.MOVE_TO_GRASP:
            self.get_logger().info("At grasp position — closing gripper.")
            self._set_phase(Phase.GRASP)

        elif self._phase == Phase.LIFT:
            self._lift_position = self._lookup_ee_position()
            self.get_logger().info(
                f"Lift complete — EE at {self._lift_position}.")
            # Re-add workspace table collision (removed before grasp approach)
            self._readd_workspace_table()
            # V4: go directly to servo mode (object stays attached)
            self._set_phase(Phase.ENABLE_SERVO)

        elif self._phase == Phase.DESCEND_TO_PLACE:
            self._place_position = self._lookup_ee_position()
            self.get_logger().info(
                f"Descent complete — EE at {self._place_position}.")
            self._set_phase(Phase.PLACE)

    # ==================================================================
    # GRASP — close gripper, attach box, lift
    # ==================================================================
    def _do_grasp(self):
        elapsed = self._elapsed()

        if not self._grasp_started:
            self._grasp_started = True
            if self._use_gripper:
                msg = Float64MultiArray()
                msg.data = self._gripper_cmd
                self._pub_gripper.publish(msg)
                self.get_logger().info("Closing gripper...")
            else:
                self.get_logger().info(
                    ">>> Place object in gripper and press ENTER <<<")
                self._input_received = False
                self._start_input_thread()
            return

        if self._use_gripper:
            if elapsed < 3.0:
                msg = Float64MultiArray()
                msg.data = self._gripper_cmd
                self._pub_gripper.publish(msg)
                return

            if not self._grasp_switch_started:
                self._grasp_switch_started = True
                self._attach_box_to_ee()
                ee_pos = self._lookup_ee_position()
                if ee_pos:
                    lift_h = self._descent_distance
                    self.get_logger().info(
                        f"Lifting {lift_h*100:.0f}cm via Pilz LIN.")
                    self._set_phase(Phase.LIFT)
                    self._send_pilz_lin_pose_goal(
                        [ee_pos[0], ee_pos[1], ee_pos[2] + lift_h],
                        vel_scale=0.15, acc_scale=0.15)
                else:
                    self.get_logger().error("Cannot read EE position for lift")
                    self._move_result = False
        else:
            if not self._input_received:
                return
            self._attach_box_to_ee()
            ee_pos = self._lookup_ee_position()
            if ee_pos:
                lift_h = self._descent_distance
                self.get_logger().info(
                    f"Lifting {lift_h*100:.0f}cm via Pilz LIN.")
                self._set_phase(Phase.LIFT)
                self._send_pilz_lin_pose_goal(
                    [ee_pos[0], ee_pos[1], ee_pos[2] + lift_h],
                    vel_scale=0.15, acc_scale=0.15)
            else:
                self.get_logger().error("Cannot read EE position for lift")
                self._move_result = False

    def _start_input_thread(self):
        def _wait():
            try:
                input()
            except EOFError:
                pass
            self._input_received = True
        threading.Thread(target=_wait, daemon=True).start()

    # ==================================================================
    # ENABLE_SERVO — switch to FVC, unpause, tare (with object), enable
    # ==================================================================
    def _do_enable_servo(self):
        elapsed = self._elapsed()

        if elapsed < 0.1:
            # FIRST: disable hybrid to prevent it from driving toward
            # the old target while we switch controllers.
            msg = Bool()
            msg.data = False
            self._pub_enabled.publish(msg)

            # Re-seed target BEFORE activating FVC/servo
            self._publish_current_pose_as_target()

            self._switch_done = False
            self._switch_controllers(
                activate=["forward_velocity_controller"],
                deactivate=[self._jtc_name],
                done_cb=self._on_switch_done)
            self._call_pause(False)

            # Tare with object in hand — zeroes out gravity+object weight
            if self._tare_cli.service_is_ready():
                self.get_logger().info(
                    "Re-taring FT sensor (zeroing object weight)...")
                self._tare_cli.call_async(Trigger.Request())
            return

        if elapsed < TARE_SETTLE_S:
            return

        # Re-seed target again after tare settle (robot may have shifted)
        if elapsed < TARE_SETTLE_S + 0.1:
            self._publish_current_pose_as_target()
            return

        # Wait for target to propagate through integrator, THEN enable
        if elapsed < TARE_SETTLE_S + 0.3:
            return

        if elapsed < TARE_SETTLE_S + 0.4:
            msg = Bool()
            msg.data = True
            self._pub_enabled.publish(msg)
            self.get_logger().info("Hybrid controller enabled (target settled).")
            return

        if elapsed < TARE_SETTLE_S + 0.4 + GUARD_PERIOD_S:
            return

        box_h = self._box_dims[2] if self._box_dims else DEFAULT_BOX_DIMS[2]
        safe_z = WORKSPACE_TABLE_Z + self._tool0_to_fingertip + box_h / 2.0 + PLACE_MARGIN_M
        # Start interaction classifier now (only needed for guiding phase)
        if self._classifier_proc is None:
            cmd = [
                "ros2", "run", "my_thesis_controller", "interaction_classifier",
                "--ros-args",
                "-p", f"use_sim_time:={str(self._use_sim).lower()}",
                "-p", "nudge_threshold:=4.0",
                "-p", "z_score_thresh:=2.0",
                "-p", "nudge_cooldown_s:=0.15",
            ]
            self._classifier_proc = subprocess.Popen(cmd)
            self.get_logger().info("Started interaction_classifier (post-lift).")

        self.get_logger().info(
            f">>> Object lifted. Guide robot to desired location. "
            f"Double tap to place. <<<")
        self.get_logger().info(
            f"    Safe placement tool0 z = {safe_z:.3f}m "
            f"(table={WORKSPACE_TABLE_Z}, obj_h={box_h*100:.1f}cm)")
        self._await_start_time = self.get_clock().now().nanoseconds / 1e9
        self._last_interaction_time = self._await_start_time
        self._nudge_received = False
        self._double_tap_received = False
        self._first_tap_time = None
        self._set_phase(Phase.AWAIT_INTERACTION)

    # ==================================================================
    # AWAIT_INTERACTION — human guides robot with object
    # ==================================================================
    def _do_await_interaction(self):
        now_s = self.get_clock().now().nanoseconds / 1e9
        old = self._prev_interaction_state
        new = self._interaction_state

        if old != new:
            self._prev_interaction_state = new

            if old == "IDLE" and new == "GUIDING":
                self._guide_cycle_count += 1
                self._current_cycle = {
                    "index": self._guide_cycle_count,
                    "start": now_s, "end": None,
                    "force_samples": [], "velocity_samples": [],
                    "latency": None,
                }
                self._force_above_thresh_time = None
                self._velocity_above_thresh_time = None
                self._force_samples = []
                self._velocity_samples = []
                self._last_interaction_time = now_s
                self.get_logger().info(
                    f"Guide cycle {self._guide_cycle_count} started.")

            elif old == "GUIDING" and new == "IDLE":
                if self._current_cycle is not None:
                    self._current_cycle["end"] = now_s
                    self._current_cycle["force_samples"] = list(
                        self._force_samples)
                    self._current_cycle["velocity_samples"] = list(
                        self._velocity_samples)
                    if (self._force_above_thresh_time is not None
                            and self._velocity_above_thresh_time is not None):
                        self._current_cycle["latency"] = (
                            self._velocity_above_thresh_time
                            - self._force_above_thresh_time)
                    self._guide_cycles.append(self._current_cycle)
                    dur = now_s - self._current_cycle["start"]
                    self.get_logger().info(
                        f"Guide cycle {self._current_cycle['index']} "
                        f"ended ({dur:.2f}s).")
                    self._current_cycle = None
                self._last_interaction_time = now_s

        # Expire stale first-tap if window elapsed
        if (self._first_tap_time is not None
                and not self._double_tap_received):
            tap_age = now_s - self._first_tap_time
            if tap_age > DOUBLE_TAP_WINDOW_S:
                self._first_tap_time = None

        if self._double_tap_received:
            self.get_logger().info(
                "Double tap detected — preparing to place object.")
            # Close any active guide cycle
            if self._current_cycle is not None:
                self._current_cycle["end"] = now_s
                self._current_cycle["force_samples"] = list(
                    self._force_samples)
                self._current_cycle["velocity_samples"] = list(
                    self._velocity_samples)
                if (self._force_above_thresh_time is not None
                        and self._velocity_above_thresh_time is not None):
                    self._current_cycle["latency"] = (
                        self._velocity_above_thresh_time
                        - self._force_above_thresh_time)
                self._guide_cycles.append(self._current_cycle)
                self._current_cycle = None
            self._set_phase(Phase.PREP_PLACE)
            return

        if now_s - self._last_interaction_time > INTERACTION_TIMEOUT_S:
            self.get_logger().warn(
                f"No interaction for {INTERACTION_TIMEOUT_S:.0f}s — "
                "auto-placing object.")
            self._set_phase(Phase.PREP_PLACE)

    # ==================================================================
    # PREP_PLACE — check height safety, place or reject
    # ==================================================================
    def _do_prep_place(self):
        if not self._prep_place_started:
            self._prep_place_started = True

            # Record where the human guided the robot
            self._place_position = self._lookup_ee_position()
            self.get_logger().info(
                f"Placing at human-guided position: {self._place_position}")

            # Safety check: object bottom must be within MAX_PLACE_HEIGHT
            # of the workspace table surface for a safe drop.
            MAX_PLACE_HEIGHT = 0.05  # max 5 cm above table surface
            if self._place_position is not None:
                box_h = self._box_dims[2] if self._box_dims else DEFAULT_BOX_DIMS[2]
                fingertip_z = self._place_position[2] - self._tool0_to_fingertip
                obj_bottom_z = fingertip_z - box_h / 2.0
                drop_height = obj_bottom_z - WORKSPACE_TABLE_Z

                if drop_height > MAX_PLACE_HEIGHT:
                    self.get_logger().warn(
                        f"UNSAFE placement height! Object would drop "
                        f"{drop_height*100:.1f}cm to table surface. "
                        f"Guide robot lower and double-tap again.")
                    # Reset double-tap state and go back to guiding
                    self._double_tap_received = False
                    self._first_tap_time = None
                    self._prep_place_started = False
                    self._set_phase(Phase.AWAIT_INTERACTION)
                    return

            # Disable hybrid controller
            msg = Bool()
            msg.data = False
            self._pub_enabled.publish(msg)

            # Pause servo
            self._call_pause(True)
            return

        # Human-chosen location is safe — place
        self._descent_needed = False
        self._set_phase(Phase.PLACE)

    # ==================================================================
    # PLACE — open gripper, detach box
    # ==================================================================
    def _do_place(self):
        elapsed = self._elapsed()

        if elapsed < 0.1:
            if self._use_gripper:
                msg = Float64MultiArray()
                msg.data = GRIPPER_OPEN_HW if self._hardware_gripper else GRIPPER_OPEN_SIM
                self._pub_gripper.publish(msg)
            self._detach_box()
            self.get_logger().info("Opening gripper — placing object.")
            return

        if self._use_gripper:
            if elapsed < 2.0:
                if int(elapsed * 10) % 5 == 0:
                    msg = Float64MultiArray()
                    msg.data = GRIPPER_OPEN_HW if self._hardware_gripper else GRIPPER_OPEN_SIM
                    self._pub_gripper.publish(msg)
                return
        else:
            if elapsed < 1.0:
                return

        self._place_position = self._lookup_ee_position()
        self.get_logger().info(
            f"Object placed at {self._place_position}.")
        self._set_phase(Phase.RETRACT)

    # ==================================================================
    # RETRACT — go home
    # ==================================================================
    def _do_retract(self):
        if self._elapsed() < 0.1:
            # Ensure hybrid is off and servo is paused (may already be)
            msg = Bool()
            msg.data = False
            self._pub_enabled.publish(msg)
            self._call_pause(True)
            # Ensure JTC is active (may already be from PREP_PLACE)
            self._switch_controllers(
                activate=[self._jtc_name],
                deactivate=["forward_velocity_controller"],
                done_cb=self._on_switch_done)
            return

        if self._elapsed() < 1.0:
            return

        # Step 1: Lift gripper clear of placed object before homing
        if not self._retract_lift_sent:
            self._retract_lift_sent = True
            self.get_logger().info(
                f"Retracting {LIFT_HEIGHT*100:.0f}cm upward before homing...")
            self._send_cartesian_z_move(+LIFT_HEIGHT, vel_scale=0.3)
            return

        # Step 2: Wait for lift to complete
        if self._move_result is None:
            return

        if not self._move_result:
            self.get_logger().warn("Retract lift failed — homing anyway.")

        # Step 3: Call move_to_home
        home_cli = self.create_client(Trigger, "/move_to_home/trigger")
        if home_cli.service_is_ready():
            self.get_logger().info("Calling /move_to_home/trigger ...")
            home_cli.call_async(Trigger.Request())
        else:
            self.get_logger().warn(
                "/move_to_home/trigger not available — skipping.")
        self._set_phase(Phase.DONE)

    # ==================================================================
    # TF helper
    # ==================================================================
    def _lookup_ee_position(self):
        try:
            tf = self._tf_buffer.lookup_transform(
                BASE_FRAME, EE_LINK, rclpy.time.Time())
            return [
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z,
            ]
        except Exception:
            return None

    def _publish_current_pose_as_target(self):
        try:
            tf = self._tf_buffer.lookup_transform(
                BASE_FRAME, EE_LINK, rclpy.time.Time())
            ps = PoseStamped()
            ps.header.frame_id = BASE_FRAME
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x = tf.transform.translation.x
            ps.pose.position.y = tf.transform.translation.y
            ps.pose.position.z = tf.transform.translation.z
            ps.pose.orientation = tf.transform.rotation
            self._pub_update_target.publish(ps)
            self.get_logger().info(
                "Published current EE pose as hybrid target.")
        except Exception as e:
            self.get_logger().warn(f"Could not publish current pose: {e}")

    # ==================================================================
    # Planning scene helpers
    # ==================================================================
    def _add_box_collision(self, center, dims):
        co = CollisionObject()
        co.header.frame_id = BASE_FRAME
        co.id = BOX_ID
        co.operation = CollisionObject.ADD

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = list(dims)

        pose = Pose()
        pose.position.x = float(center[0])
        pose.position.y = float(center[1])
        pose.position.z = float(center[2])
        pose.orientation.w = 1.0

        co.primitives = [prim]
        co.primitive_poses = [pose]

        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects = [co]
        self._pub_planning_scene.publish(scene)
        self.get_logger().info(
            f"Added '{BOX_ID}' to planning scene at "
            f"({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}).")

    def _attach_box_to_ee(self):
        aco = AttachedCollisionObject()
        aco.link_name = EE_LINK
        aco.touch_links = ROBOTIQ_TOUCH_LINKS + [EE_LINK]

        co = CollisionObject()
        co.header.frame_id = EE_LINK
        co.id = BOX_ID
        co.operation = CollisionObject.ADD

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        dims = self._box_dims if self._box_dims else DEFAULT_BOX_DIMS
        prim.dimensions = list(dims)

        pose = Pose()
        pose.position.z = dims[2] / 2.0 + 0.01
        pose.orientation.w = 1.0

        co.primitives = [prim]
        co.primitive_poses = [pose]
        aco.object = co

        scene = PlanningScene(is_diff=True)
        scene.robot_state.attached_collision_objects = [aco]
        scene.robot_state.is_diff = True
        self._pub_planning_scene.publish(scene)
        self.get_logger().info(f"Attached '{BOX_ID}' to {EE_LINK}.")

    def _detach_box(self):
        aco = AttachedCollisionObject()
        aco.link_name = EE_LINK
        aco.object.id = BOX_ID
        aco.object.operation = CollisionObject.REMOVE

        scene = PlanningScene(is_diff=True)
        scene.robot_state.attached_collision_objects = [aco]
        scene.robot_state.is_diff = True
        self._pub_planning_scene.publish(scene)

        co = CollisionObject()
        co.header.frame_id = BASE_FRAME
        co.id = BOX_ID
        co.operation = CollisionObject.REMOVE
        scene2 = PlanningScene(is_diff=True)
        scene2.world.collision_objects = [co]
        self._pub_planning_scene.publish(scene2)
        self.get_logger().info(f"Detached and removed '{BOX_ID}'.")

    def _remove_collision_object(self, obj_id):
        co = CollisionObject()
        co.header.frame_id = BASE_FRAME
        co.id = obj_id
        co.operation = CollisionObject.REMOVE
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects = [co]
        self._pub_planning_scene.publish(scene)
        self.get_logger().info(f"Removed '{obj_id}' from planning scene.")

    def _readd_workspace_table(self):
        """Re-add workspace table collision after lift."""
        co = CollisionObject()
        co.header.frame_id = "world"
        co.id = "workspace_table"
        co.operation = CollisionObject.ADD
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [0.59, 0.79, 0.69]
        pose = Pose()
        pose.position.x = -0.645
        pose.position.y = 0.0
        pose.position.z = -0.285
        pose.orientation.w = 1.0
        co.primitives = [prim]
        co.primitive_poses = [pose]
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects = [co]
        self._pub_planning_scene.publish(scene)
        self.get_logger().info("Re-added 'workspace_table' collision.")

    # ==================================================================
    # Controller switching
    # ==================================================================
    def _switch_controllers(self, activate, deactivate, done_cb):
        if not self._switch_cli.service_is_ready():
            self.get_logger().warn("switch_controller service not ready.")
            import concurrent.futures
            f = concurrent.futures.Future()
            f.set_result(None)
            done_cb(f)
            return
        req = SwitchController.Request()
        req.activate_controllers = activate
        req.deactivate_controllers = deactivate
        req.strictness = SwitchController.Request.BEST_EFFORT
        req.activate_asap = True
        self._switch_cli.call_async(req).add_done_callback(done_cb)

    def _on_switch_done(self, future):
        self._switch_done = True
        self.get_logger().info("Controller switch complete.")

    def _call_pause(self, pause: bool):
        if pause:
            if self._pause_cli.service_is_ready():
                self._pause_cli.call_async(Trigger.Request())
                self.get_logger().info("Servo PAUSED.")
            else:
                self.get_logger().warn("pause_servo not ready.")
        else:
            if self._unpause_cli.service_is_ready():
                self._unpause_cli.call_async(Trigger.Request())
                self.get_logger().info("Servo UNPAUSED.")
            else:
                self.get_logger().warn("unpause_servo not ready.")

    # ==================================================================
    # DONE — cleanup: disable hybrid, kill classifier, stop timer
    # ==================================================================
    def _do_done(self):
        elapsed = self._elapsed()

        if elapsed < 10.0:
            msg = Bool()
            msg.data = False
            self._pub_enabled.publish(msg)
            self._call_pause(True)
        else:
            if self._classifier_proc is not None:
                self._classifier_proc.terminate()
                self._classifier_proc = None
                self.get_logger().info("Killed interaction_classifier subprocess.")
            self._timer.cancel()

    # ==================================================================
    # Metrics summary
    # ==================================================================
    def _log_summary(self):
        log = self.get_logger().info
        sep = "=" * 60

        log(sep)
        log("  ASSISTIVE LIFT V4 — HUMAN-GUIDED PLACEMENT SUMMARY")
        log(sep)

        log("  Phase Durations:")
        total_time = 0.0
        for phase, duration in self._timestamps.items():
            log(f"    {phase:30s}: {duration:6.2f}s")
            total_time += duration
        log(f"    {'TOTAL':30s}: {total_time:6.2f}s")

        if self._box_center_bl and self._box_dims:
            cx, cy, cz = self._box_center_bl
            w, d, h = self._box_dims
            log("")
            log("  Perception:")
            log(f"    Method:     {'YOLO+depth' if self._yolo else 'HSV+depth'}")
            log(f"    Box center: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
            log(f"    Box dims:   {w*100:.1f} x {d*100:.1f} x {h*100:.1f} cm")
            log(f"    Attempts:   {self._perception_attempts}")

        log("")
        log("  Placement:")
        if self._guided_position:
            log(f"    Guided to:      ({self._guided_position[0]:.3f}, "
                f"{self._guided_position[1]:.3f}, "
                f"{self._guided_position[2]:.3f})")
        if self._place_position:
            log(f"    Placed at:      ({self._place_position[0]:.3f}, "
                f"{self._place_position[1]:.3f}, "
                f"{self._place_position[2]:.3f})")
        log(f"    Descent needed: {'YES' if self._descent_needed else 'NO'}")
        if self._descent_needed:
            log(f"    Descent amount: {self._descent_amount:.3f}m")

        log("")
        log("  Interaction:")
        total_guiding = 0.0
        for c in self._guide_cycles:
            if c["end"] is not None and c["start"] is not None:
                total_guiding += c["end"] - c["start"]
        log(f"    Total guiding duration: {total_guiding:.1f}s")
        log(f"    Guide cycles:           {len(self._guide_cycles)}")

        for c in self._guide_cycles:
            idx = c["index"]
            dur = (c["end"] - c["start"]) if c["end"] else 0.0

            forces = c.get("force_samples", [])
            if forces:
                f_mags = [math.sqrt(f[1]**2 + f[2]**2 + f[3]**2)
                          for f in forces]
                peak_f = max(f_mags)
                mean_f = sum(f_mags) / len(f_mags)
            else:
                peak_f = mean_f = 0.0

            vels = c.get("velocity_samples", [])
            v_peak = max(
                (math.sqrt(v[1]**2 + v[2]**2 + v[3]**2) for v in vels),
                default=0.0)

            latency = c.get("latency")
            lat_str = f"{latency:.3f}s" if latency is not None else "n/a"

            log(f"  Cycle {idx}: {dur:5.1f}s  peak={peak_f:5.1f}N  "
                f"mean={mean_f:5.1f}N  v_peak={v_peak:.2f}m/s  "
                f"latency={lat_str}")

        all_forces = []
        for c in self._guide_cycles:
            all_forces.extend(c.get("force_samples", []))

        if all_forces:
            f_mags_all = [math.sqrt(f[1]**2 + f[2]**2 + f[3]**2)
                          for f in all_forces]
            peak_idx = f_mags_all.index(max(f_mags_all))
            log("")
            log("  Overall:")
            log(f"    Peak force: {max(f_mags_all):5.1f} N "
                f"(Fx={all_forces[peak_idx][1]:.1f}, "
                f"Fy={all_forces[peak_idx][2]:.1f}, "
                f"Fz={all_forces[peak_idx][3]:.1f})")

        if self._lift_position and self._place_position:
            disp = [self._place_position[i] - self._lift_position[i]
                    for i in range(3)]
            disp_total = math.sqrt(sum(d * d for d in disp))
            log(f"    Displacement (lift→place): "
                f"[{disp[0]:.2f}, {disp[1]:.2f}, {disp[2]:.2f}]m  "
                f"total={disp_total:.2f}m")
        else:
            log("    Displacement: (TF lookup failed)")

        if self._state_log:
            log("")
            log("  State transitions:")
            for t, old, new in self._state_log:
                log(f"    t={t:.2f}  {old} -> {new}")

        log(sep)

        self._export_csv()

        log("Test complete. Ctrl+C to exit.")


    # ==================================================================
    # CSV export for thesis metrics
    # ==================================================================
    def _export_csv(self):
        tid = self._trial_id
        d = self._experiment_dir
        log = self.get_logger().info

        # ── task_summary ──────────────────────────────────────────────
        total_time = sum(self._timestamps.values())
        grasp_success = Phase.PLACE in self._timestamps

        # Per-cycle force stats
        all_forces = []
        for c in self._guide_cycles:
            for f in c.get("force_samples", []):
                all_forces.append(math.sqrt(f[1]**2 + f[2]**2 + f[3]**2))
        peak_force = max(all_forces) if all_forces else 0.0
        mean_force = (sum(all_forces) / len(all_forces)) if all_forces else 0.0

        displacement = 0.0
        if self._lift_position and self._place_position:
            displacement = math.sqrt(sum(
                (self._place_position[i] - self._lift_position[i])**2
                for i in range(3)))

        # Build phase columns
        all_phases = [
            Phase.INIT, Phase.PERCEIVE, Phase.PLAN_GRASP,
            Phase.MOVE_TO_PREGRASP, Phase.MOVE_TO_GRASP, Phase.GRASP,
            Phase.LIFT, Phase.ENABLE_SERVO, Phase.AWAIT_INTERACTION,
            Phase.PREP_PLACE, Phase.DESCEND_TO_PLACE, Phase.PLACE,
            Phase.RETRACT,
        ]
        phase_cols = [f"phase_{p}_s" for p in all_phases]
        phase_vals = [self._timestamps.get(p, 0.0) for p in all_phases]

        header = [
            "trial_id", "total_time_s", "grasp_success",
            "num_guide_cycles", "peak_force_N", "mean_force_N",
            "double_tap_time_ms", "displacement_m",
        ] + phase_cols
        row = [
            tid, f"{total_time:.3f}", str(grasp_success),
            str(len(self._guide_cycles)), f"{peak_force:.2f}",
            f"{mean_force:.2f}",
            f"{self._double_tap_time_ms:.1f}" if self._double_tap_time_ms is not None else "",
            f"{displacement:.4f}",
        ] + [f"{v:.3f}" for v in phase_vals]

        path = os.path.join(d, f"task_summary_{tid}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerow(row)
        log(f"CSV written: {path}")

        # ── guide_cycles ──────────────────────────────────────────────
        path = os.path.join(d, f"guide_cycles_{tid}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "trial_id", "cycle", "duration_s", "peak_force_N",
                "mean_force_N", "peak_velocity_mps", "latency_ms",
            ])
            for c in self._guide_cycles:
                dur = (c["end"] - c["start"]) if c["end"] else 0.0
                forces = c.get("force_samples", [])
                f_mags = [math.sqrt(s[1]**2 + s[2]**2 + s[3]**2)
                          for s in forces]
                pk_f = max(f_mags) if f_mags else 0.0
                mn_f = (sum(f_mags) / len(f_mags)) if f_mags else 0.0
                vels = c.get("velocity_samples", [])
                v_pk = max(
                    (math.sqrt(v[1]**2 + v[2]**2 + v[3]**2) for v in vels),
                    default=0.0)
                lat = c.get("latency")
                lat_ms = f"{lat * 1000.0:.1f}" if lat is not None else ""
                w.writerow([
                    tid, c["index"], f"{dur:.3f}", f"{pk_f:.2f}",
                    f"{mn_f:.2f}", f"{v_pk:.4f}", lat_ms,
                ])
        log(f"CSV written: {path}")

        # ── state_transitions ─────────────────────────────────────────
        path = os.path.join(d, f"state_transitions_{tid}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trial_id", "time_s", "from_state", "to_state"])
            for t, old, new in self._state_log:
                w.writerow([tid, f"{t:.3f}", old, new])
        log(f"CSV written: {path}")


# ======================================================================
def main():
    rclpy.init()
    node = AssistiveLiftV4Node()
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
