#!/usr/bin/env python3
"""
Hybrid position + admittance controller for UR3e using MoveIt Servo.

- Subscribes:
  - wrench_topic (geometry_msgs/WrenchStamped)  default: /force_torque
  - target_pose_topic (geometry_msgs/PoseStamped) default: /hybrid_controller/target_pose
- Publishes:
  - twist_topic (geometry_msgs/TwistStamped) default: /servo_node/delta_twist_cmds

Design:
- Position part: P control on pose error -> v_pos (6D)
- Admittance part: a = M^-1 (F_filtered - F_target - D*v_adm) -> integrate to v_adm (6D)
- Hybrid mixing: v_cmd = S * v_pos + (1 - S) * v_adm   (S is 6-element, can be in [0..1])
- Safety clamps: max linear/angular speeds
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import TwistStamped, WrenchStamped, PoseStamped
from std_msgs.msg import Bool, Float32, String
from scipy.spatial.transform import Rotation as R
import tf2_ros
from geometry_msgs.msg import Vector3Stamped
from tf2_geometry_msgs import do_transform_vector3




class HybridControlNode(Node):
    def __init__(self):
        super().__init__("hybrid_control_node")

        # ---------------------------
        # Parameters (topics/frames)
        # ---------------------------
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tool_frame", "tool0")

        self.declare_parameter("wrench_topic", "/wrench_zeroed")
        self.declare_parameter("target_pose_topic", "/hybrid_controller/target_pose")
        self.declare_parameter("twist_topic", "/servo_node/delta_twist_cmds")
        self.declare_parameter("enabled_topic", "/hybrid_controller/enabled")

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tool_frame = str(self.get_parameter("tool_frame").value)
        self.wrench_topic = str(self.get_parameter("wrench_topic").value)
        self.target_pose_topic = str(self.get_parameter("target_pose_topic").value)
        self.twist_topic = str(self.get_parameter("twist_topic").value)
        self.enabled_topic = str(self.get_parameter("enabled_topic").value)
        self.enabled = True

        self.enabled_sub = self.create_subscription(
            Bool,
            self.enabled_topic,
            self.enabled_callback,
            10,
        )


        # ---------------------------
        # Parameters (hybrid mixing)
        # ---------------------------
        default_S = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]  # position in X,Y,rot; admittance in Z
        self.declare_parameter("selection_matrix", default_S)
        self.S = np.array(self.get_parameter("selection_matrix").value, dtype=float)
        if self.S.shape != (6,):
            raise ValueError("selection_matrix must be a list of 6 scalars")
        self.S = np.clip(self.S, 0.0, 1.0)

        # ---------------------------
        # Parameters (position control)
        # ---------------------------
        # Simple diagonal P gains; tune in YAML
        self.declare_parameter("kp_pos", [2.0, 2.0, 2.0])
        self.declare_parameter("kp_rot", [1.0, 1.0, 1.0])
        self.Kp_pos = np.diag(np.array(self.get_parameter("kp_pos").value, dtype=float))
        self.Kp_rot = np.diag(np.array(self.get_parameter("kp_rot").value, dtype=float))

        # ---------------------------
        # Parameters (admittance control) - aligned with admittance_servo.py
        # ---------------------------
        self.declare_parameter("mass_linear", 3.0)
        self.declare_parameter("mass_angular", 0.5)
        self.declare_parameter("damping_linear", 40.0)
        self.declare_parameter("damping_angular", 8.0)

        self.declare_parameter("force_deadband", 3.0)
        self.declare_parameter("torque_deadband", 0.2)
        self.declare_parameter("alpha_wrench", 0.15)  # low-pass filter coefficient
        self.declare_parameter("admittance_output_alpha", 0.25)

        # Desired contact wrench (in the same frame as the wrench input after filtering)
        # You can set e.g. target_force: [0, 0, -10, 0, 0, 0] to maintain downward force.
        self.declare_parameter("target_wrench", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        m_lin = float(self.get_parameter("mass_linear").value)
        m_ang = float(self.get_parameter("mass_angular").value)
        d_lin = float(self.get_parameter("damping_linear").value)
        d_ang = float(self.get_parameter("damping_angular").value)

        self.force_deadband = float(self.get_parameter("force_deadband").value)
        self.torque_deadband = float(self.get_parameter("torque_deadband").value)
        self.alpha = float(self.get_parameter("alpha_wrench").value)
        self.admittance_output_alpha = float(self.get_parameter("admittance_output_alpha").value)

        self.target_wrench = np.array(self.get_parameter("target_wrench").value, dtype=float)
        if self.target_wrench.shape != (6,):
            raise ValueError("target_wrench must be a list of 6 scalars")

        # Adaptive damping: D_eff = D0 + k * |F|
        self.declare_parameter("adaptive_damping_k_lin", 2.0)    # (Ns/m)/N
        self.declare_parameter("adaptive_damping_k_ang", 1.0)     # (Nms/rad)/Nm
        self.adaptive_damping_k_lin = float(self.get_parameter("adaptive_damping_k_lin").value)
        self.adaptive_damping_k_ang = float(self.get_parameter("adaptive_damping_k_ang").value)

        # Intent amplification: F_scaled = k * tanh(|F| / F0) * F_hat
        self.declare_parameter("intent_saturation_force", 15.0)   # N max effective force
        self.declare_parameter("intent_ref_force", 5.0)            # N sensitivity reference
        self.declare_parameter("intent_saturation_torque", 2.0)    # Nm
        self.declare_parameter("intent_ref_torque", 0.5)            # Nm
        self.intent_sat_force = float(self.get_parameter("intent_saturation_force").value)
        self.intent_ref_force = float(self.get_parameter("intent_ref_force").value)
        self.intent_sat_torque = float(self.get_parameter("intent_saturation_torque").value)
        self.intent_ref_torque = float(self.get_parameter("intent_ref_torque").value)

        # === INTERACTION STATE TRACKING ===
        self._interaction_state = "IDLE"
        self._was_guiding = False
        self._guiding_exit_count = 0          # debounce counter for GUIDING→other
        self._GUIDING_EXIT_THRESH = 1000      # iterations (~2s at 500Hz)
        self.create_subscription(
            String, '/interaction_state', self._interaction_state_cb, 10)
        self._update_target_pub = self.create_publisher(
            PoseStamped, '/hybrid_controller/update_target', 10)

        # === PROXIMITY VELOCITY SCALING ===
        self.proximity_scale = 1.0
        self.proximity_sub = self.create_subscription(
            Float32, '/proximity/scale', self.proximity_cb, 10)

        # Publish effective interaction state so proximity_monitor gets a
        # consistent view (no race between interaction_classifier and force_gui)
        self._effective_state_pub = self.create_publisher(
            String, '/effective_interaction_state', 10)

        # === EXTERNAL WRENCH (from force_gui or test tools) ===
        # Added on top of sensor wrench in base_link frame, no transform needed
        self.external_wrench = np.zeros(6, dtype=float)
        self.create_subscription(
            WrenchStamped, '/wrench_external', self._external_wrench_cb, 10)


        # Guiding damping parameters (lower = more responsive during GUIDING)
        self.declare_parameter("guiding_damping_linear", 30.0)
        self.declare_parameter("guiding_damping_angular", 5.0)
        gd_lin = float(self.get_parameter("guiding_damping_linear").value)
        gd_ang = float(self.get_parameter("guiding_damping_angular").value)

        self.M = np.diag([m_lin, m_lin, m_lin, m_ang, m_ang, m_ang])
        self.M_inv = np.linalg.inv(self.M)
        self.D_normal = np.diag([d_lin, d_lin, d_lin, d_ang, d_ang, d_ang])
        self.D_guiding = np.diag([gd_lin, gd_lin, gd_lin, gd_ang, gd_ang, gd_ang])
        self.D = self.D_normal.copy()  # active damping matrix

        # ---------------------------
        # Parameters (timing & safety)
        # ---------------------------
        self.declare_parameter("loop_rate", 500.0)
        loop_rate = float(self.get_parameter("loop_rate").value)
        self.dt = 1.0 / loop_rate

        self.declare_parameter("resume_hold_s", 0.5)      # how long to enforce release check
        self.declare_parameter("resume_wrench_eps", 1.0)  # N (or use stop_force_epsilon)
        self.resume_hold_s = float(self.get_parameter("resume_hold_s").value)
        self.resume_wrench_eps = float(self.get_parameter("resume_wrench_eps").value)
        self._resume_until = 0.0


        self.declare_parameter("max_linear_speed", 0.25)
        self.declare_parameter("max_angular_speed", 1.0)
        self.max_v = float(self.get_parameter("max_linear_speed").value)
        self.max_w = float(self.get_parameter("max_angular_speed").value)

        # Smooth deceleration on force release (replaces instant velocity zeroing)
        self.declare_parameter("no_contact_decay_tau", 0.15)  # seconds
        self.no_contact_decay_tau = float(self.get_parameter("no_contact_decay_tau").value)

        # Acceleration rate limiter (jerk protection)
        self.declare_parameter("max_linear_accel", 1.0)   # m/s²
        self.declare_parameter("max_angular_accel", 3.0)   # rad/s²
        self.max_lin_accel = float(self.get_parameter("max_linear_accel").value)
        self.max_ang_accel = float(self.get_parameter("max_angular_accel").value)

        # ---------------------------
        # State
        # ---------------------------
        self._raw_wrench_lp = np.zeros(6, dtype=float)        # low-pass filter state (pre-bias)
        self.current_wrench = np.zeros(6, dtype=float)       # bias-corrected wrench (output)
        self.admittance_velocity = np.zeros(6, dtype=float)  # v_adm integrator state
        self.smoothed_admittance_velocity = np.zeros(6, dtype=float)
        self.current_pose = None
        self.target_pose = None

        # ---------------------------
        # TF
        # ---------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---------------------------
        # ROS interfaces
        # ---------------------------
        ft_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )

        self.wrench_sub = self.create_subscription(
            WrenchStamped,
            self.wrench_topic,
            self.wrench_callback,
            ft_qos,
        )

        pose_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        self.target_sub = self.create_subscription(
            PoseStamped,
            self.target_pose_topic,
            self.target_callback,
            pose_qos,
        )

        self.twist_pub = self.create_publisher(
            TwistStamped,
            self.twist_topic,
            10,
        )

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f"Hybrid node running. S={self.S.tolist()} "
            f"wrench={self.wrench_topic} target_pose={self.target_pose_topic} twist={self.twist_topic} "
            f"frames: {self.base_frame}->{self.tool_frame} rate={loop_rate}Hz"
        )
         #--stop rule
        self.declare_parameter("stop_force_epsilon", 0.5)   # N after deadband+filter
        self.declare_parameter("stop_torque_epsilon", 0.05) # Nm
        self.declare_parameter("stop_on_no_contact", True)

        self.stop_force_eps = float(self.get_parameter("stop_force_epsilon").value)
        self.stop_torque_eps = float(self.get_parameter("stop_torque_epsilon").value)
        self.stop_on_no_contact = bool(self.get_parameter("stop_on_no_contact").value)

        # Wrench handling
        self.declare_parameter("wrench_frame", self.tool_frame)  # transform all wrench to this frame
        self.declare_parameter("tare_on_enable", True)

        self.wrench_frame = str(self.get_parameter("wrench_frame").value)
        self.tare_on_enable = bool(self.get_parameter("tare_on_enable").value)

        self.wrench_bias = np.zeros(6, dtype=float)
        self._have_bias = False
        
        self.declare_parameter("wrench_source_frame", "ft_frame")
        self.wrench_source_frame = str(self.get_parameter("wrench_source_frame").value).strip()

        # Per-axis stop eps (optional). Default: use the scalar eps for all axes.
        self.declare_parameter("stop_fx_epsilon", self.stop_force_eps)
        self.declare_parameter("stop_fy_epsilon", self.stop_force_eps)
        self.declare_parameter("stop_fz_epsilon", self.stop_force_eps)

        self.declare_parameter("stop_tx_epsilon", self.stop_torque_eps)
        self.declare_parameter("stop_ty_epsilon", self.stop_torque_eps)
        self.declare_parameter("stop_tz_epsilon", self.stop_torque_eps)

        self.stop_fx_eps = float(self.get_parameter("stop_fx_epsilon").value)
        self.stop_fy_eps = float(self.get_parameter("stop_fy_epsilon").value)
        self.stop_fz_eps = float(self.get_parameter("stop_fz_epsilon").value)

        self.stop_tx_eps = float(self.get_parameter("stop_tx_epsilon").value)
        self.stop_ty_eps = float(self.get_parameter("stop_ty_epsilon").value)
        self.stop_tz_eps = float(self.get_parameter("stop_tz_epsilon").value)

        # Online bias estimator (auto-zero)
        self.declare_parameter("auto_bias_enable", True)
        self.declare_parameter("auto_bias_beta", 0.002)          # 0..1, small
        self.declare_parameter("auto_bias_force_thresh", 2.0)    # N
        self.declare_parameter("auto_bias_torque_thresh", 0.2)   # Nm
        self.declare_parameter("auto_bias_max_cmd_speed", 0.02)  # m/s equivalent gate

        self.auto_bias_enable = bool(self.get_parameter("auto_bias_enable").value)
        self.auto_bias_beta = float(self.get_parameter("auto_bias_beta").value)
        self.auto_bias_force_thresh = float(self.get_parameter("auto_bias_force_thresh").value)
        self.auto_bias_torque_thresh = float(self.get_parameter("auto_bias_torque_thresh").value)
        self.auto_bias_max_cmd_speed = float(self.get_parameter("auto_bias_max_cmd_speed").value)

        self._last_v_cmd = np.zeros(6, dtype=float)

        # Register dynamic parameter callback for live tuning
        self.add_on_set_parameters_callback(self._on_param_change)

    # ----------------- Dynamic parameter callback -----------------

    def _on_param_change(self, params):
        """Update internal state when parameters are changed at runtime via ros2 param set."""
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            name = p.name
            val = p.value
            if name == "max_linear_accel":
                self.max_lin_accel = float(val)
            elif name == "max_angular_accel":
                self.max_ang_accel = float(val)
            elif name == "max_linear_speed":
                self.max_v = float(val)
            elif name == "max_angular_speed":
                self.max_w = float(val)
            elif name == "no_contact_decay_tau":
                self.no_contact_decay_tau = float(val)
            elif name == "alpha_wrench":
                self.alpha = float(val)
            elif name == "admittance_output_alpha":
                self.admittance_output_alpha = float(val)
            elif name == "force_deadband":
                self.force_deadband = float(val)
            elif name == "torque_deadband":
                self.torque_deadband = float(val)
            elif name == "mass_linear":
                m = float(val)
                self.M = np.diag([m, m, m, self.M[3, 3], self.M[4, 4], self.M[5, 5]])
                self.M_inv = np.linalg.inv(self.M)
            elif name == "mass_angular":
                m = float(val)
                self.M = np.diag([self.M[0, 0], self.M[1, 1], self.M[2, 2], m, m, m])
                self.M_inv = np.linalg.inv(self.M)
            elif name == "damping_linear":
                d = float(val)
                self.D_normal = np.diag([d, d, d, self.D_normal[3, 3], self.D_normal[4, 4], self.D_normal[5, 5]])
            elif name == "damping_angular":
                d = float(val)
                self.D_normal = np.diag([self.D_normal[0, 0], self.D_normal[1, 1], self.D_normal[2, 2], d, d, d])
            elif name == "guiding_damping_linear":
                d = float(val)
                self.D_guiding = np.diag([d, d, d, self.D_guiding[3, 3], self.D_guiding[4, 4], self.D_guiding[5, 5]])
            elif name == "guiding_damping_angular":
                d = float(val)
                self.D_guiding = np.diag([self.D_guiding[0, 0], self.D_guiding[1, 1], self.D_guiding[2, 2], d, d, d])
            elif name == "kp_pos":
                self.Kp_pos = np.diag(np.array(val, dtype=float))
            elif name == "kp_rot":
                self.Kp_rot = np.diag(np.array(val, dtype=float))
            elif name == "adaptive_damping_k_lin":
                self.adaptive_damping_k_lin = float(val)
            elif name == "adaptive_damping_k_ang":
                self.adaptive_damping_k_ang = float(val)
            elif name == "intent_saturation_force":
                self.intent_sat_force = float(val)
            elif name == "intent_ref_force":
                self.intent_ref_force = float(val)
            elif name == "intent_saturation_torque":
                self.intent_sat_torque = float(val)
            elif name == "intent_ref_torque":
                self.intent_ref_torque = float(val)
            else:
                continue
            self.get_logger().info(f"Parameter updated: {name} = {val}")
        return SetParametersResult(successful=True)

    # ----------------- Callbacks -----------------

    def target_callback(self, msg: PoseStamped):
        if self._was_guiding:
            return
        self.target_pose = msg

    @staticmethod
    def _smooth_deadband(vec: np.ndarray, deadband: float) -> np.ndarray:
        """Deadband with quadratic transition — always non-negative, C1-continuous.

        Transition zone: [0.7*deadband, 1.3*deadband].
        Below lower → 0.  Above upper → standard (mag-deadband)/mag.
        In between → output_mag = half_w * t² where t ∈ [0,1].
        """
        mag = np.linalg.norm(vec)
        if mag < 1e-9:
            return np.zeros_like(vec)
        half_w = deadband * 0.4
        lower = deadband - half_w
        upper = deadband + half_w
        if mag <= lower:
            return np.zeros_like(vec)
        if mag >= upper:
            return vec * (mag - deadband) / mag
        t = (mag - lower) / (upper - lower)
        output_mag = half_w * t * t
        return vec / mag * output_mag

    def wrench_callback(self, msg: WrenchStamped):
        src = self.wrench_source_frame if self.wrench_source_frame else msg.header.frame_id.strip()
        if src == "":
            src = self.wrench_frame
        # Transform force and torque vectors into wrench_frame
        try:
            tf = self.tf_buffer.lookup_transform(self.wrench_frame, src, rclpy.time.Time())
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f"Wrench TF failed {src} -> {self.wrench_frame}: {ex}")
            return

        f_in = Vector3Stamped()
        f_in.header = msg.header
        f_in.vector = msg.wrench.force

        t_in = Vector3Stamped()
        t_in.header = msg.header
        t_in.vector = msg.wrench.torque

        f_out = do_transform_vector3(f_in, tf).vector
        t_out = do_transform_vector3(t_in, tf).vector

        raw = np.array([f_out.x, f_out.y, f_out.z, t_out.x, t_out.y, t_out.z], dtype=float)

        # Smooth deadband (cubic transition zone avoids jerk at low forces)
        filtered = np.zeros(6, dtype=float)
        filtered[:3] = self._smooth_deadband(raw[:3], self.force_deadband)
        filtered[3:] = self._smooth_deadband(raw[3:], self.torque_deadband)

        # Low-pass on RAW (pre-bias) signal — separate state from bias-corrected output
        self._raw_wrench_lp = self.alpha * filtered + (1.0 - self.alpha) * self._raw_wrench_lp

        # Initialize bias once (first good sample)
        if self.tare_on_enable and (not self._have_bias):
            if np.linalg.norm(self._raw_wrench_lp[:3]) < 1.0 and np.linalg.norm(self._raw_wrench_lp[3:]) < 0.1:
                return
            self.wrench_bias = self._raw_wrench_lp.copy()
            self._have_bias = True
            self.get_logger().info(
               f"Initial wrench bias (frame={self.wrench_frame}): {self.wrench_bias.tolist()}"
            )

        # Bias-corrected wrench
        w = self._raw_wrench_lp - self.wrench_bias

        # Online bias adaptation (auto-zero) when near rest
        if self.auto_bias_enable and self._have_bias:
            f_ok = np.linalg.norm(w[:3]) < self.auto_bias_force_thresh
            t_ok = np.linalg.norm(w[3:]) < self.auto_bias_torque_thresh

            # Gate with command speed (avoid adapting during active contact/motion)
            cmd_speed = np.linalg.norm(getattr(self, "_last_v_cmd", np.zeros(6))[:3])
            v_ok = cmd_speed < self.auto_bias_max_cmd_speed

            if f_ok and t_ok and v_ok:
                self.wrench_bias = (1.0 - self.auto_bias_beta) * self.wrench_bias + self.auto_bias_beta * self._raw_wrench_lp
                w = self._raw_wrench_lp - self.wrench_bias

        self.current_wrench = w


        

    # ----------------- State estimation -----------------

    def get_robot_state(self) -> bool:
        """Update current_pose from TF: base_frame -> tool_frame."""
        try:
            t = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tool_frame,
                rclpy.time.Time(),
            )
            self.current_pose = t.transform
            return True
        except tf2_ros.TransformException:
            return False

    # ----------------- Control computations -----------------

    def compute_position_velocity(self) -> np.ndarray:
        """Compute v_pos from pose error using diagonal P control."""
        if self.target_pose is None or self.current_pose is None:
            return np.zeros(6, dtype=float)

        curr_pos = np.array(
            [
                self.current_pose.translation.x,
                self.current_pose.translation.y,
                self.current_pose.translation.z,
            ],
            dtype=float,
        )
        target_pos = np.array(
            [
                self.target_pose.pose.position.x,
                self.target_pose.pose.position.y,
                self.target_pose.pose.position.z,
            ],
            dtype=float,
        )
        linear_error = target_pos - curr_pos
        v_linear = self.Kp_pos @ linear_error

        r_curr = R.from_quat(
            [
                self.current_pose.rotation.x,
                self.current_pose.rotation.y,
                self.current_pose.rotation.z,
                self.current_pose.rotation.w,
            ]
        )
        r_target = R.from_quat(
            [
                self.target_pose.pose.orientation.x,
                self.target_pose.pose.orientation.y,
                self.target_pose.pose.orientation.z,
                self.target_pose.pose.orientation.w,
            ]
        )
        r_err = r_target * r_curr.inv()
        rot_vec = r_err.as_rotvec()
        v_angular = self.Kp_rot @ rot_vec

        return np.concatenate([v_linear, v_angular])

    def _apply_intent_amplification(self, wrench: np.ndarray) -> np.ndarray:
        """Nonlinear force shaping: amplify small forces, compress large forces."""
        out = wrench.copy()
        f_mag = np.linalg.norm(wrench[:3])
        if f_mag > 1e-9:
            out[:3] = self.intent_sat_force * np.tanh(f_mag / self.intent_ref_force) * (wrench[:3] / f_mag)
        t_mag = np.linalg.norm(wrench[3:])
        if t_mag > 1e-9:
            out[3:] = self.intent_sat_torque * np.tanh(t_mag / self.intent_ref_torque) * (wrench[3:] / t_mag)
        return out

    def compute_admittance_velocity(self, guiding: bool = False) -> np.ndarray:
        # Total wrench = sensor + external (both in wrench_frame / tool0)
        total_wrench_tool = self.current_wrench + self.external_wrench
        # Transform to base_link so integrator is frame-consistent
        total_wrench = self._transform_wrench_to_base(total_wrench_tool)
        target_wrench_base = self._transform_wrench_to_base(self.target_wrench)

        # Raw net force (for contact detection & adaptive damping gain)
        f_input_raw = total_wrench - target_wrench_base
        force_mag = np.linalg.norm(f_input_raw[:3])
        torque_mag = np.linalg.norm(f_input_raw[3:])

        # Intent amplification: reshape force for dynamics
        f_input = self._apply_intent_amplification(f_input_raw)

        # Adaptive damping: D_eff = D0 + k * |F_raw|
        D_eff = self.D.copy()
        D_eff[0, 0] += self.adaptive_damping_k_lin * force_mag
        D_eff[1, 1] += self.adaptive_damping_k_lin * force_mag
        D_eff[2, 2] += self.adaptive_damping_k_lin * force_mag
        D_eff[3, 3] += self.adaptive_damping_k_ang * torque_mag
        D_eff[4, 4] += self.adaptive_damping_k_ang * torque_mag
        D_eff[5, 5] += self.adaptive_damping_k_ang * torque_mag

        # Admittance law with amplified force + adaptive damping
        net = f_input - (D_eff @ self.admittance_velocity)
        a = self.M_inv @ net
        v_contact = self.admittance_velocity + self.dt * a

        # Smooth blend between admittance dynamics and decay based on contact confidence
        if guiding:
            # Classifier guarantees contact — skip blend that fights admittance
            self.admittance_velocity = v_contact
        elif self.stop_on_no_contact:
            # Contact detection uses RAW force (physical threshold, not shaped)
            f_err = f_input_raw[:3]
            t_err = f_input_raw[3:]

            # Contact confidence: 0 = no contact, 1 = clear contact
            contact_mag = max(
                abs(f_err[0]) / self.stop_fx_eps,
                abs(f_err[1]) / self.stop_fy_eps,
                abs(f_err[2]) / self.stop_fz_eps,
                abs(t_err[0]) / self.stop_tx_eps,
                abs(t_err[1]) / self.stop_ty_eps,
                abs(t_err[2]) / self.stop_tz_eps,
            )
            contact_alpha = float(np.clip((contact_mag - 0.5) / 1.0, 0.0, 1.0))

            decay = np.exp(-self.dt / max(self.no_contact_decay_tau, 1e-6))
            v_decay = self.admittance_velocity * decay

            if contact_alpha < 0.01:
                self.admittance_velocity = v_decay  # pure decay, no noise accumulation
            else:
                self.admittance_velocity = contact_alpha * v_contact + (1.0 - contact_alpha) * v_decay

            if np.linalg.norm(self.admittance_velocity) < 0.001:
                self.admittance_velocity[:] = 0.0
        else:
            self.admittance_velocity = v_contact

        # Anti-windup: clamp integrator to max velocity
        lin_speed = np.linalg.norm(self.admittance_velocity[:3])
        if lin_speed > self.max_v:
            self.admittance_velocity[:3] *= self.max_v / lin_speed
        ang_speed = np.linalg.norm(self.admittance_velocity[3:])
        if ang_speed > self.max_w:
            self.admittance_velocity[3:] *= self.max_w / ang_speed

        return self.admittance_velocity

    def enabled_callback(self, msg: Bool):
        prev = self.enabled
        self.enabled = bool(msg.data)

        if prev and (not self.enabled):
            # Reset all state immediately on disable
            self.admittance_velocity[:] = 0.0
            self.smoothed_admittance_velocity[:] = 0.0
            self._raw_wrench_lp[:] = 0.0
            self.current_wrench[:] = 0.0
            self.wrench_bias[:] = 0.0
            self._last_v_cmd[:] = 0.0
            self.get_logger().info("Hybrid controller DISABLED — state reset.")

        if (not prev) and self.enabled:
            self.admittance_velocity[:] = 0.0
            self.smoothed_admittance_velocity[:] = 0.0
            self._raw_wrench_lp[:] = 0.0
            self.current_wrench[:] = 0.0
            self.wrench_bias[:] = 0.0
            self._last_v_cmd[:] = 0.0
            self._resume_until = self.get_clock().now().nanoseconds * 1e-9 + 1.0
            self.get_logger().info("Hybrid controller ENABLED — state reset, 1s guard active.")

            # External tare (ft_zeroer/tare) already zeroed the sensor;
            # wrench_bias is already reset to 0 above — skip bias init.
            self._have_bias = True
    def proximity_cb(self, msg):
        self.proximity_scale = msg.data

    def _external_wrench_cb(self, msg: WrenchStamped):
        """External wrench (from force_gui) in base_link frame.
        Transform to wrench_frame (tool0) so it can be added to sensor wrench."""
        src = msg.header.frame_id.strip() or self.base_frame
        try:
            tf = self.tf_buffer.lookup_transform(self.wrench_frame, src, rclpy.time.Time())
        except tf2_ros.TransformException:
            return

        f_in = Vector3Stamped()
        f_in.header = msg.header
        f_in.vector = msg.wrench.force

        t_in = Vector3Stamped()
        t_in.header = msg.header
        t_in.vector = msg.wrench.torque

        f_out = do_transform_vector3(f_in, tf).vector
        t_out = do_transform_vector3(t_in, tf).vector

        self.external_wrench = np.array(
            [f_out.x, f_out.y, f_out.z, t_out.x, t_out.y, t_out.z], dtype=float)

    def _interaction_state_cb(self, msg: String):
        self._interaction_state = msg.data.strip().upper()

    def _snap_target_to_current(self):
        """Update local target_pose to current EE pose (no publish, no log).
        Keeps position controller aligned during GUIDING so it never fights."""
        if self.current_pose is None:
            return
        if self.target_pose is None:
            self.target_pose = PoseStamped()
            self.target_pose.header.frame_id = self.base_frame
        self.target_pose.pose.position.x = self.current_pose.translation.x
        self.target_pose.pose.position.y = self.current_pose.translation.y
        self.target_pose.pose.position.z = self.current_pose.translation.z
        self.target_pose.pose.orientation = self.current_pose.rotation

    def _publish_current_pose_as_target(self):
        """Publish current EE pose to update_target so target integrator also snaps."""
        if self.current_pose is None:
            return
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.pose.position.x = self.current_pose.translation.x
        msg.pose.position.y = self.current_pose.translation.y
        msg.pose.position.z = self.current_pose.translation.z
        msg.pose.orientation = self.current_pose.rotation
        self._update_target_pub.publish(msg)
        self.get_logger().info("GUIDING ended — target snapped to current EE pose.")

    def _rotate_twist_to_base(self, v_tool: np.ndarray) -> np.ndarray:
        """Rotate a 6D twist from wrench_frame (tool0) into base_link frame.

        Only rotates — no translation offset — because twist is a velocity, not a position.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.wrench_frame, rclpy.time.Time())
            q = tf.transform.rotation
            rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()  # 3x3
        except tf2_ros.TransformException:
            return v_tool  # fallback: pass through unrotated

        v_out = np.zeros(6, dtype=float)
        v_out[:3] = rot @ v_tool[:3]
        v_out[3:] = rot @ v_tool[3:]
        return v_out

    def _transform_wrench_to_base(self, w_tool: np.ndarray) -> np.ndarray:
        """Rotate 6D wrench from wrench_frame (tool0) to base_link."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.wrench_frame, rclpy.time.Time())
            q = tf.transform.rotation
            rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        except tf2_ros.TransformException:
            return w_tool
        w_out = np.zeros(6, dtype=float)
        w_out[:3] = rot @ w_tool[:3]
        w_out[3:] = rot @ w_tool[3:]
        return w_out

    # ----------------- Main loop -----------------

    def control_loop(self):
        if not self.get_robot_state():
            return
        
        if not self.enabled:
            self.admittance_velocity[:] = 0.0  # keep it pinned at zero
            self.smoothed_admittance_velocity[:] = 0.0
            self._raw_wrench_lp[:] = 0.0
            self.current_wrench[:] = 0.0

            stop = TwistStamped()
            stop.header.stamp = self.get_clock().now().to_msg()
            stop.header.frame_id = self.base_frame
            # all twist components default to 0.0
            self.twist_pub.publish(stop)
            return
        
        now = self.get_clock().now().nanoseconds * 1e-9
        if now < self._resume_until:
        # only allow resume if contact is released
            self.admittance_velocity[:] = 0.0
            stop = TwistStamped()
            stop.header.stamp = self.get_clock().now().to_msg()
            stop.header.frame_id = self.base_frame
            self.twist_pub.publish(stop)
            return

        v_pos = self.compute_position_velocity()

        # --- Determine effective interaction state ---
        # If external wrench is active (from force_gui), force GUIDING
        # regardless of interaction_classifier topic race
        has_external = np.linalg.norm(self.external_wrench[:3]) > 0.5
        raw_state = "GUIDING" if has_external else self._interaction_state

        # Debounce: require sustained non-GUIDING before confirming exit
        if raw_state == "GUIDING":
            self._guiding_exit_count = 0
            effective_state = "GUIDING"
        elif self._was_guiding:
            self._guiding_exit_count += 1
            if self._guiding_exit_count < self._GUIDING_EXIT_THRESH:
                effective_state = "GUIDING"   # stay in GUIDING during debounce
            else:
                effective_state = raw_state   # confirmed exit
        else:
            effective_state = raw_state

        # Publish effective state so proximity_monitor sees consistent GUIDING
        self._effective_state_pub.publish(String(data=effective_state))

        # --- State-driven control ---
        # Always run admittance dynamics to keep the integrator state
        # consistent. The contact blend (inside compute_admittance_velocity)
        # naturally decays velocity when no force is present.
        self.D = self.D_guiding if effective_state == "GUIDING" else self.D_normal
        v_adm = self.compute_admittance_velocity(guiding=(effective_state == "GUIDING"))

        if effective_state == "GUIDING":
            self._was_guiding = True

            # EMA smoothing on admittance output (reduces sensor jitter)
            alpha_out = self.admittance_output_alpha
            self.smoothed_admittance_velocity = (
                alpha_out * v_adm + (1.0 - alpha_out) * self.smoothed_admittance_velocity
            )

            v_cmd = np.zeros(6, dtype=float)
            v_cmd[:3] = self.smoothed_admittance_velocity[:3]  # already base_link
            v_cmd[3:] = v_pos[3:]         # position hold for orientation
            # Continuously snap target to current pose during guiding so
            # position controller never fights admittance if state jitters
            self._snap_target_to_current()

        else:
            # IDLE, NUDGE, or unknown — position hold
            if self._was_guiding:
                self._was_guiding = False
                self._guiding_exit_count = 0
                self._publish_current_pose_as_target()
                self._snap_target_to_current()
                v_pos = self.compute_position_velocity()   # recompute: now ~0
                self.admittance_velocity[:] = 0.0
                self.smoothed_admittance_velocity[:] = 0.0
                self._last_v_cmd[:] = 0.0                  # safe: v_pos also ~0
            v_cmd = v_pos

        # --- NaN guard ---
        # If admittance produces NaN (e.g. near singularity), zero everything
        if np.any(np.isnan(v_cmd)) or np.any(np.isinf(v_cmd)):
            self.get_logger().warn(
                "NaN/Inf detected in v_cmd — zeroing twist and resetting admittance",
                throttle_duration_sec=1.0)
            self.admittance_velocity[:] = 0.0
            v_cmd = np.zeros(6, dtype=float)

        # Acceleration rate limiter (jerk protection)
        # Compare against PREVIOUS iteration's command (not current)
        delta = v_cmd - self._last_v_cmd
        max_lin_change = self.max_lin_accel * self.dt
        max_ang_change = self.max_ang_accel * self.dt

        lin_delta_mag = np.linalg.norm(delta[:3])
        if lin_delta_mag > max_lin_change and lin_delta_mag > 0.0:
            delta[:3] = delta[:3] / lin_delta_mag * max_lin_change

        ang_delta_mag = np.linalg.norm(delta[3:])
        if ang_delta_mag > max_ang_change and ang_delta_mag > 0.0:
            delta[3:] = delta[3:] / ang_delta_mag * max_ang_change

        v_cmd = self._last_v_cmd + delta

        # Store for NEXT iteration (must be AFTER rate limiter reads it)
        self._last_v_cmd = v_cmd.copy()

        # Saturate speeds
        v_lin = v_cmd[:3].copy()
        v_ang = v_cmd[3:].copy()

        lin_speed = np.linalg.norm(v_lin)
        if self.max_v > 0.0 and lin_speed > self.max_v and lin_speed > 0.0:
            v_lin = v_lin / lin_speed * self.max_v

        ang_speed = np.linalg.norm(v_ang)
        if self.max_w > 0.0 and ang_speed > self.max_w and ang_speed > 0.0:
            v_ang = v_ang / ang_speed * self.max_w

        # Publish TwistStamped
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame

        msg.twist.linear.x = float(v_lin[0])
        msg.twist.linear.y = float(v_lin[1])
        msg.twist.linear.z = float(v_lin[2])
        msg.twist.angular.x = float(v_ang[0])
        msg.twist.angular.y = float(v_ang[1])
        msg.twist.angular.z = float(v_ang[2])

        # Scale twist by proximity (skip if external wrench is active —
        # force_gui is a deliberate test input, proximity shouldn't block it)
        if not has_external:
            msg.twist.linear.x *= self.proximity_scale
            msg.twist.linear.y *= self.proximity_scale
            msg.twist.linear.z *= self.proximity_scale
            msg.twist.angular.x *= self.proximity_scale
            msg.twist.angular.y *= self.proximity_scale
            msg.twist.angular.z *= self.proximity_scale

        self.twist_pub.publish(msg)



def main(args=None):
    rclpy.init(args=args)
    node = HybridControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Send one zero twist on shutdown
        try:
            stop = TwistStamped()
            stop.header.stamp = node.get_clock().now().to_msg()
            stop.header.frame_id = node.base_frame
            node.twist_pub.publish(stop)
        except Exception:
            pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
