#!/usr/bin/env python3
"""
ISO/TS 15066 Safety Monitor for UR3e collaborative robot.

Enforces three collaborative methods from ISO/TS 15066:
  1. Hand Guiding     — validate force/speed limits during GUIDING state
  2. SSM              — compute minimum protective distance using ISO formula
  3. PFL              — monitor contact forces against body-region limits

Speed control uses zone-based scaling on /proximity/scale (drop-in
replacement for the old proximity_monitor).  The raw point cloud includes
static workspace geometry (table, walls) that are not humans, so the ISO
SSM formula cannot be applied directly to compute an operational speed
scale.  Instead, SSM metrics are computed and published on /safety/*
topics for ISO compliance logging, while the zone-based scale provides
the actual operational speed control.

Protective stops (PFL violation, d < ssm_min_distance) pause servo and
disable the hybrid controller.
"""

import csv
import datetime
import math
import os

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                       QoSHistoryPolicy, QoSDurabilityPolicy)

from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger
import sensor_msgs_py.point_cloud2 as pc2

from tf2_ros import Buffer, TransformListener, TransformException


class SafetyMonitor(Node):
    def __init__(self):
        super().__init__('safety_monitor')

        # ── parameters ──────────────────────────────────────────────
        self.declare_parameter('rate_hz', 100.0)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tool_frame', 'tool0')

        # Zone-based scaling
        self.declare_parameter('near_dist', 0.25)
        self.declare_parameter('warn_dist', 0.40)
        self.declare_parameter('safe_dist', 0.90)
        self.declare_parameter('crawl_scale', 0.10)
        self.declare_parameter('guiding_min_scale', 0.30)
        self.declare_parameter('scale_smoothing_alpha', 0.3)

        # SSM (ISO/TS 15066)
        self.declare_parameter('v_human', 1.6)
        self.declare_parameter('T_reaction', 0.15)
        self.declare_parameter('T_stop', 0.4)
        self.declare_parameter('C_intrusion', 0.0)
        self.declare_parameter('Z_sensor', 0.04)
        self.declare_parameter('Z_robot', 0.02)
        self.declare_parameter('ssm_min_distance', 0.10)

        # PFL
        self.declare_parameter('pfl_force_limit', 140.0)
        self.declare_parameter('pfl_transient_factor', 2.0)
        self.declare_parameter('pfl_violation_duration', 0.5)

        # Hand guiding
        self.declare_parameter('max_guide_speed', 0.25)
        self.declare_parameter('max_guide_force', 150.0)

        # Protective stop
        self.declare_parameter('resume_hold_s', 1.0)

        # Proximity cloud processing
        self.declare_parameter('monitored_links', [
            'upper_arm_link', 'forearm_link',
            'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'tool0',
        ])
        self.declare_parameter('cloud_subsample', 5)
        self.declare_parameter('min_link_dist', 0.20)
        self.declare_parameter('no_cloud_timeout_s', 2.0)
        self.declare_parameter('cloud_min_height', 0.0)
        self.declare_parameter('cloud_topic', '/cloud_human')

        self._load_params()

        # ── TF ──────────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── state ───────────────────────────────────────────────────
        self.interaction_state = 'IDLE'
        self.current_wrench = [0.0, 0.0, 0.0]
        self.min_cloud_dist = float('inf')
        self.last_cloud_time = self.get_clock().now()
        self.prev_tool_pos = None
        self.prev_tool_stamp = None
        self.tcp_speed = 0.0
        self.smoothed_scale = 1.0

        # PFL violation tracking
        self.pfl_violation_start = None
        self.protective_stopped = False
        self.violation_clear_time = None

        # Experiment metrics accumulation
        self._experiment_dir = os.path.expanduser("~/ros2_ws/experiment_data")
        os.makedirs(self._experiment_dir, exist_ok=True)
        self._trial_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._pfl_violation_count = 0
        self._pfl_violation_total_s = 0.0
        self._pfl_violation_start_accum = None
        self._max_force_over_limit = 0.0
        self._protective_stop_count = 0
        self._prev_protective_stopped = False
        self._min_distance_observed = float('inf')
        self._safety_log = []
        self._loop_count = 0
        self._start_time = self.get_clock().now()

        # ── subscriptions ───────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        cloud_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.create_subscription(
            WrenchStamped, '/wrench_zeroed', self._wrench_cb, sensor_qos)
        cloud_topic = self.get_parameter('cloud_topic').value
        self.create_subscription(
            PointCloud2, cloud_topic, self._cloud_cb, cloud_qos)
        self.get_logger().info(f'Subscribing to cloud topic: {cloud_topic}')
        self.create_subscription(
            JointState, '/joint_states', self._joint_cb, 10)
        self.create_subscription(
            String, '/interaction_state', self._interaction_cb, 10)

        # ── publishers ──────────────────────────────────────────────
        self.pub_scale = self.create_publisher(
            Float32, '/proximity/scale', 10)
        self.pub_min_dist = self.create_publisher(
            Float32, '/safety/min_distance', 10)
        self.pub_pfl_ok = self.create_publisher(
            Bool, '/safety/pfl_ok', 10)
        self.pub_ssm_ok = self.create_publisher(
            Bool, '/safety/ssm_ok', 10)
        self.pub_hand_guiding_ok = self.create_publisher(
            Bool, '/safety/hand_guiding_ok', 10)
        self.pub_status = self.create_publisher(
            String, '/safety/status', 10)
        self.pub_protective_stop = self.create_publisher(
            Bool, '/safety/protective_stop', 10)
        self.pub_enabled = self.create_publisher(
            Bool, '/hybrid_controller/enabled', 10)

        # ── service clients ─────────────────────────────────────────
        self.cli_pause = self.create_client(
            Trigger, '/servo_node/pause_servo')
        self.cli_unpause = self.create_client(
            Trigger, '/servo_node/unpause_servo')

        # ── main timer ──────────────────────────────────────────────
        rate = self.get_parameter('rate_hz').value
        self.create_timer(1.0 / rate, self._control_loop)
        self.create_timer(0.5, self._cloud_timeout_check)

        self.get_logger().info(
            f'ISO/TS 15066 Safety Monitor started @ {rate:.0f} Hz  '
            f'PFL={self.pfl_force_limit:.0f}N  '
            f'zones: STOP<{self.near_dist}m CRAWL<{self.warn_dist}m '
            f'SLOW<{self.safe_dist}m')

    # ── parameter loading ───────────────────────────────────────────
    def _load_params(self):
        g = self.get_parameter
        self.base_frame = g('base_frame').value
        self.tool_frame = g('tool_frame').value

        # Zone scaling
        self.near_dist = g('near_dist').value
        self.warn_dist = g('warn_dist').value
        self.safe_dist = g('safe_dist').value
        self.crawl_scale = g('crawl_scale').value
        self.guiding_min_scale = g('guiding_min_scale').value
        self.alpha = g('scale_smoothing_alpha').value

        # SSM
        self.v_human = g('v_human').value
        self.T_reaction = g('T_reaction').value
        self.T_stop = g('T_stop').value
        self.C_intrusion = g('C_intrusion').value
        self.Z_sensor = g('Z_sensor').value
        self.Z_robot = g('Z_robot').value
        self.ssm_min_distance = g('ssm_min_distance').value

        # PFL
        self.pfl_force_limit = g('pfl_force_limit').value
        self.pfl_transient_factor = g('pfl_transient_factor').value
        self.pfl_violation_duration = g('pfl_violation_duration').value
        self.pfl_transient_limit = (
            self.pfl_force_limit * self.pfl_transient_factor)

        # Hand guiding
        self.max_guide_speed = g('max_guide_speed').value
        self.max_guide_force = g('max_guide_force').value

        self.resume_hold_s = g('resume_hold_s').value

        self.monitored_links = list(g('monitored_links').value)
        self.cloud_subsample = g('cloud_subsample').value
        self.min_link_dist = g('min_link_dist').value
        self.no_cloud_timeout = g('no_cloud_timeout_s').value
        self.cloud_min_height = g('cloud_min_height').value

    # ── subscription callbacks ──────────────────────────────────────
    def _wrench_cb(self, msg: WrenchStamped):
        f = msg.wrench.force
        self.current_wrench = [f.x, f.y, f.z]

    def _cloud_cb(self, msg: PointCloud2):
        """Compute minimum distance from monitored links to point cloud."""
        self.last_cloud_time = self.get_clock().now()
        cloud_frame = msg.header.frame_id

        link_positions = []
        now = rclpy.time.Time()
        for link in self.monitored_links:
            try:
                tf = self.tf_buffer.lookup_transform(cloud_frame, link, now)
                t = tf.transform.translation
                link_positions.append([t.x, t.y, t.z])
            except TransformException:
                pass

        if not link_positions:
            return

        link_positions = np.array(link_positions, dtype=np.float32)

        gen = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        points = np.array([[p[0], p[1], p[2]] for p in gen], dtype=np.float32)

        if len(points) == 0:
            self.min_cloud_dist = float('inf')
            return

        if self.cloud_subsample > 1:
            points = points[::self.cloud_subsample]

        # Height filter: remove table surface / floor points below
        # cloud_min_height in the base_link frame.  Compute the Z-axis of
        # base_link expressed in the cloud frame and use a dot-product to
        # obtain each point's height efficiently (no full transform needed).
        try:
            tf_base = self.tf_buffer.lookup_transform(
                cloud_frame, self.base_frame, now)
            q = tf_base.transform.rotation
            t = tf_base.transform.translation
            # Z-axis of base_link in cloud_frame (third column of rotation matrix)
            up = np.array([
                2.0 * (q.x * q.z + q.y * q.w),
                2.0 * (q.y * q.z - q.x * q.w),
                1.0 - 2.0 * (q.x * q.x + q.y * q.y),
            ], dtype=np.float32)
            base_origin = np.array([t.x, t.y, t.z], dtype=np.float32)
            heights = (points - base_origin) @ up
            height_mask = heights >= self.cloud_min_height
            points = points[height_mask]
            if len(points) == 0:
                self.min_cloud_dist = float('inf')
                return
        except TransformException:
            pass  # skip height filter if TF not available

        diff = points[np.newaxis, :, :] - link_positions[:, np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        min_dist_to_any_link = dists.min(axis=0)
        valid_mask = min_dist_to_any_link >= self.min_link_dist
        dists_filtered = dists[:, valid_mask]

        if dists_filtered.shape[1] == 0:
            self.min_cloud_dist = float('inf')
        else:
            per_link_min = dists_filtered.min(axis=1)
            self.min_cloud_dist = float(per_link_min.min())

    def _joint_cb(self, msg: JointState):
        """Compute TCP speed via TF differentiation."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.tool_frame, rclpy.time.Time())
            t = tf.transform.translation
            pos = np.array([t.x, t.y, t.z])
            stamp = self.get_clock().now()

            if self.prev_tool_pos is not None and self.prev_tool_stamp is not None:
                dt = (stamp - self.prev_tool_stamp).nanoseconds / 1e9
                if dt > 1e-6:
                    self.tcp_speed = float(
                        np.linalg.norm(pos - self.prev_tool_pos) / dt)

            self.prev_tool_pos = pos
            self.prev_tool_stamp = stamp
        except TransformException:
            pass

    def _interaction_cb(self, msg: String):
        self.interaction_state = msg.data

    # ── zone-based scaling (operational) ────────────────────────────
    def _zone_scale(self, dist: float) -> float:
        """Graduated speed scale from distance, same as old proximity_monitor."""
        if dist <= self.near_dist:
            return 0.0
        elif dist <= self.warn_dist:
            return self.crawl_scale
        elif dist <= self.safe_dist:
            t = (dist - self.warn_dist) / (self.safe_dist - self.warn_dist)
            return self.crawl_scale + t * (1.0 - self.crawl_scale)
        else:
            return 1.0

    # ── ISO SSM computation (reporting) ─────────────────────────────
    def _compute_ssm(self, v_r: float, d: float):
        """
        ISO/TS 15066 Speed and Separation Monitoring.

        S_p = S_h + S_r + S_s + C + Z_d + Z_r

        Returns (ssm_ok, S_p).
        ssm_ok is False only when d < ssm_min_distance (imminent collision).
        """
        T_R = self.T_reaction
        T_S = self.T_stop
        S_h = self.v_human * (T_R + T_S)
        S_r = v_r * T_R
        S_s = v_r * T_S / 2.0
        S_p = S_h + S_r + S_s + self.C_intrusion + self.Z_sensor + self.Z_robot

        ssm_ok = d > self.ssm_min_distance
        return ssm_ok, S_p

    # ── PFL check ───────────────────────────────────────────────────
    def _check_pfl(self, force_mag: float):
        """
        ISO/TS 15066 Power and Force Limiting.
        Returns (pfl_ok, sustained_violation).
        """
        now = self.get_clock().now()

        if force_mag > self.pfl_force_limit:
            if self.pfl_violation_start is None:
                self.pfl_violation_start = now
            elapsed = (now - self.pfl_violation_start).nanoseconds / 1e9
            sustained = elapsed >= self.pfl_violation_duration
            return False, sustained
        else:
            self.pfl_violation_start = None
            return True, False

    # ── hand guiding check ──────────────────────────────────────────
    def _check_hand_guiding(self, v_r: float, force_mag: float):
        """ISO 10218 Hand Guiding limits."""
        if self.interaction_state != 'GUIDING':
            return True
        return v_r <= self.max_guide_speed and force_mag <= self.max_guide_force

    # ── main control loop ───────────────────────────────────────────
    def _control_loop(self):
        now = self.get_clock().now()
        v_r = self.tcp_speed
        d = self.min_cloud_dist
        force_mag = math.sqrt(sum(f * f for f in self.current_wrench))

        # 1) Zone-based speed scale (operational output)
        raw_scale = self._zone_scale(d)

        # Interaction-state override: keep robot responsive during guiding.
        # During GUIDING the human is intentionally close, so bypass near-zone
        # full stop.  Only ssm_min_distance (absolute safety floor) can stop.
        if self.interaction_state == 'GUIDING' and d > self.ssm_min_distance:
            raw_scale = max(raw_scale, self.guiding_min_scale)

        # Asymmetric EMA smoothing (fast decrease, slow increase)
        if raw_scale < self.smoothed_scale:
            a = 0.8
        else:
            a = self.alpha
        self.smoothed_scale = a * raw_scale + (1.0 - a) * self.smoothed_scale

        # 2) ISO SSM (reported for compliance logging)
        ssm_ok, S_p = self._compute_ssm(v_r, d)

        # 3) PFL
        pfl_ok, pfl_sustained = self._check_pfl(force_mag)

        # 4) Hand Guiding
        hg_ok = self._check_hand_guiding(v_r, force_mag)

        # Combine: hand guiding violation reduces scale
        scale = self.smoothed_scale
        if self.interaction_state == 'GUIDING' and not hg_ok:
            scale = min(scale, 0.1)

        # Determine overall status and protective stop
        need_stop = False
        if pfl_sustained:
            status = 'PROTECTIVE_STOP'
            scale = 0.0
            need_stop = True
        elif not pfl_ok:
            status = 'PFL_VIOLATION'
            scale = 0.0
            need_stop = True
        elif not ssm_ok:
            # d < ssm_min_distance — imminent collision
            status = 'PROTECTIVE_STOP'
            scale = 0.0
            need_stop = True
        elif not hg_ok:
            status = 'HG_WARNING'
        elif d <= self.near_dist:
            status = 'NEAR_GUIDING' if self.interaction_state == 'GUIDING' else 'NEAR_STOP'
        elif d <= self.warn_dist:
            status = 'WARN_CRAWL'
        elif d <= self.safe_dist:
            status = 'SLOW'
        else:
            status = 'OK'

        # Protective stop logic
        if need_stop and not self.protective_stopped:
            self.protective_stopped = True
            self.violation_clear_time = None
            self._trigger_stop()
        elif not need_stop and self.protective_stopped:
            if self.violation_clear_time is None:
                self.violation_clear_time = now
            elapsed = (now - self.violation_clear_time).nanoseconds / 1e9
            if elapsed >= self.resume_hold_s:
                self.protective_stopped = False
                self.violation_clear_time = None
                self._trigger_resume()
            else:
                scale = 0.0
                status = 'PROTECTIVE_STOP'

        if self.protective_stopped:
            scale = 0.0

        # Publish
        self._publish_float(self.pub_scale, scale)
        self._publish_float(self.pub_min_dist, d if d != float('inf') else -1.0)
        self._publish_bool(self.pub_pfl_ok, pfl_ok)
        self._publish_bool(self.pub_ssm_ok, ssm_ok)
        self._publish_bool(self.pub_hand_guiding_ok, hg_ok)
        self._publish_bool(self.pub_protective_stop, self.protective_stopped)
        self._publish_bool(self.pub_enabled, not self.protective_stopped)

        msg = String()
        msg.data = status
        self.pub_status.publish(msg)

        self.get_logger().info(
            f'[{status}] scale={scale:.2f} d={d:.2f}m v_r={v_r:.3f}m/s '
            f'F={force_mag:.1f}N S_p={S_p:.3f}m ssm={ssm_ok} '
            f'[{self.interaction_state}]',
            throttle_duration_sec=1.0)

        # ── metrics accumulation ─────────────────────────────────────
        # PFL violation tracking (start/end, count, duration)
        if not pfl_ok:
            if self._pfl_violation_start_accum is None:
                self._pfl_violation_start_accum = now
                self._pfl_violation_count += 1
            excess = force_mag - self.pfl_force_limit
            if excess > self._max_force_over_limit:
                self._max_force_over_limit = excess
        else:
            if self._pfl_violation_start_accum is not None:
                dur = (now - self._pfl_violation_start_accum).nanoseconds / 1e9
                self._pfl_violation_total_s += dur
                self._pfl_violation_start_accum = None

        # Protective stop rising edge
        if self.protective_stopped and not self._prev_protective_stopped:
            self._protective_stop_count += 1
        self._prev_protective_stopped = self.protective_stopped

        # Min distance
        if d < self._min_distance_observed and d != float('inf'):
            self._min_distance_observed = d

        # Downsample timeseries to ~10 Hz (every 10th iteration at 100 Hz)
        self._loop_count += 1
        if self._loop_count % 10 == 0:
            t_s = (now - self._start_time).nanoseconds / 1e9
            self._safety_log.append((
                t_s, force_mag, pfl_ok,
                d if d != float('inf') else -1.0, status))

    # ── helpers ─────────────────────────────────────────────────────
    def _trigger_stop(self):
        self.get_logger().warn('PROTECTIVE STOP triggered')
        self._publish_bool(self.pub_enabled, False)
        if self.cli_pause.service_is_ready():
            self.cli_pause.call_async(Trigger.Request())
        else:
            self.get_logger().warn('pause_servo service not ready')

    def _trigger_resume(self):
        self.get_logger().info('Protective stop cleared — resuming')
        self._publish_bool(self.pub_enabled, True)
        if self.cli_unpause.service_is_ready():
            self.cli_unpause.call_async(Trigger.Request())
        else:
            self.get_logger().warn('unpause_servo service not ready')

    def _publish_float(self, pub, value: float):
        msg = Float32()
        msg.data = float(value)
        pub.publish(msg)

    def _publish_bool(self, pub, value: bool):
        msg = Bool()
        msg.data = bool(value)
        pub.publish(msg)

    def _cloud_timeout_check(self):
        elapsed = (self.get_clock().now() - self.last_cloud_time).nanoseconds / 1e9
        if elapsed > self.no_cloud_timeout:
            self.smoothed_scale = 0.0
            self._publish_float(self.pub_scale, 0.0)
            self.get_logger().warn(
                f'No cloud for {elapsed:.1f}s — scale forced to 0.0',
                throttle_duration_sec=2.0)


    # ── CSV export for thesis metrics ──────────────────────────────
    def _export_safety_csv(self):
        tid = self._trial_id
        d = self._experiment_dir
        log = self.get_logger().info

        # Close any open PFL violation
        if self._pfl_violation_start_accum is not None:
            dur = (self.get_clock().now()
                   - self._pfl_violation_start_accum).nanoseconds / 1e9
            self._pfl_violation_total_s += dur
            self._pfl_violation_start_accum = None

        total_samples = self._loop_count // 10  # downsampled count
        pfl_ok_count = sum(1 for _, _, ok, _, _ in self._safety_log if ok)
        compliance_pct = (
            (pfl_ok_count / total_samples * 100.0)
            if total_samples > 0 else 100.0)

        min_dist = (self._min_distance_observed
                    if self._min_distance_observed != float('inf')
                    else -1.0)

        # safety_summary
        path = os.path.join(d, f"safety_summary_{tid}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "trial_id", "pfl_violation_count",
                "pfl_violation_total_s", "max_force_over_limit_N",
                "pfl_compliance_pct", "min_distance_m",
                "protective_stop_count",
            ])
            w.writerow([
                tid, self._pfl_violation_count,
                f"{self._pfl_violation_total_s:.3f}",
                f"{self._max_force_over_limit:.2f}",
                f"{compliance_pct:.1f}",
                f"{min_dist:.3f}",
                self._protective_stop_count,
            ])
        log(f"CSV written: {path}")

        # safety_timeseries
        path = os.path.join(d, f"safety_timeseries_{tid}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "force_N", "pfl_ok", "distance_m", "status"])
            for t_s, force, ok, dist, status in self._safety_log:
                w.writerow([
                    f"{t_s:.3f}", f"{force:.2f}", str(ok),
                    f"{dist:.3f}", status,
                ])
        log(f"CSV written: {path}")

    def destroy_node(self):
        try:
            self._export_safety_csv()
        except Exception as e:
            self.get_logger().warn(f"Failed to export safety CSV: {e}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SafetyMonitor()
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
