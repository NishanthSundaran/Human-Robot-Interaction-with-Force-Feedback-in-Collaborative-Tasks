#!/usr/bin/env python3
"""
Hand-Eye Calibration Node (eye-to-hand).

Camera is FIXED, robot holds/has the ChArUco board attached to tool0.
Collects pose pairs (board-in-camera + tool0-in-base_link), then solves
for the camera-to-base_link transform.

Usage:
  1. Tape the ChArUco board flat onto the gripper / a plate on tool0
  2. Launch the robot driver + RealSense camera
  3. Run this node
  4. Put robot in freedrive (teach pendant)
  5. Move robot to different poses — keep board visible to camera
  6. Press 'c' to capture a sample (need 12-15 samples)
  7. Press 's' to solve calibration
  8. Press 'q' to quit

The node outputs the camera-to-base_link static TF command.
"""

import math
import cv2
import numpy as np
from scipy.optimize import least_squares as scipy_least_squares

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
import tf2_ros


# ---------------------------------------------------------------------------
# Board parameters — CHANGE THESE to match your board
# ---------------------------------------------------------------------------
SQUARES_X = 8
SQUARES_Y = 11
SQUARE_LENGTH = 0.029     # 29mm
MARKER_LENGTH = 0.021     # 21mm
DICT_ID = cv2.aruco.DICT_5X5_50

# Frames
BASE_FRAME = "base_link"
EE_FRAME = "tool0"


def _rodrigues_to_matrix(rvec):
    R, _ = cv2.Rodrigues(rvec)
    return R


def _tf_to_R_t(tf_msg):
    """Extract R (3x3) and t (3x1) from a geometry_msgs TransformStamped."""
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])
    tvec = np.array([[t.x], [t.y], [t.z]])
    return R, tvec


def _R_t_to_quaternion(R):
    """Rotation matrix -> quaternion (x, y, z, w)."""
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


def _invert_transform(R, t):
    """Invert a rigid transform (R, t) -> (R^T, -R^T @ t)."""
    R_inv = R.T
    t_inv = -R.T @ t
    return R_inv, t_inv


def _compose_transforms(R1, t1, R2, t2):
    """Compose two transforms: T1 * T2."""
    R = R1 @ R2
    t = R1 @ t2 + t1
    return R, t


def _build_marker_id_to_colrow(sx, sy, legacy):
    """Map marker ID -> (col, row) for a ChArUco board layout."""
    mapping = {}
    mid = 0
    for row in range(sy):
        for col in range(sx):
            if legacy:
                is_marker = (col + row) % 2 == 0
            else:
                is_marker = (col + row) % 2 == 1
            if is_marker:
                mapping[mid] = (col, row)
                mid += 1
    return mapping


class HandeyeCalibrationNode(Node):
    def __init__(self):
        super().__init__("handeye_calibration")

        self.declare_parameter("rgb_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_info_topic",
                               "/camera/camera/color/camera_info")
        self._rgb_topic = self.get_parameter("rgb_topic").value
        self._info_topic = self.get_parameter("camera_info_topic").value

        self._bridge = CvBridge()
        self._K = None
        self._dist = None
        self._rgb = None

        # ArUco detector for marker detection
        self._dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)
        self._detector = cv2.aruco.ArucoDetector(self._dictionary)

        # ChArUco board variants: 2 orientations x 2 patterns (legacy/new)
        self._board_variants = []
        for sx, sy in [(SQUARES_X, SQUARES_Y), (SQUARES_Y, SQUARES_X)]:
            for legacy in [False, True]:
                board = cv2.aruco.CharucoBoard(
                    (sx, sy), SQUARE_LENGTH, MARKER_LENGTH, self._dictionary)
                if legacy:
                    board.setLegacyPattern(True)
                det = cv2.aruco.CharucoDetector(board)
                label = f"{sx}x{sy}" + (" legacy" if legacy else "")
                self._board_variants.append((board, det, label))

        # ArUco fallback: precompute marker 3D positions for all variants
        self._aruco_variants = []
        for sx, sy in [(SQUARES_X, SQUARES_Y), (SQUARES_Y, SQUARES_X)]:
            for legacy in [False, True]:
                mapping = _build_marker_id_to_colrow(sx, sy, legacy)
                label = f"{sx}x{sy}" + (" legacy" if legacy else "")
                self._aruco_variants.append((mapping, label))

        # Cached detection from display loop (so 'c' uses same frame)
        self._last_detection = None
        self._last_frame = None

        # Collected samples for calibration
        # Board poses in camera frame: p_cam = R * p_target + t
        self._R_target2cam_list = []
        self._t_target2cam_list = []
        # Gripper-to-base from TF: p_base = R * p_gripper + t
        self._R_gripper2base_list = []
        self._t_gripper2base_list = []

        # TF
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Subscriptions
        self.create_subscription(
            CameraInfo, self._info_topic, self._info_cb, 10)
        self.create_subscription(
            Image, self._rgb_topic, self._rgb_cb, 10)

        # Timer for display loop (30 Hz)
        self._timer = self.create_timer(1.0 / 30.0, self._display_loop)

        self.get_logger().info(
            f"Hand-eye calibration node started.\n"
            f"  Camera: {self._rgb_topic}\n"
            f"  Board:  {SQUARES_X}x{SQUARES_Y}, "
            f"square={SQUARE_LENGTH*1000:.0f}mm, "
            f"marker={MARKER_LENGTH*1000:.0f}mm\n"
            f"  Controls: 'c'=capture, 's'=solve, 'q'=quit")

    def _info_cb(self, msg: CameraInfo):
        self._K = np.array(msg.k).reshape(3, 3)
        self._dist = np.array(msg.d)

    def _rgb_cb(self, msg: Image):
        try:
            self._rgb = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def _aruco_board_pose(self, marker_corners, marker_ids):
        """Estimate board pose from ArUco marker corners directly."""
        if self._K is None or marker_ids is None:
            return None

        ids_flat = marker_ids.flatten()
        best_err = float('inf')
        best_result = None

        for mapping, label in self._aruco_variants:
            obj_pts_list = []
            img_pts_list = []

            for i, mid_val in enumerate(ids_flat):
                mid_int = int(mid_val)
                if mid_int not in mapping:
                    continue
                col, row = mapping[mid_int]

                cx = col * SQUARE_LENGTH + SQUARE_LENGTH / 2.0
                cy = row * SQUARE_LENGTH + SQUARE_LENGTH / 2.0
                half = MARKER_LENGTH / 2.0

                obj_pts = np.array([
                    [cx - half, cy - half, 0.0],
                    [cx + half, cy - half, 0.0],
                    [cx + half, cy + half, 0.0],
                    [cx - half, cy + half, 0.0],
                ], dtype=np.float64)

                img_pts = marker_corners[i].reshape(4, 2).astype(np.float64)
                obj_pts_list.append(obj_pts)
                img_pts_list.append(img_pts)

            if len(obj_pts_list) < 4:
                continue

            obj_all = np.vstack(obj_pts_list)
            img_all = np.vstack(img_pts_list)

            ok, rvec, tvec = cv2.solvePnP(
                obj_all, img_all, self._K, self._dist,
                flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue

            reproj, _ = cv2.projectPoints(
                obj_all, rvec, tvec, self._K, self._dist)
            err = np.mean(np.linalg.norm(
                img_all - reproj.reshape(-1, 2), axis=1))

            if err < best_err:
                best_err = err
                best_result = (rvec, tvec, label, len(obj_pts_list), err)

        return best_result

    def _detect_board(self, image):
        """Detect board and estimate pose."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        vis = image.copy()

        marker_corners, marker_ids, _ = self._detector.detectMarkers(gray)
        n_markers = len(marker_ids) if marker_ids is not None else 0
        if marker_corners is not None:
            cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)

        if n_markers < 4:
            cv2.putText(vis, f"ArUco: {n_markers} (need 4+)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)
            return None, vis

        # --- Path 1: ChArUco corner detection ---
        best_cc, best_ci, best_n = None, None, 0
        best_label, best_board = "", None
        charuco_debug = []
        for board, det, label in self._board_variants:
            cc, ci, mc, mi = det.detectBoard(gray)
            n = len(cc) if cc is not None else 0
            charuco_debug.append(f"{label}:{n}")
            if n > best_n:
                best_cc, best_ci, best_n = cc, ci, n
                best_label, best_board = label, board

        if best_n >= 6 and self._K is not None:
            obj_pts, img_pts = best_board.matchImagePoints(
                best_cc, best_ci)
            if obj_pts is not None and len(obj_pts) >= 6:
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, self._K, self._dist)
                if ok:
                    cv2.aruco.drawDetectedCornersCharuco(
                        vis, best_cc, best_ci)
                    cv2.drawFrameAxes(
                        vis, self._K, self._dist, rvec, tvec, 0.03)
                    cv2.putText(
                        vis,
                        f"ArUco: {n_markers} | ChArUco: {best_n} "
                        f"({best_label}) — READY",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
                    return {
                        "type": "charuco",
                        "rvec": rvec,
                        "tvec": tvec,
                        "label": best_label,
                    }, vis

        # --- Path 2: ArUco marker fallback ---
        aruco_result = self._aruco_board_pose(marker_corners, marker_ids)
        if aruco_result is not None:
            rvec, tvec, ar_label, n_matched, reproj_err = aruco_result
            cv2.drawFrameAxes(
                vis, self._K, self._dist, rvec, tvec, 0.03)
            debug_str = " | ".join(charuco_debug)
            cv2.putText(
                vis,
                f"ArUco fallback: {n_matched} markers, "
                f"err={reproj_err:.1f}px ({ar_label}) — READY",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"ChArUco tried: [{debug_str}]",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 165, 255), 1)
            return {
                "type": "aruco",
                "rvec": rvec,
                "tvec": tvec,
                "label": ar_label,
            }, vis

        debug_str = " | ".join(charuco_debug)
        cv2.putText(vis,
                    f"ArUco: {n_markers} | [{debug_str}] — no pose",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 165, 255), 2)
        return None, vis

    def _get_robot_pose(self):
        """Get T_gripper2base (tool0 -> base_link) from TF.

        Returns (R, t) where p_base = R * p_gripper + t,  or None.
        """
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                BASE_FRAME, EE_FRAME, rclpy.time.Time())
            return _tf_to_R_t(tf_msg)
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return None

    def _capture_sample(self):
        """Capture one calibration sample using cached detection."""
        if self._last_detection is None:
            self.get_logger().warn(
                "Board not detected — hold board steady in view, "
                "wait for green 'READY' text, then press 'c'.")
            return False

        det = self._last_detection
        rvec = det["rvec"]
        tvec = det["tvec"]
        det_type = det["type"]
        det_label = det["label"]

        robot = self._get_robot_pose()
        if robot is None:
            return False

        R_target2cam = _rodrigues_to_matrix(rvec)
        t_target2cam = tvec.reshape(3, 1)
        R_gripper2base, t_gripper2base = robot

        self._R_target2cam_list.append(R_target2cam)
        self._t_target2cam_list.append(t_target2cam)
        self._R_gripper2base_list.append(R_gripper2base)
        self._t_gripper2base_list.append(t_gripper2base)

        n = len(self._R_target2cam_list)
        self.get_logger().info(
            f"Sample {n} captured ({det_type}, {det_label})! "
            f"Board at cam: t=({tvec[0,0]:.3f}, {tvec[1,0]:.3f}, "
            f"{tvec[2,0]:.3f})")
        if n < 12:
            self.get_logger().info(
                f"  Need at least 12 samples. Move to a different pose.")
        else:
            self.get_logger().info(
                f"  {n} samples collected. Press 's' to solve or keep adding.")
        return True

    def _compute_cam2base_from_X(self, R_X, t_X):
        """Given X = T_gripper2target from calibrateHandEye, compute T_cam2base.

        For eye-to-hand:
          T_base2cam = T_target2cam_i * T_gripper2target * T_base2gripper_i
        This should be constant across samples (camera is fixed).

        Returns (R_cam2base, t_cam2base, consistency_mm) or None.
        """
        n = len(self._R_target2cam_list)
        cam_positions = []   # camera position in base frame per sample

        for i in range(n):
            R_t2c = self._R_target2cam_list[i]
            t_t2c = self._t_target2cam_list[i]
            R_g2b = self._R_gripper2base_list[i]
            t_g2b = self._t_gripper2base_list[i]

            # T_base2gripper = inv(T_gripper2base)
            R_b2g, t_b2g = _invert_transform(R_g2b, t_g2b)

            # T_base2cam = T_target2cam * X * T_base2gripper
            # Step 1: X * T_base2gripper
            R_xb, t_xb = _compose_transforms(R_X, t_X, R_b2g, t_b2g)
            # Step 2: T_target2cam * (X * T_base2gripper)
            R_b2c, t_b2c = _compose_transforms(R_t2c, t_t2c, R_xb, t_xb)

            # T_cam2base = inv(T_base2cam)
            R_c2b, t_c2b = _invert_transform(R_b2c, t_b2c)
            cam_positions.append(t_c2b.flatten())

        pos_arr = np.array(cam_positions)
        consistency = np.mean(np.std(pos_arr, axis=0)) * 1000.0  # mm

        # Average camera position
        t_c2b_avg = np.mean(pos_arr, axis=0).reshape(3, 1)

        # Use median sample's rotation (more robust than first)
        dists_from_mean = np.linalg.norm(pos_arr - t_c2b_avg.flatten(), axis=1)
        median_idx = np.argmin(dists_from_mean)

        R_g2b_med = self._R_gripper2base_list[median_idx]
        t_g2b_med = self._t_gripper2base_list[median_idx]
        R_b2g_med, t_b2g_med = _invert_transform(R_g2b_med, t_g2b_med)
        R_xb, t_xb = _compose_transforms(R_X, t_X, R_b2g_med, t_b2g_med)
        R_b2c, t_b2c = _compose_transforms(
            self._R_target2cam_list[median_idx],
            self._t_target2cam_list[median_idx],
            R_xb, t_xb)
        R_c2b_avg, _ = _invert_transform(R_b2c, t_b2c)

        return R_c2b_avg, t_c2b_avg, consistency

    def _solve_calibration(self):
        """Run hand-eye calibration (eye-to-hand) and output result.

        Uses calibrateHandEye with swapped inputs:
        - target2cam in the gripper2base slot
        - gripper2base in the target2cam slot
        This solves for X = T_gripper2target (board-on-gripper relationship).
        Then T_base2cam is computed per-sample and averaged.
        """
        n = len(self._R_target2cam_list)
        if n < 5:
            self.get_logger().error(f"Need at least 5 samples, have {n}.")
            return

        self.get_logger().info(f"Solving with {n} samples...")

        # --- calibrateHandEye methods (eye-to-hand swap) ---
        he_methods = [
            ("Tsai", cv2.CALIB_HAND_EYE_TSAI),
            ("Park", cv2.CALIB_HAND_EYE_PARK),
            ("Horaud", cv2.CALIB_HAND_EYE_HORAUD),
            ("Andreff", cv2.CALIB_HAND_EYE_ANDREFF),
            ("Daniilidis", cv2.CALIB_HAND_EYE_DANIILIDIS),
        ]

        results = []  # (name, R_c2b, t_c2b, consistency_mm)

        for name, method in he_methods:
            try:
                # Eye-to-hand swap: target2cam as "gripper2base",
                # gripper2base as "target2cam"
                R_X, t_X = cv2.calibrateHandEye(
                    self._R_target2cam_list,
                    self._t_target2cam_list,
                    self._R_gripper2base_list,
                    self._t_gripper2base_list,
                    method=method,
                )
            except cv2.error as e:
                self.get_logger().warn(f"  {name} failed: {e}")
                continue

            # X = T_gripper2target; compute T_cam2base from it
            result = self._compute_cam2base_from_X(R_X, t_X)
            if result is None:
                continue

            R_c2b, t_c2b, consistency = result
            dist = np.linalg.norm(t_c2b)
            z = t_c2b[2, 0]
            results.append((name, R_c2b, t_c2b, consistency))
            self.get_logger().info(
                f"  {name}: consistency={consistency:.1f}mm, "
                f"dist={dist:.3f}m, z={z:.3f}m")

        # --- calibrateRobotWorldHandEye methods ---
        # Needs R_base2gripper = inv(R_gripper2base)
        R_b2g_list = [R.T for R in self._R_gripper2base_list]
        t_b2g_list = [(-R.T @ t) for R, t in
                      zip(self._R_gripper2base_list,
                          self._t_gripper2base_list)]

        rw_methods = [
            ("RW-Shah", cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH),
            ("RW-Li", cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI),
        ]

        for name, method in rw_methods:
            try:
                R_b2w, t_b2w, R_g2c, t_g2c = \
                    cv2.calibrateRobotWorldHandEye(
                        self._R_target2cam_list,
                        self._t_target2cam_list,
                        R_b2g_list,
                        t_b2g_list,
                        method=method,
                    )
            except cv2.error as e:
                self.get_logger().warn(f"  {name} failed: {e}")
                continue

            # For eye-to-hand: Z = (R_g2c, t_g2c) ~ T_base2cam
            # T_cam2base = inv(Z)
            R_c2b = R_g2c.T
            t_c2b = -R_g2c.T @ t_g2c

            # Consistency: compute per-sample camera position
            # using T_b2c_i = A_i * X * inv(B_i)
            # = T_t2c_i * T_b2w * T_g2b_i
            cam_positions = []
            for i in range(n):
                R_xb, t_xb = _compose_transforms(
                    R_b2w, t_b2w,
                    self._R_gripper2base_list[i],
                    self._t_gripper2base_list[i])
                R_b2c, t_b2c = _compose_transforms(
                    self._R_target2cam_list[i],
                    self._t_target2cam_list[i],
                    R_xb, t_xb)
                R_ci, t_ci = _invert_transform(R_b2c, t_b2c)
                cam_positions.append(t_ci.flatten())

            pos_arr = np.array(cam_positions)
            consistency = np.mean(np.std(pos_arr, axis=0)) * 1000.0

            dist = np.linalg.norm(t_c2b)
            z = t_c2b[2, 0]
            results.append((name, R_c2b, t_c2b, consistency))
            self.get_logger().info(
                f"  {name}: consistency={consistency:.1f}mm, "
                f"dist={dist:.3f}m, z={z:.3f}m")

        # --- Direct nonlinear optimization ---
        # Jointly optimize T_base2cam and T_target2gripper
        # to minimize total pose error across all samples.
        #
        # Kinematic chain (eye-to-hand):
        #   T_target2cam_predicted = inv(T_base2cam) * T_gripper2base * T_target2gripper
        #
        # We parameterize with 12 DOF: rvec_b2c(3) + tvec_b2c(3) + rvec_t2g(3) + tvec_t2g(3)
        # Residual per sample: the difference between predicted and observed T_target2cam
        # expressed as [rvec_diff(3), tvec_diff(3)] = 6 residuals per sample.

        if results:
            # Use best HE/RW result as initial guess
            results.sort(key=lambda x: x[3])
            init_name, init_R_c2b, init_t_c2b, _ = results[0]

            # T_base2cam = inv(T_cam2base)
            init_R_b2c, init_t_b2c = _invert_transform(init_R_c2b, init_t_c2b)

            # Derive initial T_target2gripper from first sample:
            # T_t2c = T_b2c * T_g2b * T_t2g
            # => T_t2g = inv(T_g2b) * inv(T_b2c) * T_t2c
            #          = T_b2g * T_c2b * T_t2c
            R_g2b_0 = self._R_gripper2base_list[0]
            t_g2b_0 = self._t_gripper2base_list[0]
            R_b2g_0, t_b2g_0 = _invert_transform(R_g2b_0, t_g2b_0)

            # T_c2b * T_t2c  (T_c2b = init_R_c2b, init_t_c2b from results)
            R_cb_tc, t_cb_tc = _compose_transforms(
                init_R_c2b, init_t_c2b,
                self._R_target2cam_list[0], self._t_target2cam_list[0])
            # T_t2g = T_b2g * (T_c2b * T_t2c)
            init_R_t2g, init_t_t2g = _compose_transforms(
                R_b2g_0, t_b2g_0, R_cb_tc, t_cb_tc)
        else:
            # No HE/RW result; derive initial guess from first two samples
            self.get_logger().warn(
                "No HE/RW results; using crude initial guess for optimization.")
            init_R_b2c = np.eye(3)
            init_t_b2c = np.array([[0.0], [0.0], [0.5]])
            init_R_t2g = np.eye(3)
            init_t_t2g = np.zeros((3, 1))

        # Pack into parameter vector: [rvec_b2c(3), tvec_b2c(3), rvec_t2g(3), tvec_t2g(3)]
        rvec_b2c_init, _ = cv2.Rodrigues(init_R_b2c)
        rvec_t2g_init, _ = cv2.Rodrigues(init_R_t2g)
        x0 = np.concatenate([
            rvec_b2c_init.flatten(),
            init_t_b2c.flatten(),
            rvec_t2g_init.flatten(),
            init_t_t2g.flatten(),
        ])

        def residuals(x):
            rv_b2c = x[0:3].reshape(3, 1)
            tv_b2c = x[3:6].reshape(3, 1)
            rv_t2g = x[6:9].reshape(3, 1)
            tv_t2g = x[9:12].reshape(3, 1)
            R_b2c, _ = cv2.Rodrigues(rv_b2c)
            R_t2g, _ = cv2.Rodrigues(rv_t2g)

            res = []
            for i in range(n):
                # predicted T_t2c = T_b2c * T_g2b * T_t2g
                R_tmp, t_tmp = _compose_transforms(
                    R_b2c, tv_b2c,
                    self._R_gripper2base_list[i],
                    self._t_gripper2base_list[i])
                R_pred, t_pred = _compose_transforms(
                    R_tmp, t_tmp, R_t2g, tv_t2g)

                # observed
                R_obs = self._R_target2cam_list[i]
                t_obs = self._t_target2cam_list[i]

                # rotation error: Rodrigues(R_pred^T @ R_obs) should be ~0
                R_err = R_pred.T @ R_obs
                rv_err, _ = cv2.Rodrigues(R_err)
                # translation error
                t_err = t_pred - t_obs

                res.extend(rv_err.flatten().tolist())
                res.extend(t_err.flatten().tolist())

            return np.array(res)

        try:
            opt_result = scipy_least_squares(
                residuals, x0, method='lm',
                ftol=1e-12, xtol=1e-12, gtol=1e-12,
                max_nfev=10000)

            xopt = opt_result.x
            rv_b2c = xopt[0:3].reshape(3, 1)
            tv_b2c = xopt[3:6].reshape(3, 1)
            R_b2c_opt, _ = cv2.Rodrigues(rv_b2c)

            # T_cam2base = inv(T_base2cam)
            R_c2b_opt = R_b2c_opt.T
            t_c2b_opt = -R_b2c_opt.T @ tv_b2c

            # Compute consistency (same as HE methods)
            rv_t2g = xopt[6:9].reshape(3, 1)
            tv_t2g = xopt[9:12].reshape(3, 1)
            R_t2g_opt, _ = cv2.Rodrigues(rv_t2g)

            cam_positions = []
            for i in range(n):
                R_b2g_i, t_b2g_i = _invert_transform(
                    self._R_gripper2base_list[i],
                    self._t_gripper2base_list[i])
                # T_base2cam_i = T_t2c_i * inv(T_t2g) * T_b2g_i
                R_g2t, t_g2t = _invert_transform(R_t2g_opt, tv_t2g)
                R_tmp, t_tmp = _compose_transforms(
                    self._R_target2cam_list[i],
                    self._t_target2cam_list[i],
                    R_g2t, t_g2t)
                R_b2c_i, t_b2c_i = _compose_transforms(
                    R_tmp, t_tmp, R_b2g_i, t_b2g_i)
                R_c2b_i, t_c2b_i = _invert_transform(R_b2c_i, t_b2c_i)
                cam_positions.append(t_c2b_i.flatten())

            pos_arr = np.array(cam_positions)
            opt_consistency = np.mean(np.std(pos_arr, axis=0)) * 1000.0

            # Also compute RMS residual in mm
            final_res = residuals(xopt)
            n_samples = n
            # translation residuals are at indices 3,4,5 of each 6-element block
            t_residuals = []
            for i in range(n_samples):
                t_res = final_res[i*6+3 : i*6+6]
                t_residuals.append(np.linalg.norm(t_res))
            rms_t_mm = np.sqrt(np.mean(np.array(t_residuals)**2)) * 1000.0

            dist = np.linalg.norm(t_c2b_opt)
            z = t_c2b_opt[2, 0]
            results.append(("NL-Optim", R_c2b_opt, t_c2b_opt, opt_consistency))
            self.get_logger().info(
                f"  NL-Optim: consistency={opt_consistency:.1f}mm, "
                f"rms_t={rms_t_mm:.1f}mm, "
                f"dist={dist:.3f}m, z={z:.3f}m "
                f"(cost={opt_result.cost:.2e}, nfev={opt_result.nfev})")

        except Exception as e:
            self.get_logger().warn(f"  NL-Optim failed: {e}")

        if not results:
            self.get_logger().error("All calibration methods failed!")
            return

        # Pick the method with lowest consistency error
        results.sort(key=lambda x: x[3])
        best_name, best_R, best_t, best_err = results[0]

        cam_x = best_t[0, 0]
        cam_y = best_t[1, 0]
        cam_z = best_t[2, 0]
        cam_dist = np.linalg.norm(best_t)

        qx, qy, qz, qw = _R_t_to_quaternion(best_R)

        self.get_logger().info("")
        self.get_logger().info("=" * 60)
        self.get_logger().info(
            f"  CALIBRATION RESULT  (method: {best_name})")
        self.get_logger().info("=" * 60)
        self.get_logger().info(
            f"  Consistency: {best_err:.1f}mm")
        self.get_logger().info(
            f"  Camera position in base_link frame:")
        self.get_logger().info(f"    x = {cam_x:.4f} m")
        self.get_logger().info(f"    y = {cam_y:.4f} m")
        self.get_logger().info(f"    z = {cam_z:.4f} m")
        self.get_logger().info(
            f"  Distance from base: {cam_dist:.3f} m")
        self.get_logger().info(
            f"  Quaternion (base_link -> camera_link):")
        self.get_logger().info(
            f"    x={qx:.6f}, y={qy:.6f}, z={qz:.6f}, w={qw:.6f}")
        self.get_logger().info("")
        self.get_logger().info(
            "  Static TF command (paste into launch file):")
        self.get_logger().info(
            f"  ros2 run tf2_ros static_transform_publisher "
            f"--x {cam_x:.4f} --y {cam_y:.4f} --z {cam_z:.4f} "
            f"--qx {qx:.6f} --qy {qy:.6f} --qz {qz:.6f} --qw {qw:.6f} "
            f"--frame-id base_link --child-frame-id camera_link")
        self.get_logger().info("=" * 60)

        # RPY for URDF
        sy = math.sqrt(best_R[0, 0]**2 + best_R[1, 0]**2)
        if sy > 1e-6:
            roll = math.atan2(best_R[2, 1], best_R[2, 2])
            pitch = math.atan2(-best_R[2, 0], sy)
            yaw = math.atan2(best_R[1, 0], best_R[0, 0])
        else:
            roll = math.atan2(-best_R[1, 2], best_R[1, 1])
            pitch = math.atan2(-best_R[2, 0], sy)
            yaw = 0.0

        self.get_logger().info("")
        self.get_logger().info(
            "  For URDF camera_stand joint (base -> camera):")
        self.get_logger().info(
            f"  xyz: {cam_x:.4f} {cam_y:.4f} {cam_z:.4f}")
        self.get_logger().info(
            f"  rpy: {roll:.4f} {pitch:.4f} {yaw:.4f}")
        self.get_logger().info("=" * 60)

        # Sanity checks
        if cam_dist < 0.1 or cam_dist > 3.0:
            self.get_logger().warn(
                f"  Camera distance ({cam_dist:.2f}m) seems unusual. "
                f"Expected 0.3-1.5m for a typical setup.")
        if cam_z < 0:
            self.get_logger().warn(
                f"  Camera Z ({cam_z:.3f}m) is below the robot base. "
                f"Double-check if camera is really below base_link.")
        if best_err > 20.0:
            self.get_logger().warn(
                f"  High consistency error ({best_err:.0f}mm). "
                f"Try more diverse poses with different orientations.")

    def _display_loop(self):
        """Show live camera feed with board detection overlay."""
        if self._rgb is None:
            return

        result, vis = self._detect_board(self._rgb)

        self._last_detection = result
        self._last_frame = (
            self._rgb.copy() if self._rgb is not None else None)

        n = len(self._R_target2cam_list)
        cv2.putText(vis,
                    f"Samples: {n}/12+ | 'c'=capture 's'=solve 'q'=quit",
                    (20, vis.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Hand-Eye Calibration", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            self._capture_sample()
        elif key == ord('s'):
            self._solve_calibration()
        elif key == ord('q'):
            cv2.destroyAllWindows()
            self.get_logger().info("Quitting.")
            raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = HandeyeCalibrationNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
