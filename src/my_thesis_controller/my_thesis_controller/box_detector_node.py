#!/usr/bin/env python3
"""
box_detector_node.py

Detects a coloured box (red, blue, or green) using the RealSense RGBD
camera.  Uses HSV colour segmentation on the RGB image, back-projects
matching pixels with valid depth to 3-D, computes the mean surface
centroid, then corrects for the box half-height to estimate the true
geometric centre.  Publishes the result in base_link frame.

Publishes:
    /detected_box_pose  (geometry_msgs/PoseStamped)  in base_link frame

Parameter:
    color (string): "red" (default), "blue", or "green"
"""

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

import tf2_ros

try:
    from cv_bridge import CvBridge
    import cv2
    _HAS_CV = True
except ImportError:
    _HAS_CV = False

# Known box height — used to correct from top-face centroid to geometric centre
BOX_HEIGHT = 0.06  # 6 cm

# HSV colour ranges for supported box colours
COLOR_RANGES = {
    "red": [
        (np.array([0, 80, 80]), np.array([10, 255, 255])),
        (np.array([170, 80, 80]), np.array([180, 255, 255])),
    ],
    "blue": [
        (np.array([100, 80, 80]), np.array([130, 255, 255])),
    ],
    "green": [
        (np.array([40, 80, 80]), np.array([80, 255, 255])),
    ],
}


class BoxDetectorNode(Node):
    def __init__(self):
        super().__init__("box_detector")

        if not _HAS_CV:
            self.get_logger().error(
                "cv2 / cv_bridge not available — box detection disabled."
            )
            return

        self.declare_parameter('color', 'red')
        self._color = self.get_parameter('color').value.lower()
        if self._color not in COLOR_RANGES:
            self.get_logger().error(
                f"Unknown color '{self._color}'. "
                f"Supported: {list(COLOR_RANGES.keys())}")
            return

        self._bridge = CvBridge()
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Camera intrinsics (populated by camera_info callback)
        self._K = None
        self._camera_frame = None

        # Latest frames
        self._rgb = None
        self._depth = None

        # Publisher
        self._pub = self.create_publisher(
            PoseStamped, "/detected_box_pose", 10
        )

        # Subscribers — topic names match the Ignition→ROS bridge output
        self.create_subscription(
            CameraInfo, "/realsense/camera_info", self._info_cb, 10
        )
        self.create_subscription(
            Image, "/realsense/image", self._rgb_cb, 10
        )
        self.create_subscription(
            Image, "/realsense/depth_image", self._depth_cb, 10
        )

        # Run detection at 5 Hz
        self.create_timer(0.2, self._tick)

        self.get_logger().info(
            f"BoxDetectorNode ready — detecting '{self._color}' boxes.")

    # ── callbacks ──────────────────────────────────────────────────────
    def _info_cb(self, msg: CameraInfo):
        self._K = np.array(msg.k).reshape(3, 3)
        self._camera_frame = msg.header.frame_id or "camera_depth_frame"

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

    # ── detection tick ─────────────────────────────────────────────────
    def _tick(self):
        if self._rgb is None or self._depth is None or self._K is None:
            return

        # ---- 1. HSV colour segmentation ----
        hsv = cv2.cvtColor(self._rgb, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in COLOR_RANGES[self._color]:
            mask |= cv2.inRange(hsv, lo, hi)

        # morphological cleanup
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)

        # ---- 2. Largest contour ----
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 50:
            return

        # Fill contour into a clean mask for pixel sampling
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [largest], -1, 255, cv2.FILLED)

        # ---- 3. Back-project all colour pixels with valid depth ----
        fx, fy = self._K[0, 0], self._K[1, 1]
        cx, cy = self._K[0, 2], self._K[1, 2]

        # Get pixel coordinates of all colour pixels
        vs, us = np.where(contour_mask > 0)
        if len(vs) == 0:
            return

        # Sub-sample for efficiency (max ~200 points)
        step = max(1, len(vs) // 200)
        vs = vs[::step]
        us = us[::step]

        # Read depth at those pixels
        depths_raw = self._depth[vs, us].astype(np.float64)
        if self._depth.dtype == np.uint16:
            depths_m = depths_raw / 1000.0
        else:
            depths_m = depths_raw

        # Keep only valid depth
        valid = (depths_m > 0.1) & (depths_m < 3.0) & np.isfinite(depths_m)
        if np.sum(valid) < 5:
            return

        us_v = us[valid].astype(np.float64)
        vs_v = vs[valid].astype(np.float64)
        ds_v = depths_m[valid]

        # Back-project to 3D in camera optical frame
        x_opt = (us_v - cx) * ds_v / fx
        y_opt = (vs_v - cy) * ds_v / fy
        z_opt = ds_v

        # Mean 3D centroid of the visible surface
        x_c = np.mean(x_opt)
        y_c = np.mean(y_opt)
        z_c = np.mean(z_opt)

        # ---- 4. Transform optical frame → base_link ----
        optical_frame = "camera_depth_optical_frame"
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                "base_link", optical_frame, rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(
                f"TF lookup failed: {e}", throttle_duration_sec=5.0
            )
            return

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        R = _quat_to_rot(q.x, q.y, q.z, q.w)
        p_surface = R @ np.array([x_c, y_c, z_c]) + np.array([t.x, t.y, t.z])

        # ---- 5. Correct for box height ----
        # Camera sees the top face; true box centre is half-height below
        # in the world Z direction (which is base_link Z direction).
        p_centre = p_surface.copy()
        p_centre[2] -= BOX_HEIGHT / 2.0

        # ---- 6. Publish ----
        ps = PoseStamped()
        ps.header.frame_id = "base_link"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(p_centre[0])
        ps.pose.position.y = float(p_centre[1])
        ps.pose.position.z = float(p_centre[2])
        ps.pose.orientation.w = 1.0
        self._pub.publish(ps)

        self.get_logger().info(
            f"{self._color} box @ base_link ({p_centre[0]:.3f}, {p_centre[1]:.3f}, {p_centre[2]:.3f})  "
            f"[{np.sum(valid)} pts, surface_z={p_surface[2]:.3f}]",
            throttle_duration_sec=2.0,
        )


# ── helper ────────────────────────────────────────────────────────────
def _quat_to_rot(x, y, z, w):
    """Quaternion (x,y,z,w) → 3×3 rotation matrix."""
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)    ],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)    ],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def main(args=None):
    rclpy.init(args=args)
    node = BoxDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
