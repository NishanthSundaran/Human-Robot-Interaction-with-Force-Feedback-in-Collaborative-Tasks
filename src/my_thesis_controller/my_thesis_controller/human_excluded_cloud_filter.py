#!/usr/bin/env python3
"""
human_excluded_cloud_filter.py

Removes human pixels from the depth-derived point cloud using YOLOv8 instance
segmentation. Output cloud feeds the OctoMap so MoveIt sees the static scene
(table, fixtures, objects) but NOT the human operator.

Pipeline:
  /realsense/image                 ──► YOLOv8-seg (class=person, masks)
  /realsense/camera_info           ──► intrinsics
  /cloud_no_robot (sensor_msgs/PC2)──► reproject pixels NOT in human mask
                                       ──► /cloud_no_human

Fail-open: if model not loaded yet OR no detections, the input cloud is
forwarded unchanged so OctoMap doesn't go stale.

Notes:
  - Uses BEST_EFFORT QoS to match RealSense / Gazebo bridges.
  - Eager YOLO load + warmup so the first inference doesn't block the executor.
  - Mask is dilated by ~10 px to absorb edge bleed and animation jitter.
"""

import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                       QoSHistoryPolicy, QoSDurabilityPolicy)

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge

try:
    import cv2
except ImportError:
    cv2 = None


class HumanExcludedCloudFilter(Node):
    def __init__(self):
        super().__init__('human_excluded_cloud_filter')

        # Parameters
        self.declare_parameter('model', 'yolov8n-seg.pt')
        self.declare_parameter('confidence', 0.25)
        self.declare_parameter('rate_limit', 10.0)        # max detection Hz
        self.declare_parameter('mask_dilate_px', 10)
        self.declare_parameter('max_depth', 3.0)
        self.declare_parameter('min_depth', 0.2)
        self.declare_parameter('color_topic', '/realsense/image')
        self.declare_parameter('camera_info_topic', '/realsense/camera_info')
        self.declare_parameter('depth_topic', '/realsense/depth_image')
        self.declare_parameter('cloud_in_topic', '/cloud_no_robot')
        self.declare_parameter('cloud_out_topic', '/cloud_no_human')

        self.model_name = self.get_parameter('model').value
        self.confidence = self.get_parameter('confidence').value
        self.rate_limit = self.get_parameter('rate_limit').value
        self.dilate_px = int(self.get_parameter('mask_dilate_px').value)
        self.max_depth = self.get_parameter('max_depth').value
        self.min_depth = self.get_parameter('min_depth').value

        color_topic = self.get_parameter('color_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        cloud_in_topic = self.get_parameter('cloud_in_topic').value
        cloud_out_topic = self.get_parameter('cloud_out_topic').value

        self.bridge = CvBridge()
        self.camera_info = None
        self.last_color = None        # (timestamp, bgr image)
        self.last_human_mask = None   # bool array (H, W) — None if no recent detection
        self.last_mask_time = 0.0
        self.last_detect_time = 0.0
        self.mask_ttl = 0.5           # seconds — how long to trust the latest mask

        # Eager YOLO load + warmup
        self.model = None
        try:
            from ultralytics import YOLO
            self.get_logger().info(f'Loading YOLO seg model: {self.model_name}')
            self.model = YOLO(self.model_name)
            _dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self.model(_dummy, conf=0.5, classes=[0], verbose=False)
            self.get_logger().info('YOLO seg model loaded and warmed up')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO seg at startup: {e}')

        # Input QoS: BEST_EFFORT to match Gazebo bridges + robot_self_filter.
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Output QoS: RELIABLE so both BEST_EFFORT consumers (octomap_gate) AND
        # RELIABLE consumers (RViz default, tf tools) are compatible.
        # RELIABLE→BEST_EFFORT is compatible; BEST_EFFORT→RELIABLE is not.
        output_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.create_subscription(CameraInfo, info_topic, self._info_cb, sensor_qos)
        self.create_subscription(Image, color_topic, self._color_cb, sensor_qos)
        self.create_subscription(PointCloud2, cloud_in_topic, self._cloud_cb, sensor_qos)

        self.pub_cloud = self.create_publisher(PointCloud2, cloud_out_topic, output_qos)
        self.pub_debug_img = self.create_publisher(
            Image, '/human_excluded_cloud_filter/debug_image', output_qos)

        # Diagnostics
        self._color_n = 0
        self._cloud_n = 0
        self._mask_n = 0
        self._dropped_total = 0
        self._filtered_frames = 0
        self._last_cloud_frame = ""
        self._last_valid = 0
        self._last_in_image = 0
        self._last_mask_px = 0
        self._last_total_pts = 0
        self.create_timer(5.0, self._diag_cb)

        self.get_logger().info(
            f'human_excluded_cloud_filter ready '
            f'(model={self.model_name}, conf={self.confidence}, '
            f'rate={self.rate_limit}Hz, dilate={self.dilate_px}px)')
        self.get_logger().info(
            f'Subscribing: color={color_topic} info={info_topic} '
            f'cloud_in={cloud_in_topic} -> cloud_out={cloud_out_topic}')

    # ── Callbacks ─────────────────────────────────────────────────────

    def _info_cb(self, msg: CameraInfo):
        if self.camera_info is None:
            self.get_logger().info(
                f'Got camera_info: {msg.width}x{msg.height} '
                f'frame={msg.header.frame_id}')
        self.camera_info = msg

    def _color_cb(self, msg: Image):
        """Run YOLOv8-seg, cache the latest human mask."""
        self._color_n += 1

        # Rate limit detection
        now = time.monotonic()
        if (now - self.last_detect_time) < (1.0 / self.rate_limit):
            return
        self.last_detect_time = now

        if self.model is None:
            return

        try:
            color_bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f'Color conversion failed: {e}',
                                   throttle_duration_sec=5.0)
            return

        h, w = color_bgr.shape[:2]

        # Run segmentation (class 0 = person)
        results = self.model(
            color_bgr,
            conf=self.confidence,
            classes=[0],
            verbose=False,
        )

        if (len(results) == 0 or
                results[0].masks is None or
                len(results[0].masks) == 0):
            # No human detected → drop cached mask after TTL
            if (now - self.last_mask_time) > self.mask_ttl:
                self.last_human_mask = None
            self._publish_debug(color_bgr, None, msg.header, n_persons=0)
            return

        # Combine all person masks into one boolean mask
        # results[0].masks.data is a (N, H_m, W_m) tensor at YOLO's internal
        # resolution; need to resize to color image size.
        masks_tensor = results[0].masks.data.cpu().numpy()  # (N, H_m, W_m)
        n_persons = int(masks_tensor.shape[0])
        combined = np.any(masks_tensor > 0.5, axis=0).astype(np.uint8)

        if combined.shape != (h, w) and cv2 is not None:
            combined = cv2.resize(combined, (w, h),
                                  interpolation=cv2.INTER_NEAREST)

        # Dilate to handle edge bleed
        if self.dilate_px > 0 and cv2 is not None:
            k = self.dilate_px
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            combined = cv2.dilate(combined, kernel, iterations=1)

        self.last_human_mask = combined.astype(bool)
        self.last_mask_time = now
        self._mask_n += 1

        self._publish_debug(color_bgr, combined, msg.header, n_persons=n_persons)

    def _publish_debug(self, color_bgr, mask, header, n_persons=0):
        """Publish annotated image: red overlay on detected humans + label."""
        if cv2 is None:
            return
        try:
            out = color_bgr.copy()
            if mask is not None:
                overlay = out.copy()
                overlay[mask > 0] = (0, 0, 255)
                out = cv2.addWeighted(overlay, 0.5, out, 0.5, 0)
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
            label = (f"HUMAN x{n_persons}" if n_persons > 0 else "no human")
            color = (0, 255, 0) if n_persons > 0 else (200, 200, 200)
            cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, color, 2, cv2.LINE_AA)
            img_msg = self.bridge.cv2_to_imgmsg(out, encoding='bgr8')
            img_msg.header = header
            self.pub_debug_img.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f'Debug image publish failed: {e}',
                                   throttle_duration_sec=5.0)

    def _cloud_cb(self, msg: PointCloud2):
        """Filter incoming cloud against the latest human mask, republish."""
        self._cloud_n += 1

        # Fail-open: if no intrinsics, no model, or no fresh mask → forward as-is
        now = time.monotonic()
        mask_fresh = (self.last_human_mask is not None and
                      (now - self.last_mask_time) < self.mask_ttl)

        if (self.camera_info is None or not mask_fresh):
            self.pub_cloud.publish(msg)
            return

        # Parse the cloud. We assume the standard XYZ layout (point_step=12 or 16).
        try:
            pts = self._read_xyz(msg)
        except Exception as e:
            self.get_logger().warn(f'Cloud parse failed: {e}',
                                   throttle_duration_sec=5.0)
            self.pub_cloud.publish(msg)
            return

        if pts.shape[0] == 0:
            self.pub_cloud.publish(msg)
            return

        # Project each 3D point back to image plane to test against mask.
        # This works regardless of whether the cloud was self-filtered or not.
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]
        W = self.camera_info.width
        H = self.camera_info.height

        # Axis convention depends on source:
        #   - ROS optical frame (real RealSense): +X right, +Y down, +Z forward.
        #     frame_id ends with "_optical_frame".
        #   - Gazebo Ignition depth_camera: body frame (+X forward, +Y left, +Z up).
        frame_id = msg.header.frame_id or ""
        if frame_id.endswith("_optical_frame"):
            opt_x = pts[:, 0]
            opt_y = pts[:, 1]
            depth = pts[:, 2]
        else:
            bx = pts[:, 0]
            by = pts[:, 1]
            bz = pts[:, 2]
            depth = bx
            opt_x = -by
            opt_y = -bz

        # Only points in front of camera with valid depth
        valid = (depth > self.min_depth) & (depth < self.max_depth) & np.isfinite(depth)

        u = np.full_like(depth, -1.0)
        v = np.full_like(depth, -1.0)
        u[valid] = (opt_x[valid] * fx / depth[valid]) + cx
        v[valid] = (opt_y[valid] * fy / depth[valid]) + cy

        ui = u.astype(np.int32)
        vi = v.astype(np.int32)
        in_image = valid & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

        # Default: keep all points (fail-open). Only drop those mapped onto
        # human mask pixels.
        keep = np.ones(pts.shape[0], dtype=bool)
        dropped = 0
        if np.any(in_image):
            mask = self.last_human_mask
            human_hit = mask[vi[in_image], ui[in_image]]
            # Drop points that hit a human pixel
            in_image_idx = np.where(in_image)[0]
            keep[in_image_idx[human_hit]] = False
            dropped = int(human_hit.sum())

        self._dropped_total += dropped
        if dropped > 0:
            self._filtered_frames += 1

        # Remember last-cloud geometry for diag
        self._last_cloud_frame = msg.header.frame_id
        self._last_total_pts = int(pts.shape[0])
        self._last_valid = int(valid.sum())
        self._last_in_image = int(in_image.sum())
        self._last_mask_px = int(self.last_human_mask.sum()) if self.last_human_mask is not None else 0

        kept_pts = pts[keep]
        out_msg = self._make_cloud(kept_pts, msg.header)
        self.pub_cloud.publish(out_msg)

    def _diag_cb(self):
        info_frame = self.camera_info.header.frame_id if self.camera_info else "NONE"
        self.get_logger().info(
            f'Diag: color={self._color_n} cloud_in={self._cloud_n} '
            f'masks={self._mask_n} dropped_pts={self._dropped_total} '
            f'filtered_frames={self._filtered_frames} '
            f'last_cloud: frame={self._last_cloud_frame} '
            f'total={self._last_total_pts} valid={self._last_valid} '
            f'in_image={self._last_in_image} mask_px={self._last_mask_px} '
            f'info_frame={info_frame}')

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _read_xyz(msg: PointCloud2) -> np.ndarray:
        """Extract Nx3 float32 XYZ array from a PointCloud2 with x,y,z fields."""
        # Find x,y,z field offsets
        offsets = {}
        for f in msg.fields:
            if f.name in ('x', 'y', 'z'):
                offsets[f.name] = f.offset
        if not all(k in offsets for k in ('x', 'y', 'z')):
            raise ValueError('Cloud missing x/y/z fields')

        n = msg.width * msg.height
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32)

        raw = np.frombuffer(msg.data, dtype=np.uint8)
        # Reshape into rows of point_step bytes
        if raw.size < n * msg.point_step:
            raise ValueError('Cloud data shorter than expected')
        rows = raw[:n * msg.point_step].reshape(n, msg.point_step)

        ox, oy, oz = offsets['x'], offsets['y'], offsets['z']
        x = rows[:, ox:ox + 4].copy().view(np.float32).reshape(-1)
        y = rows[:, oy:oy + 4].copy().view(np.float32).reshape(-1)
        z = rows[:, oz:oz + 4].copy().view(np.float32).reshape(-1)
        return np.stack([x, y, z], axis=-1)

    def _make_cloud(self, points: np.ndarray, header) -> PointCloud2:
        msg = PointCloud2()
        msg.header.stamp = header.stamp
        msg.header.frame_id = header.frame_id
        msg.height = 1
        msg.width = int(points.shape[0])
        msg.fields = [
            PointField(name='x', offset=0,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,
                       datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * msg.width
        if msg.width > 0:
            msg.data = points.astype(np.float32).tobytes()
        else:
            msg.data = bytes()
        msg.is_dense = True
        return msg


def main():
    rclpy.init()
    node = HumanExcludedCloudFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass


if __name__ == '__main__':
    main()
