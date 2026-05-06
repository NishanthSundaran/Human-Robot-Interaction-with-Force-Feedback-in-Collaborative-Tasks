"""
Tkinter GUI with force sliders for simulating wrench input.

Publishes WrenchStamped to /wrench_external at ~100 Hz in base_link frame.
Sliders correspond to intuitive world directions:
  Fx = forward/backward, Fy = left/right, Fz = up/down

The hybrid_control_node subscribes to /wrench_external and adds it on top
of the real FT sensor wrench — no topic collision.

Also publishes a force arrow Marker for RViz visualization (/force_marker).

Usage: ros2 run my_thesis_controller force_gui
"""
import threading
import tkinter as tk

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped, Point
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker
import tf2_ros


class ForceGuiNode(Node):
    def __init__(self):
        super().__init__("force_gui")
        self.pub = self.create_publisher(WrenchStamped, "/wrench_external", 10)
        self.state_pub = self.create_publisher(String, "/interaction_state", 10)
        self.marker_pub = self.create_publisher(Marker, "/force_marker", 10)
        self.nudge_pub = self.create_publisher(Bool, "/interaction/is_nudge", 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(0.01, self.publish_wrench)  # 100 Hz
        self.fx = 0.0
        self.fy = 0.0
        self.fz = 0.0
        self._nudge_pending = False
        self._guiding_override = True

    def publish_wrench(self):
        now = self.get_clock().now()

        # Publish force in base_link frame (no transform needed)
        msg = WrenchStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "base_link"
        msg.wrench.force.x = self.fx
        msg.wrench.force.y = self.fy
        msg.wrench.force.z = self.fz
        self.pub.publish(msg)

        # Publish interaction state
        f_mag = np.linalg.norm([self.fx, self.fy, self.fz])
        if self._nudge_pending:
            self._nudge_pending = False
            self.state_pub.publish(String(data="NUDGE"))
            self.nudge_pub.publish(Bool(data=True))
            self.get_logger().info("NUDGE published via GUI button")
        elif self._guiding_override and f_mag > 0.5:
            self.state_pub.publish(String(data="GUIDING"))

        # Publish force arrow marker for RViz
        self._publish_marker(f_mag, now)

    def _publish_marker(self, f_mag, now):
        if f_mag < 0.1:
            m = Marker()
            m.header.stamp = now.to_msg()
            m.header.frame_id = "base_link"
            m.ns = "force_gui"
            m.id = 0
            m.action = Marker.DELETE
            self.marker_pub.publish(m)
            return

        # Get tool0 position for arrow start
        try:
            tf = self.tf_buffer.lookup_transform("base_link", "tool0", rclpy.time.Time())
            sx = tf.transform.translation.x
            sy = tf.transform.translation.y
            sz = tf.transform.translation.z
        except tf2_ros.TransformException:
            return

        f_vec = np.array([self.fx, self.fy, self.fz])
        f_dir = f_vec / f_mag
        arrow_len = min(f_mag * 0.05, 0.5)  # 5cm per Newton, max 50cm

        m = Marker()
        m.header.stamp = now.to_msg()
        m.header.frame_id = "base_link"
        m.ns = "force_gui"
        m.id = 0
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.points = [
            Point(x=sx, y=sy, z=sz),
            Point(x=sx + f_dir[0] * arrow_len,
                  y=sy + f_dir[1] * arrow_len,
                  z=sz + f_dir[2] * arrow_len),
        ]
        m.scale.x = 0.015   # shaft diameter
        m.scale.y = 0.030   # head diameter
        m.scale.z = 0.030   # head length
        m.color.r = 1.0
        m.color.g = 0.2
        m.color.b = 0.2
        m.color.a = 0.9
        m.lifetime.nanosec = 200_000_000
        self.marker_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = ForceGuiNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    root = tk.Tk()
    root.title("Force GUI — /wrench_external (base_link)")
    root.resizable(False, False)

    sliders = {}
    labels = [
        ("Fx", "#d44", "Forward (+) / Backward (-)"),
        ("Fy", "#4a4", "Left (+) / Right (-)"),
        ("Fz", "#44d", "Up (+) / Down (-)"),
    ]
    for axis, color, desc in labels:
        frame = tk.Frame(root)
        frame.pack(padx=10, pady=5, fill="x")
        tk.Label(frame, text=f"{axis} (N):", width=8, anchor="w",
                 font=("monospace", 12)).pack(side="left")
        s = tk.Scale(frame, from_=-30.0, to=30.0, resolution=0.5,
                     orient="horizontal", length=350,
                     troughcolor=color, font=("monospace", 10))
        s.set(0.0)
        s.pack(side="left", fill="x", expand=True)
        tk.Label(frame, text=desc, font=("monospace", 9), fg="gray").pack(side="left", padx=5)
        sliders[axis] = s

    guiding_var = tk.BooleanVar(value=True)

    def toggle_guiding():
        node._guiding_override = guiding_var.get()

    check_frame = tk.Frame(root)
    check_frame.pack(pady=3)
    tk.Checkbutton(check_frame, text="Force GUIDING state while sliding",
                   variable=guiding_var, command=toggle_guiding,
                   font=("monospace", 10)).pack()

    def zero_all():
        for s in sliders.values():
            s.set(0.0)

    def send_nudge():
        node._nudge_pending = True

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=8)
    tk.Button(btn_frame, text="Zero All", command=zero_all,
              font=("monospace", 11), width=12).pack(side="left", padx=5)
    tk.Button(btn_frame, text="NUDGE", command=send_nudge,
              font=("monospace", 11, "bold"), width=12,
              bg="#f90", fg="white", activebackground="#c70").pack(side="left", padx=5)

    def update_node():
        node.fx = sliders["Fx"].get()
        node.fy = sliders["Fy"].get()
        node.fz = sliders["Fz"].get()
        root.after(50, update_node)

    update_node()

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
