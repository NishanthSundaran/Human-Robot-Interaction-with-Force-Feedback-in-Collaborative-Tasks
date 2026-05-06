"""
Robotiq 2F-140 gripper control via URCap TCP socket (port 63352).

Communicates with the Robotiq URCap's built-in socket server on the UR
controller.  Requires teach pendant Tool I/O set to
"Controlled by: Robotiq_Grippers".

No /tmp/ttyUR, socat, or raw Modbus needed.
"""

import socket
import re
import time
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

GRIPPER_MAX_JOINT_VAL = 0.695
ROBOTIQ_MAX_POS = 255
SOCKET_PORT = 63352


def joint_to_robotiq_pos(fj: float) -> int:
    ratio = max(0.0, min(1.0, fj / GRIPPER_MAX_JOINT_VAL))
    return int(ratio * ROBOTIQ_MAX_POS)


class RobotiqURCapBridge(Node):
    def __init__(self):
        super().__init__("robotiq_urscript_bridge")

        self.declare_parameter("gripper_topic", "/robotiq_gripper_command")
        self.declare_parameter("robot_ip", "192.168.123.3")
        self.declare_parameter("speed", 255)
        self.declare_parameter("force", 128)

        self._topic = self.get_parameter("gripper_topic").value
        self._robot_ip = self.get_parameter("robot_ip").value
        self._speed = self.get_parameter("speed").value
        self._force = self.get_parameter("force").value

        self._sock = None
        self._activated = False
        self._last_pos = -1
        self._lock = threading.Lock()

        self.create_subscription(
            Float64MultiArray, self._topic, self._cmd_cb, 10)

        self._connect_timer = self.create_timer(3.0, self._try_connect)
        self.get_logger().info(
            f"RobotiqURCapBridge: connecting to {self._robot_ip}:{SOCKET_PORT}, "
            f"listening on {self._topic}")

    # ── TCP connection ────────────────────────────────────────────────
    def _try_connect(self):
        if self._sock is not None:
            return

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3.0)
            s.connect((self._robot_ip, SOCKET_PORT))
            self._sock = s
            self._connect_timer.cancel()
            self.get_logger().info(
                f"Connected to URCap socket {self._robot_ip}:{SOCKET_PORT}")
            # Activate in background thread
            self._activate_thread = threading.Thread(
                target=self._activate, daemon=True)
            self._activate_thread.start()
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            self.get_logger().warn(
                f"Cannot connect to {self._robot_ip}:{SOCKET_PORT}: {e}")
            self._sock = None

    def _send_cmd(self, cmd: str) -> str:
        """Send a text command and return the response."""
        with self._lock:
            if self._sock is None:
                return ""
            try:
                self._sock.sendall((cmd + "\n").encode("utf-8"))
                time.sleep(0.1)
                self._sock.settimeout(2.0)
                resp = self._sock.recv(1024).decode("utf-8").strip()
                self.get_logger().debug(f"TX: {cmd!r}  RX: {resp!r}")
                return resp
            except (socket.timeout, OSError) as e:
                self.get_logger().warn(f"Socket error: {e}")
                return ""

    # ── Activation ───────────────────────────────────────────────────
    def _activate(self):
        if self._activated:
            return

        self.get_logger().info("Gripper: activating via URCap socket...")

        # Reset first
        self._send_cmd("SET ACT 0")
        time.sleep(0.5)

        # Activate
        self._send_cmd("SET ACT 1")
        time.sleep(0.5)
        self._send_cmd("SET GTO 1")

        # Set speed and force
        self._send_cmd(f"SET SPE {self._speed}")
        self._send_cmd(f"SET FOR {self._force}")

        # Poll for activation complete
        for i in range(40):
            time.sleep(0.5)
            sta_resp = self._send_cmd("GET STA")
            act_resp = self._send_cmd("GET ACT")
            self.get_logger().info(
                f"Gripper poll [{i}]: ACT={act_resp}, STA={sta_resp}")

            # STA=3 means activation complete
            sta_match = re.search(r'\d+', sta_resp)
            if sta_match and int(sta_match.group()) == 3:
                self._activated = True
                self.get_logger().info("Gripper activated successfully!")
                return

        self.get_logger().warn(
            "Gripper activation timed out. Setting activated=True anyway.")
        self._activated = True

    # ── Command callback ─────────────────────────────────────────────
    def _cmd_cb(self, msg: Float64MultiArray):
        if not self._activated or self._sock is None:
            return
        if not msg.data:
            return

        fj = msg.data[0]
        pos = joint_to_robotiq_pos(fj)

        if pos == self._last_pos:
            return
        self._last_pos = pos

        resp = self._send_cmd(f"SET POS {pos}")
        self.get_logger().info(
            f"Gripper move: fj={fj:.3f} -> pos={pos}, resp={resp}")

    # ── Cleanup ──────────────────────────────────────────────────────
    def destroy_node(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RobotiqURCapBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
