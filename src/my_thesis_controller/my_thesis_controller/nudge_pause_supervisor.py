#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Bool
from std_srvs.srv import Trigger


class NudgePauseSupervisor(Node):
    def __init__(self):
        super().__init__('nudge_pause_supervisor')

        self.declare_parameter('nudge_topic', '/interaction/is_nudge')
        self.declare_parameter('pause_service', '/servo_node/pause_servo')
        self.declare_parameter('unpause_service', '/servo_node/unpause_servo')
        self.declare_parameter('debounce_s', 0.5)
        

        self.nudge_topic = self.get_parameter('nudge_topic').value
        self.pause_service = self.get_parameter('pause_service').value
        self.unpause_service = self.get_parameter('unpause_service').value
        self.debounce_s = float(self.get_parameter('debounce_s').value)
        
        self.is_paused = False
        self.last_toggle_t = -1e9
        
        self.declare_parameter('enabled_topic', '/hybrid_controller/enabled')
        self.enabled_topic = self.get_parameter('enabled_topic').value
        self.enabled_pub = self.create_publisher(Bool, self.enabled_topic, 10)
        self.enabled_pub.publish(Bool(data=True))  # not paused

        self.cb_group = ReentrantCallbackGroup()

        self.pause_cli = self.create_client(Trigger, self.pause_service, callback_group=self.cb_group)
        self.unpause_cli = self.create_client(Trigger, self.unpause_service, callback_group=self.cb_group)

        while not self.pause_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Waiting for {self.pause_service} ...')

        while not self.unpause_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Waiting for {self.unpause_service} ...')

        self.sub = self.create_subscription(
            Bool,
            self.nudge_topic,
            self.on_nudge,
            10,
            callback_group=self.cb_group
        )

        self.get_logger().info(
            f"Ready. NUDGE={self.nudge_topic}, pause={self.pause_service}, unpause={self.unpause_service}"
        )

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_nudge(self, msg: Bool):
        if not msg.data:
            return

        t = self.now_s()
        if (t - self.last_toggle_t) < self.debounce_s:
            return
        self.last_toggle_t = t

        target_pause = not self.is_paused
        req = Trigger.Request()

        if target_pause:
            fut = self.pause_cli.call_async(req)
            fut.add_done_callback(lambda f: self.on_response(f, paused=True))
        else:
            fut = self.unpause_cli.call_async(req)
            fut.add_done_callback(lambda f: self.on_response(f, paused=False))

        self.get_logger().info(f"NUDGE -> requesting {'PAUSE' if target_pause else 'UNPAUSE'}")

    def on_response(self, fut, paused: bool):
        try:
            resp = fut.result()
            if resp.success:
                self.is_paused = paused
                self.enabled_pub.publish(Bool(data=(not self.is_paused)))
                self.get_logger().info(f"{'Paused' if paused else 'Unpaused'} ok: {resp.message}")
            else:
                self.get_logger().error(f"Servo service failed: {resp.message}")
        except Exception as e:
            self.get_logger().error(f"Servo service call error: {e}")


def main():
    rclpy.init()
    node = NudgePauseSupervisor()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
