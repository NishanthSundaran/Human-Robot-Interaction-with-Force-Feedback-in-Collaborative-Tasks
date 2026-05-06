# setup_planning_scene.py
import rclpy
from rclpy.node import Node
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose


# Static obstacles for the planning scene.
# Each dict defines one CollisionObject.
OBSTACLES = [
    {
        # Extended robot table (0.9 x 0.7 x 0.66m).
        # Pieces and assembly target sit on this single table.
        # World center: (-0.15, 0, 0.33). In base_link: (0.0, 0.15, -0.33).
        # Top: z = -0.33 + 0.33 = 0.0m (table surface = base_link z=0).
        "id": "robot_table",
        "frame_id": "world",
        "type": SolidPrimitive.BOX,
        "dimensions": [0.9, 0.7, 0.66],
        "position": (-0.15, 0.0, -0.33),
    },
]


class SceneSetup(Node):
    def __init__(self):
        super().__init__('setup_planning_scene')
        self._pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        self._attempts = 0
        self._max_attempts = 3
        self.create_timer(2.0, self._publish_scene)

    def _publish_scene(self):
        if self._attempts >= self._max_attempts:
            return

        collision_objects = []
        for obs in OBSTACLES:
            co = CollisionObject()
            co.header.frame_id = obs["frame_id"]
            co.id = obs["id"]
            co.operation = CollisionObject.ADD

            primitive = SolidPrimitive()
            primitive.type = obs["type"]
            primitive.dimensions = list(obs["dimensions"])

            pose = Pose()
            pose.position.x = obs["position"][0]
            pose.position.y = obs["position"][1]
            pose.position.z = obs["position"][2]
            pose.orientation.w = 1.0

            co.primitives = [primitive]
            co.primitive_poses = [pose]
            collision_objects.append(co)

        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects = collision_objects
        self._pub.publish(scene)

        self._attempts += 1
        self.get_logger().info(
            f'Planning scene published (attempt {self._attempts}/{self._max_attempts}): '
            f'{[o["id"] for o in OBSTACLES]}'
        )


def main():
    rclpy.init()
    rclpy.spin(SceneSetup())
