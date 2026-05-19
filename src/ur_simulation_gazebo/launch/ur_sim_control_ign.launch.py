from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    world = PathJoinSubstitution([
        FindPackageShare("ur_simulation_gazebo"),
        "worlds",
        "ft_world.sdf"
    ])

    # Start Ignition Gazebo
    ign = ExecuteProcess(
        cmd=["ign", "gazebo", "-r", world],
        output="screen"
    )

    # Bridge Force-Torque sensor
    ft_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/force_torque@geometry_msgs/msg/WrenchStamped[ignition.msgs.Wrench"
        ],
        output="screen",
    )

    return LaunchDescription([
        ign,
        ft_bridge
    ])

