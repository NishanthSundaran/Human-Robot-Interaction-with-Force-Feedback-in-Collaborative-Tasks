#!/usr/bin/env python3
import os
import shutil
import subprocess
import yaml

from ament_index_python.packages import get_package_share_directory, get_package_prefix

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _run_xacro(xacro_path: str, mappings: dict) -> str:
    """
    Run xacro -> URDF string (robustly).
    """
    xacro_exe = shutil.which("xacro")
    if not xacro_exe:
        raise RuntimeError("xacro executable not found in PATH. Install ros-humble-xacro.")

    cmd = [xacro_exe, xacro_path]
    for k, v in mappings.items():
        cmd.append(f"{k}:={v}")

    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return out.decode("utf-8")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "xacro failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Output:\n{e.output.decode('utf-8', errors='replace')}"
        )


def _load_yaml_file(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _launch_setup(context, *args, **kwargs):
    # Launch args
    ur_type = LaunchConfiguration("ur_type").perform(context)
    robot_name = LaunchConfiguration("robot_name").perform(context)
    prefix = LaunchConfiguration("prefix").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() in ("true", "1", "yes")
    world_path = LaunchConfiguration("world").perform(context)
    spawn_z = float(LaunchConfiguration("spawn_z").perform(context))

    launch_servo = LaunchConfiguration("launch_servo").perform(context)
    launch_admittance = LaunchConfiguration("launch_admittance").perform(context)

    # Packages
    ur_desc_share = get_package_share_directory("ur_description")
    ur_sim_share = get_package_share_directory("ur_simulation_gazebo")
    try:
        ur_moveit_share = get_package_share_directory("ur3e_moveit_config")
    except Exception:
        ur_moveit_share = ""  # Servo will still start if you disable it, but user said not optional.

    # Paths
    urdf_xacro = os.path.join(ur_desc_share, "urdf", "ur.urdf.xacro")
    srdf_xacro = os.path.join(ur_moveit_share, "srdf", "ur.srdf.xacro") if ur_moveit_share else ""

    servo_yaml_path = os.path.join(ur_moveit_share, "config", "ur_servo.yaml") if ur_moveit_share else ""
    # Some installs might use a different name; keep a fallback:
    if ur_moveit_share and not os.path.exists(servo_yaml_path):
        alt = os.path.join(ur_moveit_share, "config", "servo.yaml")
        if os.path.exists(alt):
            servo_yaml_path = alt

    # Build robot_description (URDF)
    #
    # IMPORTANT:
    # Your current ur.urdf.xacro (the one you pasted) accepts: name, ur_type, prefix, use_sim_time, force_abs_paths, etc.
    # It does NOT accept robot_ip / sim_ignition in your modified version.
    # So we only pass args that exist in your file to avoid "Invalid parameter ..." crashes.
    robot_description_xml = _run_xacro(
        urdf_xacro,
        {
            "ur_type": ur_type,
            "name": robot_name,
            "prefix": prefix,
            "use_sim_time": "true" if use_sim_time else "false",
            # if your xacro supports force_abs_paths, it’s safe; if not, remove this line:
            "force_abs_paths": "false",
        },
    )

    # Build robot_description_semantic (SRDF) for Servo
    # (ur_moveit_config ships ur.srdf.xacro)
    robot_description_semantic_xml = ""
    if srdf_xacro and os.path.exists(srdf_xacro):
        robot_description_semantic_xml = _run_xacro(
            srdf_xacro,
            {
                "name": robot_name,
                "prefix": prefix,
            },
        )

    # Load Servo params
    servo_params = {}
    if servo_yaml_path and os.path.exists(servo_yaml_path):
        servo_yaml = _load_yaml_file(servo_yaml_path)
        
        # LOGIC FIX: Check if file is nested or flat
        if "moveit_servo" in servo_yaml:
            # It is nested (standard ROS style)
            servo_params = servo_yaml["moveit_servo"]["ros__parameters"]
        else:
            # It is flat (your style)
            servo_params = servo_yaml

    # ROS nodes providing "robot state"
    # This answers your “where is the robot state?” question:
    # - joint_state_publisher -> /joint_states
    # - robot_state_publisher -> /tf, /tf_static
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"robot_description": robot_description_xml},
        ],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"robot_description": robot_description_xml},
        ],
    )

    # Spawn robot into Ignition/Gazebo Sim from /robot_description topic
    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-name",
            robot_name,
            "-topic",
            "robot_description",
            "-x",
            "0",
            "-y",
            "0",
            "-z",
            str(spawn_z),
        ],
    )

    # Bridges
    # NOTE: Your previous working bridge used ignition.msgs.Wrench — keep that.
    # If your system uses Gazebo Harmonic+ you may need gz.msgs.Wrench instead.
    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
        ],
    )

    ft_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/force_torque@geometry_msgs/msg/WrenchStamped[ignition.msgs.Wrench",
        ],
    )

    # MoveIt Servo (NOT optional)
    servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="servo_node",
        output="screen",
        condition=IfCondition(launch_servo),
        parameters=[
            {"use_sim_time": use_sim_time},
            {"robot_description": robot_description_xml},
            {"robot_description_semantic": robot_description_semantic_xml},
            servo_params,
        ],
    )

    # Your thesis admittance -> publishes TwistStamped to servo input
    # Your admittance_servo.py subscribes to:
    #   /force_torque_sensor_broadcaster/wrench
    # But we are bridging Ignition sensor to:
    #   /force_torque
    # So remap it here.
    admittance_servo = Node(
        package="my_thesis_controller",
        executable="admittance_servo",
        name="admittance_servo",
        output="screen",
        condition=IfCondition(launch_admittance),
        parameters=[{"use_sim_time": use_sim_time}],
        remappings=[
            ("/force_torque_sensor_broadcaster/wrench", "/force_torque"),
        ],
    )
    load_velocity_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["forward_velocity_controller", "--inactive"], # Start inactive
    )
    # Start Gazebo (Ignition)
    ign = ExecuteProcess(
        cmd=[
            "ign",
            "gazebo",
            "-r",
            world_path,
        ],
        output="screen",
    )

    # Delay spawn so the world is ready
    spawn_delayed = TimerAction(period=3.0, actions=[spawn_entity])

    return [
        ign,
        clock_bridge,
        ft_bridge,
        joint_state_publisher,
        robot_state_publisher,
        spawn_delayed,
        servo_node,
        admittance_servo,
        load_velocity_controller,
    ]


def generate_launch_description():
    # Default world path: use your installed ft_world.sdf from ur_simulation_gazebo
    ur_sim_share = get_package_share_directory("ur_simulation_gazebo")
    default_world = os.path.join(ur_sim_share, "worlds", "ft_world.sdf")

    # Fix “not full UR3e” in Ignition: make model://ur_description/... resolvable
    # model://ur_description/... resolves if the parent "share" folder is in the resource path.
    ur_desc_share_parent = os.path.join(get_package_prefix("ur_description"), "share")

    # Append to existing env if present
    existing_gz = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    new_gz = ur_desc_share_parent if not existing_gz else (existing_gz + ":" + ur_desc_share_parent)

    existing_ign = os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")
    new_ign = ur_desc_share_parent if not existing_ign else (existing_ign + ":" + ur_desc_share_parent)

    declared_args = [
        DeclareLaunchArgument("ur_type", default_value="ur3e"),
        DeclareLaunchArgument("robot_name", default_value="ur3e"),
        DeclareLaunchArgument("prefix", default_value=""),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("world", default_value=default_world),
        DeclareLaunchArgument("spawn_z", default_value="0.05"),
        DeclareLaunchArgument("launch_servo", default_value="true"),
        DeclareLaunchArgument("launch_admittance", default_value="true"),
    ]

    return LaunchDescription(
        declared_args
        + [
            SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", new_gz),
            SetEnvironmentVariable("IGN_GAZEBO_RESOURCE_PATH", new_ign),
            OpaqueFunction(function=_launch_setup),
        ]
    )

