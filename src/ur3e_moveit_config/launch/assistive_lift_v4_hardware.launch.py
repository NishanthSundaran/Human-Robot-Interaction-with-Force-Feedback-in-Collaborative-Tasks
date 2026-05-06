"""
Assistive Lift V4 — Human-Guided Placement (HARDWARE).

Same task as v4_sim but on real UR3e + Robotiq 2F-140 + RealSense D435i.

Minimal hardware launch — matches sim as closely as possible.
OctoMap, self-filter, safety monitor, etc. can be added later.

Key difference from sim: the UR driver uses ur_hardware.urdf.xacro (standard
ur_description macro with real ros2_control HW interfaces for both arm and
Robotiq gripper via tool communication).  RSP/MoveIt/Servo use the visual
URDF from ur_description_gz which has the same kinematic chain.
"""

import os
import subprocess
import yaml

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # -----------------------------
    # Launch arguments
    # -----------------------------
    robot_ip = LaunchConfiguration("robot_ip")
    ur_type = LaunchConfiguration("ur_type")
    launch_rviz = LaunchConfiguration("launch_rviz")
    use_robotiq = LaunchConfiguration("use_robotiq")
    _rviz_enabled = LaunchConfiguration("_rviz_enabled")

    declare_args = [
        DeclareLaunchArgument("robot_ip", default_value="192.168.123.3"),
        DeclareLaunchArgument("ur_type", default_value="ur3e"),
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        DeclareLaunchArgument("use_robotiq", default_value="true"),
        DeclareLaunchArgument("_rviz_enabled", default_value=launch_rviz),
    ]

    # -----------------------------
    # Packages / paths
    # -----------------------------
    pkg_ur_moveit = get_package_share_directory("ur3e_moveit_config")
    pkg_ur_description = get_package_share_directory("ur_description_gz")
    # Navigate from install/<pkg>/share/<pkg> up 4 levels to workspace root
    robot_calibration_file = os.path.realpath(
        os.path.join(pkg_ur_moveit, "..", "..", "..", "..", "robot_calibration.yaml")
    )
    if not os.path.isfile(robot_calibration_file):
        robot_calibration_file = os.path.join(
            pkg_ur_description, "config", "ur3e", "default_kinematics.yaml"
        )

    # ---- URDF for RSP / MoveIt / Servo (visual model with all links) ----
    urdf_xacro = PathJoinSubstitution(
        [FindPackageShare("ur_description_gz"), "urdf", "ur.urdf.xacro"]
    )
    robot_description_content = Command([
        FindExecutable(name="xacro"), " ",
        urdf_xacro,
        " name:=ur3e",
        " ur_type:=", ur_type,
        " use_fake_hardware:=false",
        " robotiq_ros2_control:=true",
        " include_camera:=false",
        " kinematics_params:=", robot_calibration_file,
    ])
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    # Pre-resolved URDF string (for nodes that can't handle Command substitutions)
    urdf_xacro_path_str = os.path.join(pkg_ur_description, "urdf", "ur.urdf.xacro")
    urdf_content = subprocess.check_output(
        ["/opt/ros/humble/bin/xacro", urdf_xacro_path_str,
         "name:=ur3e", "ur_type:=ur3e", "use_fake_hardware:=false",
         "robotiq_ros2_control:=true", "include_camera:=false",
         "kinematics_params:=" + robot_calibration_file],
        stderr=subprocess.STDOUT,
    ).decode("utf-8")
    robot_description_string = {"robot_description": urdf_content}

    # ---- SRDF ----
    srdf_xacro_path = os.path.join(pkg_ur_moveit, "srdf", "ur3e.srdf.xacro")
    srdf_content = subprocess.check_output(
        ["/opt/ros/humble/bin/xacro", srdf_xacro_path, "include_camera:=false"],
        stderr=subprocess.STDOUT,
    ).decode("utf-8")
    robot_description_semantic = {"robot_description_semantic": srdf_content}

    # ---- Config files ----
    servo_yaml = os.path.join(pkg_ur_moveit, "config", "servo.yaml")
    moveit_kinematics_yaml = os.path.join(pkg_ur_moveit, "config", "moveitkinematics.yaml")
    kinematics_yaml = os.path.join(pkg_ur_moveit, "config", "kinematics.yaml")
    hybrid_yaml_path = os.path.join(pkg_ur_moveit, "config", "hybrid_controller.yaml")
    with open(hybrid_yaml_path, "r") as f:
        hybrid_yaml_raw = yaml.safe_load(f)
    # Extract ros__parameters (strip ROS 2 params-file nesting)
    hybrid_params = hybrid_yaml_raw.get(
        "hybrid_control_node", hybrid_yaml_raw
    ).get("ros__parameters", hybrid_yaml_raw)

    joint_limits_yaml_path = os.path.join(pkg_ur_moveit, "config", "joint_limits.yaml")
    with open(joint_limits_yaml_path, "r") as f:
        joint_limits_raw = yaml.safe_load(f)
    pilz_cartesian_limits_path = os.path.join(pkg_ur_moveit, "config", "pilz_cartesian_limits.yaml")
    with open(pilz_cartesian_limits_path, "r") as f:
        pilz_cartesian_limits = yaml.safe_load(f)
    joint_limits_config = {"robot_description_planning": {**joint_limits_raw, **pilz_cartesian_limits}}

    ompl_planning_yaml_path = os.path.join(pkg_ur_moveit, "config", "ompl_planning.yaml")
    with open(ompl_planning_yaml_path, "r") as f:
        ompl_planning_config = yaml.safe_load(f)

    moveit_controllers_yaml_path = os.path.join(pkg_ur_moveit, "config", "moveit_controllers.yaml")
    with open(moveit_controllers_yaml_path, "r") as f:
        moveit_controllers_config = yaml.safe_load(f)
    moveit_controllers_config.pop("controller_manager", None)

    # --- MoveGroup controller config (hardware uses scaled_joint_trajectory_controller) ---
    hw_controller_config = {
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "moveit_simple_controller_manager": {
            "controller_names": ["scaled_joint_trajectory_controller"],
            "scaled_joint_trajectory_controller": {
                "type": "FollowJointTrajectory",
                "action_ns": "follow_joint_trajectory",
                "default": True,
                "joints": [
                    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
                ],
            },
        },
    }

    # =====================================================================
    # Nodes
    # =====================================================================

    # --- 1) Real UR driver (hardware URDF — arm ros2_control only) ---
    # Robotiq ros2_control is NOT included (its HW interface needs /tmp/ttyUR
    # which only exists after UR connects).  Physical gripper is controlled
    # via URScript bridge node instead.
    ur_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("ur_robot_driver"), "launch", "ur_control.launch.py"
            ])
        ),
        launch_arguments={
            "ur_type": ur_type,
            "robot_ip": robot_ip,
            "launch_rviz": "false",
            "launch_robot_state_publisher": "false",
            "activate_joint_controller": "true",
            "description_package": "ur_description_gz",
            "description_file": "ur_hardware.urdf.xacro",
            "kinematics_params_file": robot_calibration_file,
            "use_tool_communication": "false",
        }.items(),
    )

    # --- 2) Full-model RSP (arm + gripper + camera TFs) ---
    full_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="full_robot_state_publisher",
        output="log",
        parameters=[robot_description_string],
    )

    # --- Hand-eye calibrated camera TF (base_link → camera_link) ---
    # NOTE: Original calibration output was base_link → optical_frame.
    # Corrected here for base_link → camera_link by removing the
    # RealSense internal camera_link → optical rotation.
    camera_static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_handeye_tf",
        arguments=[
            "--x", "0.0137", "--y", "1.0702", "--z", "0.9151",
            "--qx", "0.207982", "--qy", "0.210572",
            "--qz", "-0.675404", "--qw", "0.675449",
            "--frame-id", "base_link",
            "--child-frame-id", "camera_link",
        ],
        parameters=[{"use_sim_time": False}],
    )

    # --- 4) FT zeroer ---
    ft_zeroer = Node(
        package="my_thesis_controller",
        executable="ft_zeroer",
        name="ft_zeroer",
        output="screen",
        parameters=[{"use_sim_time": False}],
    )

    # --- 5) MoveIt Servo ---
    servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="servo_node",
        output="screen",
        parameters=[
            servo_yaml,
            kinematics_yaml,
            moveit_kinematics_yaml,
            robot_description,
            robot_description_semantic,
            {"use_sim_time": False},
        ],
        remappings=[("planning_scene", "monitored_planning_scene")],
    )

    # --- 6) move_group (NO OctoMap — same as sim) ---
    ompl_ns_config = {
        "planning_plugin": "ompl_interface/OMPLPlanner",
        "request_adapters": (
            "default_planner_request_adapters/AddTimeOptimalParameterization "
            "default_planner_request_adapters/FixWorkspaceBounds "
            "default_planner_request_adapters/FixStartStateBounds "
            "default_planner_request_adapters/FixStartStateCollision "
            "default_planner_request_adapters/FixStartStatePathConstraints"
        ),
        "start_state_max_bounds_error": 0.1,
    }
    ompl_ns_config.update(ompl_planning_config)

    pilz_ns_config = {
        "planning_plugin": "pilz_industrial_motion_planner/CommandPlanner",
        "request_adapters": "",
        "default_planner_config": "PTP",
    }

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        name="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            moveit_kinematics_yaml,
            joint_limits_config,
            {"planning_pipelines": {"pipeline_names": ["ompl", "pilz"]}},
            {"ompl": ompl_ns_config},
            {"pilz": pilz_ns_config},
            {"use_sim_time": False},
            hw_controller_config,
            {"moveit_manage_controllers": False,
             "trajectory_execution.allowed_execution_duration_scaling": 1.2,
             "trajectory_execution.allowed_goal_duration_margin": 0.5,
             "trajectory_execution.allowed_start_tolerance": 0.05},
            {"publish_planning_scene": True,
             "publish_geometry_updates": True,
             "publish_state_updates": True,
             "publish_transforms_updates": True},
        ],
    )

    # --- 7) Hybrid control node (after move_to_home starts — it disables hybrid immediately) ---
    hybrid_node = TimerAction(
        period=15.0,
        actions=[Node(
            package="my_thesis_controller",
            executable="hybrid_control_node",
            name="hybrid_control_node",
            output="screen",
            parameters=[
                {"use_sim_time": False},
                # Load hybrid_controller.yaml as dict (NOT --params-file) so
                # inline overrides below take effect (--params-file always wins).
                hybrid_params,
                {"wrench_topic": "/wrench_zeroed"},
                # Hardware: UR FT broadcaster uses tool0 frame (sim uses ft_frame)
                {"wrench_source_frame": "tool0"},
                # V4: Z compliance disabled (human guides XY, robot handles Z)
                {"selection_matrix": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]},
                {"guiding_damping_linear": 25.0},
                {"guiding_damping_angular": 4.0},
                {"max_linear_speed": 0.20},
                {"max_linear_accel": 0.7},
            ],
        )],
    )

    # --- 8) Target pose integrator (after hybrid_node — needs TFs) ---
    target_pose_integrator = TimerAction(
        period=16.0,
        actions=[Node(
            package="my_thesis_controller",
            executable="target_pose_integrator",
            name="target_pose_integrator",
            output="screen",
            parameters=[{"use_sim_time": False}],
        )],
    )

    # --- 9) RealSense camera driver ---
    realsense_node = TimerAction(
        period=5.0,
        actions=[Node(
            package="realsense2_camera",
            executable="realsense2_camera_node",
            name="camera",
            namespace="camera",
            parameters=[{
                "enable_depth": True,
                "enable_color": True,
                "pointcloud.enable": False,
                "align_depth.enable": True,
                "enable_gyro": False,
                "enable_accel": False,
                "use_sim_time": False,
                "spatial_filter.enable": True,
                "spatial_filter.smooth_alpha": 0.5,
                "spatial_filter.smooth_delta": 20,
                "temporal_filter.enable": True,
                "temporal_filter.smooth_alpha": 0.4,
                "temporal_filter.smooth_delta": 20,
                "threshold_filter.enable": True,
                "threshold_filter.min_distance": 0.15,
                "threshold_filter.max_distance": 2.5,
                "publish_tf": False,
            }],
            output="screen",
        )],
    )

    # --- Camera internal TF (camera_link → camera_color_optical_frame) ---
    # The RealSense driver publishes a wrong rotation after startup, so we
    # disable its TF (publish_tf: False above) and broadcast the D435i URDF
    # nominal values ourselves: rpy(-π/2, 0, -π/2) = quat(-0.5, 0.5, -0.5, 0.5)
    camera_internal_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_internal_tf",
        arguments=[
            "--x", "0.0", "--y", "0.015", "--z", "0.0",
            "--qx", "-0.5", "--qy", "0.5",
            "--qz", "-0.5", "--qw", "0.5",
            "--frame-id", "camera_link",
            "--child-frame-id", "camera_color_optical_frame",
        ],
        parameters=[{"use_sim_time": False}],
    )

    # --- 10) Interaction classifier ---
    # NOTE: interaction_classifier is started by the v4 node after lift,
    # not at launch time, to avoid false triggers during the pick sequence.

    # --- 11) Planning scene setup (delayed) ---
    scene_setup = TimerAction(
        period=15.0,
        actions=[Node(
            package="my_thesis_controller",
            executable="setup_planning_scene",
            name="setup_planning_scene",
            output="screen",
            parameters=[{"use_sim_time": False}],
        )],
    )

    # --- 12) Move-to-home (delayed — controllers need ~5s to spawn) ---
    move_to_home = TimerAction(
        period=12.0,
        actions=[Node(
            package="my_thesis_controller",
            executable="move_to_home",
            name="move_to_home",
            output="screen",
            parameters=[{"use_sim_time": False}],
        )],
    )

    # --- 13) Start Servo (after move_to_home completes — avoids singularity spam at startup pose) ---
    start_servo = TimerAction(
        period=25.0,
        actions=[ExecuteProcess(
            cmd=["ros2", "service", "call",
                 "/servo_node/start_servo",
                 "std_srvs/srv/Trigger", "{}"],
            output="screen",
        )],
    )

    # --- 14) Gripper joint state publisher (publishes default gripper joint states for RSP) ---
    gripper_joint_state_publisher = Node(
        package="my_thesis_controller",
        executable="gripper_joint_state_pub",
        name="gripper_joint_state_publisher",
        output="log",
        parameters=[{"use_sim_time": False}],
    )

    # --- 15) Robotiq URCap bridge (controls physical gripper via TCP socket 63352) ---
    robotiq_bridge = TimerAction(
        period=10.0,
        actions=[Node(
            condition=IfCondition(use_robotiq),
            package="my_thesis_controller",
            executable="robotiq_urscript_bridge",
            name="robotiq_urscript_bridge",
            output="screen",
            parameters=[{
                "use_sim_time": False,
                "gripper_topic": "/robotiq_gripper_command",
                "robot_ip": "192.168.123.3",
                "speed": 128,   # 0-255, collaborative-safe (was 255 max)
                "force": 100,   # 0-255, gentle grip (was 128)
            }],
        )],
    )

    # --- 16) Open gripper at startup (after bridge connects ~15s) ---
    open_gripper = TimerAction(
        period=15.0,
        actions=[ExecuteProcess(
            cmd=["ros2", "topic", "pub", "--once",
                 "/robotiq_gripper_command",
                 "std_msgs/msg/Float64MultiArray",
                 "{data: [0.0]}"],
            output="screen",
        )],
    )

    # --- 17) publish_enabled REMOVED — v4_node and move_to_home manage
    #         hybrid enable/disable. The fixed-delay publish was racing with
    #         ongoing MoveGroup trajectories, causing controller deactivation.

    # --- 18) Assistive Lift V4 node (delayed 25s — after homing) ---
    assistive_lift_v4_node = TimerAction(
        period=25.0,
        actions=[Node(
            package="my_thesis_controller",
            executable="assistive_lift_v4",
            name="assistive_lift_v4",
            output="screen",
            parameters=[{
                "use_sim_time": False,
                "use_gripper": True,
                "hardware_gripper": True,
                "jtc_name": "scaled_joint_trajectory_controller",
                "gripper_topic": "/robotiq_gripper_command",
                "rgb_topic": "/camera/camera/color/image_raw",
                "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
                "camera_info_topic": "/camera/camera/color/camera_info",
                "camera_optical_frame": "camera_color_optical_frame",
            }],
        )],
    )

    # --- 17) RViz (optional) ---
    rviz_config_file = os.path.join(
        get_package_share_directory("my_thesis_controller"),
        "rviz", "view_robot.rviz")

    rviz_node = Node(
        condition=IfCondition(_rviz_enabled),
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
        parameters=[
            robot_description,
            robot_description_semantic,
            moveit_kinematics_yaml,
            joint_limits_config,
            {"planning_pipelines": {"pipeline_names": ["ompl", "pilz"]}},
            {"ompl": ompl_ns_config},
            {"pilz": pilz_ns_config},
            {"use_sim_time": False},
        ],
    )

    return LaunchDescription(
        declare_args
        + [
            ur_driver,
            full_robot_state_publisher,
            ft_zeroer,
            realsense_node,
            camera_static_tf,
            camera_internal_tf,
            move_group,
            servo_node,
            move_to_home,
            start_servo,
            hybrid_node,
            target_pose_integrator,
            scene_setup,
            gripper_joint_state_publisher,
            robotiq_bridge,
            open_gripper,
            rviz_node,
            assistive_lift_v4_node,
        ]
    )
