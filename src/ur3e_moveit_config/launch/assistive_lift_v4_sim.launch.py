"""
Assistive Lift V4 — Human-Guided Placement (SIMULATION).

Same infrastructure as V2 launch but uses the V4 node which:
  - Picks and lifts the object, then enters servo mode WITH the object
  - Human guides the robot to the desired placement location
  - On double tap (NUDGE), robot computes safe height and places the object

All motion planning uses OMPL/RRTConnect with self-collision + static
collision checking.  No raw FollowJointTrajectory calls.
"""

import os
import subprocess
import yaml

from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.substitutions import (
    Command,
    FindExecutable,
    PathJoinSubstitution,
)

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_ur_description = get_package_share_directory("ur_description_gz")
    pkg_ur_moveit = get_package_share_directory("ur3e_moveit_config")

    # --- Gazebo resource paths ---
    robotiq_share = get_package_share_directory("robotiq_description")
    extra_model_paths = [
        "/usr/share/gazebo-11/models",
        os.path.dirname(get_package_share_directory("ur_description_gz")),
        os.path.dirname(robotiq_share),
    ]
    existing_ign = os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")
    existing_gz = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    ign_value = ":".join([p for p in [existing_ign, *extra_model_paths] if p])
    gz_value = ":".join([p for p in [existing_gz, *extra_model_paths] if p])
    set_ign_path = SetEnvironmentVariable("IGN_GAZEBO_RESOURCE_PATH", ign_value)
    set_gz_path = SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", gz_value)

    # --- Robot description ---
    urdf_xacro = PathJoinSubstitution(
        [FindPackageShare("ur_description_gz"), "urdf", "ur.urdf.xacro"]
    )
    robot_description_content = Command([
        FindExecutable(name="xacro"), " ",
        urdf_xacro,
        " name:=ur3e",
        " ur_type:=ur3e",
        " use_fake_hardware:=false",
        " q_shoulder_pan:=0.0",
        " q_shoulder_lift:=-1.2",
        " q_elbow:=1.0",
        " q_wrist_1:=-1.37",
        " q_wrist_2:=-1.57",
        " q_wrist_3:=0.0",
    ])
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    # SRDF
    srdf_xacro_path = os.path.join(pkg_ur_moveit, "srdf", "ur3e.srdf.xacro")
    srdf_content = subprocess.check_output(
        ["/opt/ros/humble/bin/xacro", srdf_xacro_path],
        stderr=subprocess.STDOUT,
    ).decode("utf-8")
    robot_description_semantic = {"robot_description_semantic": srdf_content}

    # --- Config files ---
    servo_yaml = os.path.join(pkg_ur_moveit, "config", "servo.yaml")
    moveit_kinematics_yaml = os.path.join(pkg_ur_moveit, "config", "moveitkinematics.yaml")
    kinematics_yaml = os.path.join(pkg_ur_moveit, "config", "kinematics.yaml")
    hybrid_yaml = os.path.join(pkg_ur_moveit, "config", "hybrid_controller.yaml")

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

    sim_controller_config = {
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "moveit_simple_controller_manager": {
            "controller_names": ["joint_trajectory_controller"],
            "joint_trajectory_controller": {
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
    # OMPL planning pipeline config (shared by move_group and rviz)
    # =====================================================================
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

    # =====================================================================
    # Nodes
    # =====================================================================

    # --- Ignition Gazebo ---
    ignition = ExecuteProcess(
        cmd=[
            "ign", "gazebo", "-r",
            PathJoinSubstitution(
                [FindPackageShare("ur_simulation_gazebo"), "worlds", "assistive_lift_world.sdf"]
            ),
        ],
        output="screen",
    )

    # --- Spawn robot (63cm table height) ---
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-string", robot_description_content,
            "-name", "ur3e",
            "-allow_renaming", "true",
            "-z", "0.63",
        ],
    )

    # --- RSP ---
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description, {"use_sim_time": True}],
    )

    # --- Bridges ---
    bridge_clock = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock"],
        output="screen",
    )
    bridge_ft = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/force_torque@geometry_msgs/msg/WrenchStamped[ignition.msgs.Wrench"],
        output="screen",
    )

    # --- Camera bridges ---
    bridge_camera_rgb = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/realsense/image@sensor_msgs/msg/Image[ignition.msgs.Image"],
        output="screen",
    )
    bridge_camera_depth = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/realsense/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image"],
        output="screen",
    )
    bridge_camera_info = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/realsense/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo"],
        output="screen",
    )
    bridge_camera_points = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/realsense/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked"],
        output="screen",
    )

    # --- Camera frame bridge ---
    gz_camera_frame_bridge = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0", "0", "0", "0", "0", "0",
                   "camera_depth_frame",
                   "ur3e/camera_stand/depth_camera"],
        parameters=[{"use_sim_time": True}],
    )

    # --- Controller spawners ---
    spawner_js = Node(
        package="controller_manager", executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )
    spawner_vc = Node(
        package="controller_manager", executable="spawner",
        arguments=["forward_velocity_controller", "--controller-manager", "/controller_manager",
                   "--inactive"],
    )
    spawner_ft = Node(
        package="controller_manager", executable="spawner",
        arguments=["force_torque_sensor_broadcaster", "--controller-manager", "/controller_manager"],
    )
    spawner_jtc = Node(
        package="controller_manager", executable="spawner",
        arguments=["joint_trajectory_controller", "--controller-manager", "/controller_manager"],
    )

    # --- Robotiq gripper controller spawner ---
    spawner_robotiq = Node(
        package="controller_manager", executable="spawner",
        arguments=["robotiq_group_position_controller", "--controller-manager", "/controller_manager"],
    )

    # --- FT zeroer ---
    ft_zeroer = Node(
        package="my_thesis_controller",
        executable="ft_zeroer",
        name="ft_zeroer",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # --- MoveIt Servo ---
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
            {"use_sim_time": True},
        ],
        remappings=[("planning_scene", "monitored_planning_scene")],
    )

    # --- move_group ---
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
            {"use_sim_time": True},
            sim_controller_config,
            {"moveit_manage_controllers": False,
             "trajectory_execution.allowed_execution_duration_scaling": 1.2,
             "trajectory_execution.allowed_goal_duration_margin": 0.5,
             "trajectory_execution.allowed_start_tolerance": 0.01},
            {"publish_planning_scene": True,
             "publish_geometry_updates": True,
             "publish_state_updates": True,
             "publish_transforms_updates": True},
        ],
    )

    # --- Hybrid control node ---
    hybrid_node = TimerAction(
        period=12.0,
        actions=[Node(
            package="my_thesis_controller",
            executable="hybrid_control_node",
            name="hybrid_control_node",
            output="screen",
            parameters=[
                {"use_sim_time": True},
                hybrid_yaml,
                {"wrench_topic": "/wrench_zeroed"},
                # V4: Z compliance disabled (human guides XY, robot handles Z)
                {"selection_matrix": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]},
                {"guiding_damping_linear": 25.0},
                {"guiding_damping_angular": 4.0},
                {"max_linear_speed": 0.20},
                {"max_linear_accel": 0.7},
            ],
        )],
    )

    # --- Target pose integrator ---
    target_pose_integrator = TimerAction(
        period=13.0,
        actions=[Node(
            package="my_thesis_controller",
            executable="target_pose_integrator",
            name="target_pose_integrator",
            output="screen",
            parameters=[{"use_sim_time": True}],
        )],
    )

    # --- Interaction classifier ---
    # NOTE: interaction_classifier is started by the v4 node after lift,
    # not at launch time, to avoid false triggers during the pick sequence.

    # --- Planning scene setup (delayed) ---
    scene_setup = TimerAction(
        period=15.0,
        actions=[Node(
            package="my_thesis_controller",
            executable="setup_planning_scene",
            name="setup_planning_scene",
            output="screen",
            parameters=[{"use_sim_time": True}],
        )],
    )

    # --- Move-to-home (collision-aware via MoveGroup/RRTConnect) ---
    move_to_home = TimerAction(
        period=8.0,
        actions=[Node(
            package="my_thesis_controller",
            executable="move_to_home",
            name="move_to_home",
            output="screen",
            parameters=[{"use_sim_time": True}],
        )],
    )

    # --- Publish enabled=True after V4 node starts ---
    publish_enabled = TimerAction(
        period=30.0,
        actions=[ExecuteProcess(
            cmd=["ros2", "topic", "pub", "-r", "1", "-t", "5",
                 "/hybrid_controller/enabled",
                 "std_msgs/msg/Bool", "{data: true}"],
            output="screen",
        )],
    )

    # --- Open gripper at startup ---
    open_gripper = TimerAction(
        period=5.0,
        actions=[ExecuteProcess(
            cmd=["ros2", "topic", "pub", "--once",
                 "/robotiq_group_position_controller/commands",
                 "std_msgs/msg/Float64MultiArray",
                 "{data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}"],
            output="screen",
        )],
    )

    # --- Start Servo (delayed) ---
    start_servo = TimerAction(
        period=10.0,
        actions=[ExecuteProcess(
            cmd=["ros2", "service", "call",
                 "/servo_node/start_servo",
                 "std_srvs/srv/Trigger", "{}"],
            output="screen",
        )],
    )

    # --- Assistive Lift V4 node (delayed 20s — after homing + scene) ---
    assistive_lift_v4_node = TimerAction(
        period=20.0,
        actions=[Node(
            package="my_thesis_controller",
            executable="assistive_lift_v4",
            name="assistive_lift_v4",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "use_gripper": True,
                "hardware_gripper": False,
                "tool0_to_fingertip": 0.177,
                "jtc_name": "joint_trajectory_controller",
                "rgb_topic": "/realsense/image",
                "depth_topic": "/realsense/depth_image",
                "camera_info_topic": "/realsense/camera_info",
                "camera_optical_frame": "camera_depth_optical_frame",
            }],
        )],
    )

    # --- RViz (with full OMPL config for MotionPlanning plugin) ---
    rviz_config_file = os.path.join(
        get_package_share_directory("my_thesis_controller"),
        "rviz", "view_robot.rviz")
    rviz_node = Node(
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
            {"use_sim_time": True},
        ],
    )

    return LaunchDescription([
        set_ign_path,
        set_gz_path,
        ignition,
        bridge_clock,
        bridge_ft,
        bridge_camera_rgb,
        bridge_camera_depth,
        bridge_camera_info,
        bridge_camera_points,
        gz_camera_frame_bridge,
        rsp,
        spawn_robot,
        spawner_js,
        spawner_vc,
        spawner_ft,
        spawner_jtc,
        spawner_robotiq,
        open_gripper,
        ft_zeroer,
        move_group,
        servo_node,
        move_to_home,
        start_servo,
        publish_enabled,
        hybrid_node,
        target_pose_integrator,
        scene_setup,
        rviz_node,
        assistive_lift_v4_node,
    ])
