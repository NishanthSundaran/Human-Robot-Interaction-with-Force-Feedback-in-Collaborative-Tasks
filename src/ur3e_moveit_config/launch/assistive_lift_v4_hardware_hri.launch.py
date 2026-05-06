"""
Assistive Lift V4 — Human-Guided Placement (HARDWARE + HRI).

Same as assistive_lift_v4_hardware.launch.py but adds the human detection /
exclusion pipeline so MoveIt's OctoMap ignores humans in the workspace:

    RealSense /camera/camera/depth/color/points
       -> robot_self_filter    -> /cloud_no_robot
       -> human_excluded_filter-> /cloud_no_human   (YOLOv8-seg person mask)
       -> octomap_gate         -> /cloud_gated     (blanks during GUIDING)
       -> MoveIt OctoMap (sensors_3d.yaml)
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

    # OctoMap sensor config (PointCloudOctomapUpdater on /cloud_gated)
    sensors_3d_yaml_path = os.path.join(pkg_ur_moveit, "config", "sensors_3d.yaml")
    with open(sensors_3d_yaml_path, "r") as f:
        sensors_3d_config = yaml.safe_load(f)

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

    # --- 1) Real UR driver ---
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

    # --- 2) Full-model RSP ---
    full_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="full_robot_state_publisher",
        output="log",
        parameters=[robot_description_string],
    )

    # --- Hand-eye calibrated camera TF (base_link → camera_link) ---
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

    # --- 6) move_group (WITH OctoMap) ---
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
            # OctoMap / 3D sensor manager
            sensors_3d_config,
            {"octomap_resolution": 0.025,
             "octomap_frame_id": "base_link",
             "moveit_sensor_manager": "occupancy_map_monitor/PointCloudOctomapUpdater"},
        ],
    )

    # --- 7) Hybrid control node ---
    hybrid_node = TimerAction(
        period=15.0,
        actions=[Node(
            package="my_thesis_controller",
            executable="hybrid_control_node",
            name="hybrid_control_node",
            output="screen",
            parameters=[
                {"use_sim_time": False},
                hybrid_params,
                {"wrench_topic": "/wrench_zeroed"},
                {"wrench_source_frame": "tool0"},
                {"selection_matrix": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]},
                {"guiding_damping_linear": 25.0},
                {"guiding_damping_angular": 4.0},
                {"max_linear_speed": 0.20},
                {"max_linear_accel": 0.7},
            ],
        )],
    )

    # --- 8) Target pose integrator ---
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

    # --- 9) RealSense camera driver (pointcloud ENABLED for HRI pipeline) ---
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
                "pointcloud.enable": True,
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

    # --- Depth optical TF (camera_link → camera_depth_optical_frame) ---
    # Required because URDF is built with include_camera:=false, so depth
    # frame isn't in TF tree. RealSense publishes point cloud in this frame.
    # D435i depth imager is ~15mm left of color imager; approximate here.
    camera_depth_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_depth_tf",
        arguments=[
            "--x", "0.0", "--y", "0.0", "--z", "0.0",
            "--qx", "-0.5", "--qy", "0.5",
            "--qz", "-0.5", "--qw", "0.5",
            "--frame-id", "camera_link",
            "--child-frame-id", "camera_depth_optical_frame",
        ],
        parameters=[{"use_sim_time": False}],
    )

    # --- Robot self-filter (removes robot links from RealSense point cloud) ---
    robot_self_filter = TimerAction(
        period=6.0,
        actions=[Node(
            package="robot_self_filter",
            executable="self_filter",
            name="robot_self_filter",
            parameters=[robot_description, {
                "use_sim_time": False,
                "sensor_frame": "camera_depth_optical_frame",
                "min_sensor_dist": 0.01,
                "default_box_padding": [0.05, 0.05, 0.05],
                "default_cylinder_padding": [0.05, 0.05],
                "default_sphere_padding": 0.05,
                "self_see_links.names": [
                    "base_link_inertia",
                    "shoulder_link",
                    "upper_arm_link",
                    "forearm_link",
                    "wrist_1_link",
                    "wrist_2_link",
                    "wrist_3_link",
                    "robotiq_140_robotiq_140_base_link",
                    "robotiq_140_left_outer_knuckle",
                    "robotiq_140_left_outer_finger",
                    "robotiq_140_left_inner_finger",
                    "robotiq_140_left_inner_finger_pad",
                    "robotiq_140_left_inner_knuckle",
                    "robotiq_140_right_outer_knuckle",
                    "robotiq_140_right_outer_finger",
                    "robotiq_140_right_inner_finger",
                    "robotiq_140_right_inner_finger_pad",
                    "robotiq_140_right_inner_knuckle",
                ],
            }],
            remappings=[
                ("cloud_in", "/camera/camera/depth/color/points"),
                ("cloud_out", "/cloud_no_robot"),
            ],
            output="screen",
        )],
    )

    # --- Human-pixel exclusion (YOLOv8-seg) ---
    human_excluded_filter = TimerAction(
        period=8.0,
        actions=[Node(
            package="my_thesis_controller",
            executable="human_excluded_cloud_filter",
            name="human_excluded_cloud_filter",
            output="screen",
            parameters=[{
                "use_sim_time": False,
                "model": "yolov8n-seg.pt",
                "confidence": 0.25,
                "rate_limit": 10.0,
                "mask_dilate_px": 10,
                "max_depth": 3.0,
                "min_depth": 0.2,
                "color_topic": "/camera/camera/color/image_raw",
                "camera_info_topic": "/camera/camera/color/camera_info",
                "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
                "cloud_in_topic": "/cloud_no_robot",
                "cloud_out_topic": "/cloud_no_human",
            }],
        )],
    )

    # --- OctoMap gate (blocks point cloud during GUIDING, clears OctoMap) ---
    octomap_gate = Node(
        package="my_thesis_controller",
        executable="octomap_gate",
        name="octomap_gate",
        output="screen",
        parameters=[{"use_sim_time": False}],
        remappings=[
            ("/cloud_filtered", "/cloud_no_human"),
        ],
    )

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

    # --- 12) Move-to-home ---
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

    # --- 13) Start Servo ---
    start_servo = TimerAction(
        period=25.0,
        actions=[ExecuteProcess(
            cmd=["ros2", "service", "call",
                 "/servo_node/start_servo",
                 "std_srvs/srv/Trigger", "{}"],
            output="screen",
        )],
    )

    # --- 14) Gripper joint state publisher ---
    gripper_joint_state_publisher = Node(
        package="my_thesis_controller",
        executable="gripper_joint_state_pub",
        name="gripper_joint_state_publisher",
        output="log",
        parameters=[{"use_sim_time": False}],
    )

    # --- 15) Robotiq URCap bridge ---
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
                "speed": 128,
                "force": 100,
            }],
        )],
    )

    # --- 16) Open gripper at startup ---
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

    # --- 18) Assistive Lift V4 node ---
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
            camera_depth_tf,
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
            robot_self_filter,
            human_excluded_filter,
            octomap_gate,
            rviz_node,
            assistive_lift_v4_node,
        ]
    )
