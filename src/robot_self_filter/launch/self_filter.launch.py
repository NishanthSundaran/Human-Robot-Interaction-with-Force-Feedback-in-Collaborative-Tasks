# self_filter.launch.py
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    description_name_arg = DeclareLaunchArgument(
        'description_name',
        default_value='/robot_description'
    )
    zero_for_removed_points_arg = DeclareLaunchArgument(
        'zero_for_removed_points',
        default_value='true'
    )
    lidar_sensor_type_arg = DeclareLaunchArgument(
        'lidar_sensor_type',
        default_value='2'
    )
    in_pointcloud_topic_arg = DeclareLaunchArgument(
        'in_pointcloud_topic',
        default_value='/cloud_in'
    )
    out_pointcloud_topic_arg = DeclareLaunchArgument(
        'out_pointcloud_topic',
        default_value='/cloud_out'
    )
    robot_description_arg = DeclareLaunchArgument(
        'robot_description'
    )
    filter_config_arg = DeclareLaunchArgument(
        'filter_config'
    )
    # Declare use_sim_time argument
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true', # Keep default as true for standalone use
        description='Use simulation (Gazebo) clock if true'
    )

    # Create a log action to print the config
    log_config = LogInfo(msg=LaunchConfiguration('filter_config'))

    self_filter_node = Node(
        package='robot_self_filter',
        executable='self_filter',
        name='self_filter',
        output='screen',
        parameters=[
            LaunchConfiguration('filter_config'),  # loads the YAML file
            {
                'lidar_sensor_type': LaunchConfiguration('lidar_sensor_type'),
                'robot_description': ParameterValue(
                    LaunchConfiguration('robot_description'),
                    value_type=str
                ),
                'zero_for_removed_points': LaunchConfiguration('zero_for_removed_points'),
                'use_sim_time': LaunchConfiguration('use_sim_time') # Use the launch argument
            }
        ],
        remappings=[
            ('/robot_description', LaunchConfiguration('description_name')),
            ('/cloud_in', LaunchConfiguration('in_pointcloud_topic')),
            ('/cloud_out', LaunchConfiguration('out_pointcloud_topic')),
        ],
    )

    return LaunchDescription([
        description_name_arg,
        zero_for_removed_points_arg,
        lidar_sensor_type_arg,
        in_pointcloud_topic_arg,
        out_pointcloud_topic_arg,
        robot_description_arg,
        filter_config_arg,
        use_sim_time_arg, # Add to launch description
        log_config,
        self_filter_node
    ])
