from setuptools import find_packages, setup

package_name = 'my_thesis_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/rviz', [
            'rviz/view_robot.rviz',
            'rviz/hri_sim.rviz',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nishanth Sundaran',
    maintainer_email='sundharnishanth@gmail.com',
    description='Human-robot interaction with force feedback on UR3e — thesis code.',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'ft_zeroer = my_thesis_controller.ftzeroer:main',
            'hybrid_control_node = my_thesis_controller.hybrid_control_node:main',
            'interaction_classifier = my_thesis_controller.human_interaction_classifier:main',
            'target_pose_integrator = my_thesis_controller.targetposeintegratornode:main',
            'setup_planning_scene = my_thesis_controller.setupplanningscenenode:main',
            'gripper_joint_state_pub = my_thesis_controller.gripper_joint_state_publisher:main',
            'move_to_home = my_thesis_controller.move_to_home:main',
            'octomap_gate = my_thesis_controller.octomap_gate:main',
            'human_excluded_cloud_filter = my_thesis_controller.human_excluded_cloud_filter:main',
            'robotiq_urscript_bridge = my_thesis_controller.robotiq_urscript_bridge:main',
            'assistive_lift_v4 = my_thesis_controller.assistive_lift_v4_node:main',
        ],
    },
)
