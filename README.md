# Human-Robot Interaction with Force Feedback in Collaborative Tasks

Master's thesis code by **Nishanth Sundaran**.

This repository contains the ROS 2 Humble code for a force-feedback-based
human-robot interaction system on a UR3e collaborative robot. The system
demonstrates an *assistive lift* task where the robot picks an object,
hands it off to a human via admittance-based hand-guiding, and places it
at the human-chosen location — using force/torque feedback as the primary
interaction modality.

## Hardware

- **Robot**: Universal Robots UR3e (with built-in F/T sensor)
- **Gripper**: Robotiq 2F-140 (controlled via URCap TCP socket)
- **Camera**: Intel RealSense D435i (hand-eye calibrated)
- **OS / middleware**: Ubuntu 22.04, ROS 2 Humble

## Repository layout

```
src/
├── my_thesis_controller/        # Custom thesis nodes
│   ├── my_thesis_controller/
│   │   ├── ftzeroer.py                       # FT bias / re-tare service
│   │   ├── hybrid_control_node.py            # Hybrid position + admittance controller
│   │   ├── human_interaction_classifier.py   # IDLE / GUIDING / NUDGE state machine
│   │   ├── targetposeintegratornode.py       # Streaming target-pose state holder
│   │   ├── setupplanningscenenode.py         # MoveIt collision world setup
│   │   ├── move_to_home.py                   # Homing service + hybrid restore
│   │   ├── gripper_joint_state_publisher.py  # Default gripper joint states for RSP
│   │   ├── robotiq_urscript_bridge.py        # Robotiq 2F-140 control via URCap TCP
│   │   ├── assistive_lift_v4_node.py         # Top-level FSM: pick → guide → place
│   │   ├── human_excluded_cloud_filter.py    # YOLOv8-seg: drop human pixels from cloud
│   │   ├── octomap_gate.py                   # Block cloud during GUIDING (clears OctoMap)
│   │   └── __init__.py
│   ├── rviz/
│   │   ├── view_robot.rviz                   # Hardware RViz config
│   │   └── hri_sim.rviz                      # Sim HRI RViz config
│   ├── package.xml
│   └── setup.py
└── ur3e_moveit_config/          # MoveIt 2 + launch files
    ├── config/                  # MoveIt configs (servo, kinematics, planners, OctoMap)
    ├── launch/
    │   ├── assistive_lift_v4_hardware.launch.py       # Real UR3e (no human-pipeline)
    │   ├── assistive_lift_v4_hardware_hri.launch.py   # Real UR3e + human exclusion
    │   └── assistive_lift_v4_hri_sim.launch.py        # Gazebo Ignition sim + HRI
    ├── srdf/ur3e.srdf.xacro
    ├── worlds/
    │   └── assistive_lift_world_hri.sdf               # Gazebo world (walking actor)
    ├── package.xml
    └── CMakeLists.txt
```

## System architecture

Pipeline at runtime:

```
F/T sensor (UR3e) ──► ft_zeroer ──► /wrench_zeroed
                                        │
                              hybrid_control_node ◄── /interaction_state
                                        │              (from interaction_classifier)
                                        ▼
                          /servo_node/delta_twist_cmds
                                        │
                                MoveIt Servo ──► UR robot driver

RealSense D435i ──► /camera/camera/depth/color/points
                                        │
                              robot_self_filter ──► /cloud_no_robot
                                        │
                  human_excluded_cloud_filter ──► /cloud_no_human
                  (YOLOv8-seg person mask)         │
                                        ▼
                               octomap_gate ──► /cloud_gated
                                                  │
                                          MoveIt OctoMap
```

## Dependencies

### ROS 2 packages installed via apt

```
sudo apt install \
  ros-humble-moveit \
  ros-humble-moveit-servo \
  ros-humble-moveit-ros-move-group \
  ros-humble-controller-manager \
  ros-humble-joint-state-broadcaster \
  ros-humble-realsense2-camera \
  ros-humble-tf2-ros \
  ros-humble-tf2-geometry-msgs \
  ros-humble-cv-bridge \
  ros-humble-sensor-msgs-py \
  ros-humble-rviz2 \
  ros-humble-ros-gz-sim \
  ros-humble-ros-gz-bridge \
  python3-colcon-common-extensions
```

### Python packages

```
pip3 install ultralytics opencv-python numpy<2 scipy
```

(numpy must be `<2` because ROS 2 Humble's `cv_bridge` segfaults under numpy 2.x.)

### Source dependencies (clone into `src/`)

These are not included in this repo; clone them from upstream:

| Package                                 | Purpose                                          |
|-----------------------------------------|--------------------------------------------------|
| `Universal_Robots_ROS2_Description`     | UR robot URDF / meshes (used for sim and viz)    |
| `Universal_Robots_ROS2_Driver`          | UR hardware driver, `ur_control.launch.py`       |
| `robot_self_filter`                     | Removes robot links from RealSense point cloud   |
| `ros2_robotiq_gripper`                  | Robotiq URDF (referenced from UR description)    |

A `dependencies.repos` file is included for `vcs import`:

```
vcs import src < dependencies.repos
```

> **Note**: the launches reference a forked package called `ur_description_gz`
> with custom camera-mount geometry (D435i hand-eye fixture). If the upstream
> `Universal_Robots_ROS2_Description` is used as-is, you may need to provide
> a static TF for the camera or supply your own description package.

## Build

```
cd thesis_share
vcs import src < dependencies.repos      # pull third-party deps
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## Run — three launch options

### 1. Hardware (no human-pipeline)

```
ros2 launch ur3e_moveit_config assistive_lift_v4_hardware.launch.py \
  ur_type:=ur3e robot_ip:=192.168.123.3
```

Bring up real UR3e + Robotiq + RealSense. **Press *Play* on the teach
pendant** when the driver prompts.

### 2. Hardware + human detection / exclusion

```
ros2 launch ur3e_moveit_config assistive_lift_v4_hardware_hri.launch.py \
  ur_type:=ur3e robot_ip:=192.168.123.3
```

Same as (1) plus the YOLOv8-seg human-exclusion pipeline so MoveIt's
OctoMap ignores any human in the workspace. Diagnostic image is published
on `/human_excluded_cloud_filter/debug_image`.

### 3. Simulation (Gazebo Ignition + HRI)

```
ros2 launch ur3e_moveit_config assistive_lift_v4_hri_sim.launch.py
```

Spawns the UR3e in a Gazebo world with a walking actor. Same control
pipeline as the hardware HRI launch — useful for review without hardware.

## Task flow (the FSM, V4)

```
INIT → PERCEIVE → PLAN_GRASP → MOVE_TO_PREGRASP → MOVE_TO_GRASP → GRASP
     → LIFT → ENABLE_SERVO → AWAIT_INTERACTION (human guides)
     → [double-tap] → PREP_PLACE → DESCEND_TO_PLACE → PLACE → RETRACT → DONE
```

During **AWAIT_INTERACTION**, the human grabs the tool, the classifier
detects sustained contact (GUIDING), and the hybrid controller switches
from position-hold to admittance compliance. A **double-tap** on the
gripper signals "place here", triggering descent.

## Debug topics

| Topic                                      | Purpose                                |
|--------------------------------------------|----------------------------------------|
| `/wrench_zeroed`                           | Bias-corrected F/T from UR3e sensor    |
| `/interaction_state`                       | IDLE / GUIDING / NUDGE                 |
| `/effective_interaction_state`             | Debounced GUIDING state                |
| `/servo_node/delta_twist_cmds`             | Velocity commands to MoveIt Servo      |
| `/cloud_no_robot`                          | Self-filtered cloud                    |
| `/cloud_no_human`                          | Human-excluded cloud (YOLOv8-seg)      |
| `/cloud_gated`                             | Cloud delivered to OctoMap             |
| `/human_excluded_cloud_filter/debug_image` | RGB with detected person overlay       |

## License

Apache 2.0. See LICENSE.

## Contact

Nishanth Sundaran — sundharnishanth@gmail.com
