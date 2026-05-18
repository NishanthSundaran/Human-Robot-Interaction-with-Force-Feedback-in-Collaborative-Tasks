# Human-Robot Interaction with Force Feedback in Collaborative Tasks

Master's thesis code by **Nishanth Sundaran**.

ROS 2 Humble implementation of a force-feedback-driven human-robot
interaction system on a Universal Robots UR3e collaborative robot. The
robot autonomously perceives and picks an object from a worktable, then
**hands it off to a human via admittance-based hand-guiding** — letting
the human physically pull the held object to any reachable target. A
double-tap on the gripper signals "place here" and the robot completes
the placement. A YOLOv8-segmentation pipeline keeps any human in the
camera's view from polluting MoveIt's collision OctoMap, so the planner
treats the static workspace correctly while ignoring the operator.

The system covers all four thesis objectives: (1) safe HRI control using
the UR3e's built-in F/T sensor, (2) ROS 2 + MoveIt 2 motion planning
integration, (3) real-time force-feedback algorithms (hybrid position +
admittance control with adaptive damping, intent amplification, and
hysteretic contact detection), and (4) a validated end-to-end use case
(collaborative pick-and-place / assistive lift).

## Demo

<video src="media/hri_demo.webm" controls width="640">
  Your browser cannot render embedded video.
  <a href="media/hri_demo.webm">Download / watch the demo (webm)</a>
</video>

> If the embedded player does not render in your viewer, the file is
> available at [`media/hri_demo.webm`](media/hri_demo.webm).
> The demo shows the full task on real hardware: perception → grasp →
> lift → human-guided placement → double-tap → placement → home.

## Hardware

- **Robot**: Universal Robots UR3e (with built-in F/T sensor)
- **Gripper**: Robotiq 2F-140 (controlled via URCap TCP socket)
- **Camera**: Intel RealSense D435i (hand-eye calibrated)
- **OS / middleware**: Ubuntu 22.04, ROS 2 Humble

## Repository layout

```
robot_calibration.yaml             # Hand-eye + UR3e DH calibration (referenced at launch)
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
│   │   ├── force_gui.py                      # tkinter wrench-injection GUI (debug)
│   │   ├── hybrid_tuning_gui.py              # Live tuning sliders (debug)
│   │   ├── safety_monitor.py                 # ISO/TS 15066 PFL + SSM monitor (debug)
│   │   └── __init__.py
│   ├── rviz/
│   │   ├── view_robot.rviz                   # Hardware RViz config
│   │   └── hri_sim.rviz                      # Sim HRI RViz config
│   ├── package.xml
│   └── setup.py
├── ur3e_moveit_config/          # MoveIt 2 + launch files
│   ├── config/                  # MoveIt configs (servo, kinematics, planners, OctoMap, ros2_controllers)
│   ├── launch/
│   │   ├── assistive_lift_v4_hardware.launch.py       # Real UR3e (no human-pipeline)
│   │   ├── assistive_lift_v4_hardware_hri.launch.py   # Real UR3e + human exclusion
│   │   └── assistive_lift_v4_hri_sim.launch.py        # Gazebo Ignition sim + HRI
│   ├── srdf/ur3e.srdf.xacro
│   ├── worlds/
│   │   └── assistive_lift_world_hri.sdf               # Gazebo world (walking actor)
│   ├── package.xml
│   └── CMakeLists.txt
├── ur_description_gz/           # Customized fork of UR description (UR3e + camera mount)
│   ├── urdf/                    # ur.urdf.xacro, ur_hardware.urdf.xacro, macros, includes
│   ├── config/ur3e/             # joint_limits, default_kinematics, physical/visual params
│   ├── meshes/ur3e/             # Visual + collision meshes (UR3e only — other variants pruned)
│   ├── package.xml
│   └── CMakeLists.txt
├── robotiq_description/         # Robotiq 2F-140 / 2F-85 URDFs + meshes (bundled)
│   ├── urdf/                    # robotiq_2f_140_macro, 2f_140.ros2_control, etc.
│   ├── meshes/{visual,collision}/{2f_140,2f_85}/
│   ├── config/
│   └── package.xml
└── robot_self_filter/           # leggedrobotics fork (bundled)
    ├── src/, include/           # C++ self-filter node
    ├── launch/, params/
    └── package.xml
scripts/
├── calibrate_doubletap.py       # 5-round nudge calibration tool
└── test_doubletap.py            # double-tap verifier
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

## Setup (copy-paste, fresh Ubuntu 22.04 + ROS 2 Humble)

```bash
# 1. ROS 2 Humble + extras
sudo apt update
sudo apt install -y \
  ros-humble-moveit \
  ros-humble-moveit-servo \
  ros-humble-moveit-ros-move-group \
  ros-humble-controller-manager \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-joint-state-broadcaster \
  ros-humble-realsense2-description \
  ros-humble-realsense2-camera \
  ros-humble-tf2-ros \
  ros-humble-tf2-geometry-msgs \
  ros-humble-cv-bridge \
  ros-humble-sensor-msgs-py \
  ros-humble-rviz2 \
  ros-humble-ros-gz \
  ros-humble-ros-gz-bridge \
  ros-humble-ros-gz-sim \
  ros-humble-pcl-ros \
  ros-humble-filters \
  python3-colcon-common-extensions python3-vcstool python3-rosdep

# 2. Python packages
pip3 install "numpy<2" "opencv-python<4.11" scipy ultralytics

# 3. Clone + import deps + build
git clone https://github.com/NishanthSundaran/Human-Robot-Interaction-with-Force-Feedback-in-Collaborative-Tasks.git ~/thesis_ws
cd ~/thesis_ws
vcs import src < dependencies.repos
sudo rosdep init || true        # ok if already initialised
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

> `numpy` must be **<2** and `opencv-python` **<4.11** because ROS 2
> Humble's `cv_bridge` segfaults under numpy 2.x and newer OpenCV.

### What's bundled vs. external

| Package                | Source                                 |
|------------------------|----------------------------------------|
| `my_thesis_controller` | bundled (this repo)                    |
| `ur3e_moveit_config`   | bundled (this repo)                    |
| `ur_description_gz`    | bundled (custom fork, UR3e meshes only)|
| `robotiq_description`  | bundled (2F-140 + 2F-85 URDFs/meshes)  |
| `robot_self_filter`    | bundled (leggedrobotics fork)          |
| `Universal_Robots_ROS2_Driver` | `dependencies.repos` (vcs)     |
| `realsense2_description` | apt: `ros-humble-realsense2-description` |
| `pcl_ros`, `filters` (deps of self_filter) | apt              |
| MoveIt 2 / ros_gz / ros2_control | apt                          |

The hardware gripper is controlled via the custom `robotiq_urscript_bridge`
node (URCap TCP socket on port 63352), so the full `ros2_robotiq_gripper`
driver/controller package is **not** required — only the bundled
`robotiq_description` is needed for URDF/meshes.

## Run — three launch options

> **Hardware launches (1) and (2) require:** UR3e at the given IP, FT
> sensor enabled in URCap, RealSense D435i connected, Robotiq 2F-140 with
> the Robotiq URCap installed and selected as "Controlled by:
> Robotiq_Grippers" on the teach pendant. Without this hardware they
> cannot run. Only launch **(3)** runs without hardware.
>
> **First sim run** downloads the Gazebo Fuel actor mesh
> (`https://fuel.gazebosim.org/.../walk.dae`). Internet is required on the
> first launch; cached in `~/.gz/fuel/` afterward.

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

## Debug & tuning tools

The repo ships with a set of GUIs and scripts used during development and
calibration:

### Debug nodes (in `my_thesis_controller`)

| Node                 | Purpose                                                    |
|----------------------|------------------------------------------------------------|
| `force_gui`          | tkinter GUI to inject test wrenches on `/wrench_external` (lets you simulate human pushes without touching the robot). |
| `hybrid_tuning_gui`  | Live tkinter sliders for runtime tuning of admittance M / D / K-equivalent and damping params via `ros2 param set`. |
| `safety_monitor`     | ISO/TS 15066 compliance monitor — power & force limiting, speed & separation monitoring, hand-guiding speed cap. Publishes `/proximity/scale`, `/safety/status`. |

Run them stand-alone (after sourcing the workspace):

```
ros2 run my_thesis_controller force_gui
ros2 run my_thesis_controller hybrid_tuning_gui
ros2 run my_thesis_controller safety_monitor
```

### Calibration scripts (in `scripts/`)

| Script                       | Purpose                                                         |
|------------------------------|-----------------------------------------------------------------|
| `scripts/calibrate_doubletap.py` | Records 5 rounds of double-taps from `/wrench_zeroed`, prints recommended values for `nudge_threshold`, `z_score_thresh`, `nudge_cooldown_s`, and `DOUBLE_TAP_WINDOW_S`. |
| `scripts/test_doubletap.py`     | Stand-alone double-tap detector that subscribes to `/interaction/is_nudge` and prints tap count / gap timing — for verifying the calibrated values. |

Run while a hardware launch is active:

```
python3 scripts/calibrate_doubletap.py     # 5 rounds × ~6 s each
python3 scripts/test_doubletap.py          # interactive verifier
```

The currently calibrated values for the author's tap profile (used in
`assistive_lift_v4_node.py`) are: `nudge_threshold=4.0 N`,
`z_score_thresh=2.0`, `nudge_cooldown_s=0.15 s`, `DOUBLE_TAP_WINDOW_S=1.5 s`.

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
