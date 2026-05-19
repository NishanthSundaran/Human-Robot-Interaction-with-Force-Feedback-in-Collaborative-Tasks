<h1 align="center">Human-Robot Interaction with Force Feedback on a UR3e Cobot</h1>

<p align="center">
  <b>Master's thesis · THD Deggendorf · 2025–2026</b><br>
  <i>ROS 2 Humble · MoveIt 2 · YOLOv8 · RealSense D435i · ISO/TS 15066</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ROS%202-Humble-22314E?style=flat-square&logo=ros&logoColor=white" />
  <img src="https://img.shields.io/badge/MoveIt-2-0277BD?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square" />
  <img src="https://img.shields.io/badge/UR3e-supported-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/ISO%2FTS%2015066-compliant-success?style=flat-square" />
</p>

<p align="center">
  <img src="media/demo.gif" alt="HRI demo on real hardware" width="720"/>
</p>

> A complete real-time **Human-Robot Interaction framework** for safe assistive manipulation on a Universal Robots UR3e cobot. The robot autonomously perceives and grasps an object, then lets a human guide it to the placement location through admittance-based hand-leading. A double-tap gesture on the end-effector signals placement completion. YOLOv8 segmentation filters human presence from collision detection, enabling safe shared-workspace operation under ISO/TS 15066 SSM, PFL, and hand-guiding modes.

---

## Highlights

- **Hybrid position + admittance control** with adaptive damping and intent amplification.
- **Multimodal perception**: Intel RealSense D435i with intrinsic and hand-eye calibration, YOLOv8-seg human exclusion, HSV object detection, and depth back-projection into the robot base frame.
- **ISO/TS 15066 safety layer**: SSM (Speed and Separation Monitoring), PFL (Power and Force Limiting), hand-guiding.
- **13-state task FSM** integrating autonomous grasp planning with human-guided placement.
- **Three launch paths**: real hardware, hardware + full human-exclusion pipeline, Gazebo Ignition simulation with walking actor.
- **Reproducible**: dependency manifests, rosdep-managed external packages, Apache-2.0 licensed.

---

## System architecture

```
                         +-----------------------+
                         |   Intel RealSense     |
                         |        D435i          |
                         +-----------+-----------+
                                     |
                                     v
+----------------+    +------------------------------+
| YOLOv8-seg     |--->| human_excluded_cloud_filter  |---> OctoMap
+----------------+    +------------------------------+
                                     |
                                     v
              +----------------------+----------------------+
              |                                             |
+----------------------+                       +-------------------------+
|   ft_zeroer (F/T)    |---------------------> |   hybrid_control_node    |
+----------------------+                       +------------+------------+
                                                            |
                                                            v
                                              +-------------------------+
                                              |       MoveIt Servo      |
                                              +------------+------------+
                                                            |
                                                            v
                                              +-------------------------+
                                              |     UR3e ROS 2 driver   |
                                              +-------------------------+

         human_interaction_classifier  --->  IDLE / GUIDING / NUDGE FSM
         assistive_lift_v4_node        --->  13-state task FSM
         robotiq_urscript_bridge       --->  TCP socket to URCap (gripper)
```

---

## Task FSM

`INIT` → `PERCEIVE` → `PLAN_GRASP` → `MOVE_TO_PREGRASP` → `MOVE_TO_GRASP` → `GRASP` → `LIFT` →
`ENABLE_SERVO` → `AWAIT_INTERACTION` *(human guides via admittance)* → *(double-tap)* →
`PREP_PLACE` → `DESCEND_TO_PLACE` → `PLACE` → `RETRACT` → `DONE`

During **AWAIT_INTERACTION**, the human grabs the tool, the classifier
detects sustained contact (GUIDING), and the hybrid controller switches
from position-hold to admittance compliance. A **double-tap** on the
gripper signals "place here", triggering the descent.

---

## Repository layout

```
src/
├── my_thesis_controller/              [Custom thesis nodes]
│   ├── hybrid_control_node.py          Hybrid position + admittance controller
│   ├── human_interaction_classifier.py IDLE / GUIDING / NUDGE FSM
│   ├── assistive_lift_v4_node.py       Top-level 13-state task controller
│   ├── human_excluded_cloud_filter.py  YOLOv8-seg person masking on cloud
│   ├── octomap_gate.py                 Cloud gating during human guidance
│   ├── robotiq_urscript_bridge.py      Gripper TCP control via URCap
│   ├── ft_zeroer.py                    Force/torque bias removal
│   └── safety_monitor.py               ISO/TS 15066 SSM + PFL supervisor
├── ur3e_moveit_config/                MoveIt 2 configs and launch files
├── ur_description_gz/                 Customised UR3e URDF and meshes
└── robotiq_description/               Robotiq 2F-140/-85 URDF and meshes

scripts/
├── calibrate_doubletap.py             5-round tap threshold calibration
└── test_doubletap.py                  Tap verification tool

cpp/
└── ft_zeroer_cpp/                     C++ port of ft_zeroer node (rclcpp)
```

---

## Hardware

- **Robot**: Universal Robots UR3e (with built-in F/T sensor)
- **Gripper**: Robotiq 2F-140 (controlled via URCap TCP socket)
- **Camera**: Intel RealSense D435i (hand-eye calibrated)
- **OS / middleware**: Ubuntu 22.04, ROS 2 Humble

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
| `ur_simulation_gazebo` | bundled (sim worlds + bring-up launch) |
| `Universal_Robots_ROS2_Driver` | `dependencies.repos` (vcs)     |
| `realsense2_description` | apt: `ros-humble-realsense2-description` |
| `pcl_ros`, `filters` (deps of self_filter) | apt              |
| MoveIt 2 / ros_gz / ros2_control | apt                          |

The hardware gripper is controlled via the custom `robotiq_urscript_bridge`
node (URCap TCP socket on port 63352), so the full `ros2_robotiq_gripper`
driver/controller package is **not** required; only the bundled
`robotiq_description` is needed for URDF/meshes.

## Run (four launch options)

> **Hardware launches (1) and (2) require:** UR3e at the given IP, FT
> sensor enabled in URCap, RealSense D435i connected, Robotiq 2F-140 with
> the Robotiq URCap installed and selected as "Controlled by:
> Robotiq_Grippers" on the teach pendant. Without this hardware they
> cannot run. The two simulation launches **(3)** and **(4)** run without
> any hardware.
>
> **First HRI sim run** downloads the Gazebo Fuel walking-actor mesh
> (`https://fuel.gazebosim.org/.../walk.dae`). Internet is required on
> the first launch; the asset is cached in `~/.gz/fuel/` afterward.

### 1. Hardware (no human pipeline)

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

### 3. Simulation (Gazebo Ignition, no human pipeline)

```
ros2 launch ur3e_moveit_config assistive_lift_v4_sim.launch.py
```

Spawns the UR3e in a static Gazebo Ignition world for the full pick →
guide → place task. No walking actor, no human-exclusion pipeline.
Useful as a smoke test or first demo without hardware.

### 4. Simulation (Gazebo Ignition + HRI)

```
ros2 launch ur3e_moveit_config assistive_lift_v4_hri_sim.launch.py
```

Same as (3) plus a walking human actor in the world and the full
human-exclusion pipeline (`robot_self_filter` → `human_excluded_cloud_filter`
→ `octomap_gate`). Closest to (2) without hardware.

---

## Critical implementation notes

| Lesson | Workaround |
|---|---|
| ROS 2 Humble `cv_bridge` segfaults under NumPy 2.x and `opencv-python>=4.11` | Pin `numpy<2` and `opencv-python<4.11` |
| `ompl_planning.yaml` and `sensors_3d.yaml` are MoveIt-native YAML | Load with `yaml.safe_load()` and pass as dicts |
| MoveIt 2 Humble planning pipelines need nested config under pipeline name | Use `planning_pipelines: ["ompl"]` + `ompl.planning_plugin: ...` |
| UR driver loads `forward_velocity_controller` inactive | Activate explicitly in hardware launch |

---

## Debug & tuning tools

The repo ships with a set of GUIs and scripts used during development and
calibration.

### Debug nodes (in `my_thesis_controller`)

| Node                 | Purpose                                                    |
|----------------------|------------------------------------------------------------|
| `force_gui`          | tkinter GUI to inject test wrenches on `/wrench_external` (lets you simulate human pushes without touching the robot). |
| `hybrid_tuning_gui`  | Live tkinter sliders for runtime tuning of admittance M / D and damping params via `ros2 param set`. |
| `safety_monitor`     | ISO/TS 15066 compliance monitor: power & force limiting, speed & separation monitoring, hand-guiding speed cap. Publishes `/proximity/scale`, `/safety/status`. |

Run them stand-alone (after sourcing the workspace):

```
ros2 run my_thesis_controller force_gui
ros2 run my_thesis_controller hybrid_tuning_gui
ros2 run my_thesis_controller safety_monitor
```

### Calibration scripts (in `scripts/`)

| Script                            | Purpose                                                                                          |
|-----------------------------------|--------------------------------------------------------------------------------------------------|
| `scripts/calibrate_doubletap.py`  | Records 5 rounds of double-taps from `/wrench_zeroed`, prints recommended values for `nudge_threshold`, `z_score_thresh`, `nudge_cooldown_s`, and `DOUBLE_TAP_WINDOW_S`. |
| `scripts/test_doubletap.py`       | Stand-alone double-tap detector that subscribes to `/interaction/is_nudge` and prints tap count / gap timing for verifying the calibrated values. |

Run while a hardware launch is active:

```
python3 scripts/calibrate_doubletap.py     # 5 rounds, ~6 s each
python3 scripts/test_doubletap.py          # interactive verifier
```

The calibrated values currently baked into `assistive_lift_v4_node.py`
for the author's tap profile are: `nudge_threshold=4.0 N`,
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

---

## Credits

- **Author**: Nishanth Sundaran ([GitHub](https://github.com/NishanthSundaran), [sundharnishanth@gmail.com](mailto:sundharnishanth@gmail.com))
- **Supervisor**: Prof. Dr. Dmitrii Dobriborsci ([@dimadobriy](https://github.com/dimadobriy)), THD Research
- **Hardware**: THD Cham robotics lab

---

## License

Apache 2.0. See [LICENSE](LICENSE).

If you use this work for research, please cite:

```bibtex
@mastersthesis{sundaran2026hri,
  author       = {Nishanth Sundaran},
  title        = {Human-Robot Interaction with Force Feedback in Collaborative Tasks},
  school       = {Deggendorf Institute of Technology (THD)},
  year         = {2026}
}
```
