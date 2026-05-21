# Human-Robot Interaction thesis — full sim + hardware environment.
#
# Base: CUDA 12.2 + cuDNN runtime on Ubuntu 22.04 so YOLOv8 runs on GPU.
# NVIDIA_DRIVER_CAPABILITIES=all also exposes the OpenGL stack, so RViz and
# Gazebo Ignition render with GPU acceleration when run with `--gpus all`.
#
# The whole workspace is built INTO the image (colcon build during docker
# build), so the image is self-contained and reproducible.
#
# Build:  docker build -f docker/Dockerfile -t thesis:latest .
# Run:    see docker/run.sh  (handles X11 + GPU + RealSense USB + UR network)

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
# Expose compute + graphics (OpenGL) to the container for RViz / Gazebo.
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

# ---- Locale ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        locales curl gnupg2 lsb-release software-properties-common ca-certificates \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# ---- ROS 2 apt repository ----
RUN add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
         -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
         > /etc/apt/sources.list.d/ros2.list

# ---- ROS 2 Humble desktop + thesis runtime dependencies ----
# (Mirrors the apt list in the top-level README. ros-humble-desktop pulls
#  rviz2, and ros-gz provides the Gazebo Ignition integration.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-desktop \
        ros-humble-moveit \
        ros-humble-moveit-servo \
        ros-humble-moveit-ros-move-group \
        ros-humble-controller-manager \
        ros-humble-ros2-control \
        ros-humble-ros2-controllers \
        ros-humble-joint-state-broadcaster \
        ros-humble-realsense2-camera \
        ros-humble-realsense2-description \
        ros-humble-tf2-ros \
        ros-humble-tf2-geometry-msgs \
        ros-humble-cv-bridge \
        ros-humble-sensor-msgs-py \
        ros-humble-ros-gz \
        ros-humble-ros-gz-bridge \
        ros-humble-ros-gz-sim \
        ignition-tools \
        ros-humble-pcl-ros \
        ros-humble-filters \
        python3-colcon-common-extensions \
        python3-vcstool \
        python3-rosdep \
        python3-pip \
        git \
    && rm -rf /var/lib/apt/lists/*

# ---- Python packages ----
# numpy<2 and opencv-python<4.11: cv_bridge segfaults on Humble otherwise.
# ultralytics pulls a CUDA-enabled torch wheel (matches the cuda base image).
RUN pip3 install --no-cache-dir "numpy<2" "opencv-python<4.11" scipy ultralytics

# ---- rosdep ----
RUN rosdep init && rosdep update

# ---- Build the thesis workspace ----
WORKDIR /thesis_ws
COPY . /thesis_ws/

# Pull source dependencies (UR driver) then resolve any remaining rosdeps.
RUN vcs import src < dependencies.repos
RUN apt-get update \
    && rosdep install --from-paths src --ignore-src -r -y \
    && rm -rf /var/lib/apt/lists/*

# colcon build (source ROS first).
RUN . /opt/ros/humble/setup.sh \
    && colcon build --symlink-install

# Pre-fetch the YOLOv8-seg weights so the first run is offline-friendly.
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')" || true

# ---- Entrypoint sources ROS + the workspace ----
RUN chmod +x /thesis_ws/docker/entrypoint.sh
ENTRYPOINT ["/thesis_ws/docker/entrypoint.sh"]
CMD ["bash"]
