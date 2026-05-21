#!/usr/bin/env bash
# Launch the thesis container with everything wired up:
#   --gpus all           : YOLOv8 on GPU + GL rendering for RViz/Gazebo
#   --network host       : reach the UR3e (e.g. 192.168.123.3) and ROS DDS
#   --privileged + /dev  : RealSense D435i USB passthrough
#   X11 socket           : RViz / Gazebo GUI on the host display
#
# Usage:
#   ./docker/run.sh                  # interactive shell in the container
#   ./docker/run.sh ros2 launch ur3e_moveit_config assistive_lift_v4_hri_sim.launch.py
#
# Requires: Docker + NVIDIA Container Toolkit on the host.

set -e

IMAGE="${THESIS_IMAGE:-thesis:latest}"

# Allow the container's X clients to talk to the host X server.
xhost +local:docker >/dev/null 2>&1 || true

docker run -it --rm \
    --name thesis \
    --gpus all \
    --network host \
    --ipc host \
    --privileged \
    -e "DISPLAY=${DISPLAY}" \
    -e QT_X11_NO_MITSHM=1 \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e XDG_RUNTIME_DIR=/tmp/runtime-root \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /dev:/dev \
    -v "${HOME}/.ignition:/root/.ignition" \
    "${IMAGE}" \
    "$@"

# Revoke the X grant when the container exits (best-effort).
xhost -local:docker >/dev/null 2>&1 || true
