#!/usr/bin/env bash
# Source ROS 2 and the built thesis workspace, then exec the command.
set -e

source /opt/ros/humble/setup.bash
if [ -f /thesis_ws/install/setup.bash ]; then
    source /thesis_ws/install/setup.bash
fi

# Gazebo Fortress caches Fuel assets (the walking-actor mesh) under
# ~/.ignition/fuel. Mount /root/.ignition (see run.sh) to persist it.
export IGN_FUEL_CACHE_PATH="${IGN_FUEL_CACHE_PATH:-/root/.ignition/fuel}"

# Some Qt/GL apps need XDG_RUNTIME_DIR to exist with sane perms.
if [ -n "${XDG_RUNTIME_DIR}" ] && [ ! -d "${XDG_RUNTIME_DIR}" ]; then
    mkdir -p "${XDG_RUNTIME_DIR}" && chmod 700 "${XDG_RUNTIME_DIR}"
fi

exec "$@"
