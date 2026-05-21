#!/usr/bin/env bash
# Source ROS 2 and the built thesis workspace, then exec the command.
set -e

source /opt/ros/humble/setup.bash
if [ -f /thesis_ws/install/setup.bash ]; then
    source /thesis_ws/install/setup.bash
fi

# Gazebo Fuel cache (walking actor mesh) lives here; persists if mounted.
export GZ_FUEL_CACHE_PATH="${GZ_FUEL_CACHE_PATH:-/root/.gz/fuel}"

exec "$@"
