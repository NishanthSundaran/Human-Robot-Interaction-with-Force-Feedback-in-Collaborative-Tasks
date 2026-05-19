cmake_minimum_required(VERSION 3.8)
project(ur3e_moveit_config)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

install(
  DIRECTORY config launch srdf worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
