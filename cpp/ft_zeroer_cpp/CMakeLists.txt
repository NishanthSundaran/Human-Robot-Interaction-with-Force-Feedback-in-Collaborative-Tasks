cmake_minimum_required(VERSION 3.10)
project(ft_zeroer_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)

add_executable(ft_zeroer_node src/ft_zeroer_node.cpp)
ament_target_dependencies(ft_zeroer_node
  rclcpp
  geometry_msgs
  std_srvs
)

install(TARGETS ft_zeroer_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
