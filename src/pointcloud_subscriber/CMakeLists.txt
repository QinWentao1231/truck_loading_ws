cmake_minimum_required(VERSION 3.8)
project(pointcloud_subscriber)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(ament_index_cpp REQUIRED)

# 包含头文件目录
include_directories(include)

# 添加可执行文件
add_executable(pointcloud_save_node src/pointcloud_save_node.cpp)

# 链接依赖项
ament_target_dependencies(pointcloud_save_node
  rclcpp
  sensor_msgs
  pcl_conversions
  pcl_ros
  ament_index_cpp
)

# 安装目标
install(TARGETS
  pointcloud_save_node
  DESTINATION lib/${PROJECT_NAME}
)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
ament_package()
