cmake_minimum_required(VERSION 3.0.2)
project(docking_control)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11) # -g for gdb
# set(CMAKE_BUILD_TYPE "Debug")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    mavros_msgs
    tf2_geometry_msgs
    nav_msgs
    tf2
    tf2_ros)

# find_package(octomap REQUIRED)
# find_package(ompl REQUIRED)
# find_package(fcl REQUIRED)
# find_package(Eigen3 REQUIRED)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES docking_control
 CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
${catkin_INCLUDE_DIRS}
)

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
  src/docking_control/mission_control.py
  src/docking_control/pwm_publish_node.py
  src/docking_control/video.py
  src/docking_control/manager.py
  src/docking_control/auto_dock.py
  src/docking_control/mpc_casadi.py
  src/docking_control/mpc_acados.py
  src/docking_control/auv_hinsdale.py
  src/docking_control/marker_detection.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


## Mark other files for installation (e.g. launch and bag files, etc.)
install(FILES
    launch/mission_control.launch
    launch/mav_pluginlists.yaml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
