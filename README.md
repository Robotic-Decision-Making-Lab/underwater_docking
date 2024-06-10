# docking_control

This package was developed to control a BlueROV2 to perform autonomous underwater docking. This package utilizes Model Predictive Control (MPC) to achieve optimal control of the vehicle.

This repository is a modified version of the [bluerov_ros_playground](https://github.com/patrickelectric/bluerov_ros_playground) repo authored by [patrickelectric](https://github.com/patrickelectric).

## Getting Started

### Requirements

- [Python](https://www.python.org/downloads/) 3.6 or newer
  - [Numpy](https://pypi.org/project/numpy/)
  - [OpenCV](https://pypi.org/project/opencv-python/)
  - [PyYAML](https://pypi.org/project/PyYAML/)
  - [Gi & Gobject](https://wiki.ubuntu.com/Novacut/GStreamer1.0)
  - [CasADi](https://pypi.org/project/casadi/)
  - [acados](https://docs.acados.org/index.html)
  - [Pandas](https://pypi.org/project/pandas/)
- [ROS](http://wiki.ros.org/ROS/Installation)
  - kinetic or newer
- Geographlib
- [imutils](https://github.com/PyImageSearch/imutils)
- [MAVROS](http://wiki.ros.org/mavros)

### Installation

#### Installing ROS: *Ubuntu 20.04* (See [ROS wiki](http://wiki.ros.org/ROS/Installation) for other OS)

1. Set up sources.list
   - `sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'`
2. Set up keys
   - `sudo apt install curl`
   - `curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -`
3. Installation
   - `sudo apt update`
   - Pick how much ROS you want to install: must be either Desktop or Desktop-Full
      - ROS Desktop (includes rqt and rviz)
         - `sudo apt install ros-noetic-desktop`
      - Desktop-Full (includes Desktop and 2D/3D simulators)
         - `sudo apt install ros-noetic-desktop-full`
4. Set up environment
   - You must source this script in every bash terminal you use ROS in
      - `source /opt/ros/noetic/setup.bash`
   - To automatically source this script every time a new shell is launched, run
      - `echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc`
      - `source ~/.bashrc`
   - Check that *source /opt/ros/noetic/setup.bash* has been added to the end of .bashrc
5. Install dependencies for building packages
   - `sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential`
6. Initialize rosdep
   - `sudo apt install python3-rosdep`
   - `sudo rosdep init`
   - `rosdep update`

#### Building a Catkin Workspace

1. Create and build workspace
   - `mkdir -p ~/catkin_ws/src`
   - `cd ~/catkin_ws/`
   - `catkin_make`
   - This will create a workspace with 'build', 'devel', and 'src' folders and a *CMakeLists.txt* in 'src'
2. Source
   - `source devel/setup.bash`
   - To automatically source this script every time a new shell is launched, run
      - `echo "source /home/<username>/catkin_ws/devel/setup.bash" >> ~/.bashrc`
      - `source ~/.bashrc`
      - Check that *source /home/< username>/catkin_ws/devel/setup.bash* has been added to the end of .bashrc
3. Make sure the ROS_PACKAGE_PATH includes the directory
   - `echo $ROS_PACKAGE_PATH`
   /home/youruser/catkin_ws/src:/opt/ros/noetic/share

#### Installing CasADi

`pip install casadi`

#### Installing acados

Please follow the instructions provided in the following two links:

- [Source Installation](https://docs.acados.org/installation/index.html)
- [Python Interface](https://docs.acados.org/python_interface/)

#### Installing MAVROS

`sudo apt-get install ros-noetic-mavros ros-noetic-mavros-extras`

#### Installing Geographlib

`sudo /opt/ros/noetic/lib/mavros/install_geographiclib_datasets.sh`

#### Installing imutils

`pip install imutils`

#### Installing Pandas

`pip install pandas`

#### Installing OpenCV

1. Make sure python-opencv is not installed
   - `pip uninstall opencv-python`
2. Install opencv-contrib-python
   - `pip install opencv-contrib-python`

#### Cloning Project

 1. Go to your ROS package source directory:
    - `cd catkin_ws/src`
 2. Clone this project
    - `git clone https://github.com/rakeshv24/bluerov2_dock`
 3. Go back to your workspace:
    - `cd ../`
 4. Build and install the project:
    - `catkin_make --pkg bluerov2_dock`
 5. Reload your ROS env.
    - `source devel/setup.sh`

## Usage

- Before proceeding further, make sure that all the steps mentioned in the guide for [Software Setup](https://bluerobotics.com/learn/bluerov2-software-setup
) has been followed.
- From a terminal, run `roslaunch bluerov2_dock mission_control.launch`
  - If the terminal initially outputs *Controller error:'joy'*, move the sticks to clear the error.
  - Press button "A" on the joystick to enable autonomous docking mode.
    - To switch back to manual mode, move either of the sticks in any direction.
