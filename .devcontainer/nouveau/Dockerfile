FROM ghcr.io/robotic-decision-making-lab/underwater_docking:noetic-desktop

# Install ROS dependencies
# This is done in a previous stage, but we include it again here in case anyone wants to
# add new dependencies during development
ENV USERNAME=ros
ENV USER_WORKSPACE=/home/$USERNAME/ws_dock
# Define UID and GID for chown operations
ENV USER_UID=1000
ENV USER_GID=1000
WORKDIR $USER_WORKSPACE

COPY --chown=$USER_UID:$USER_GID . src/$PROJECT_NAME

RUN ls src/$PROJECT_NAME

RUN sudo rm -f /etc/apt/sources.list.d/ros1-latest.list \
  && sudo rm -f /usr/share/keyrings/ros1-latest-archive-keyring.gpg

RUN sudo apt-get update \
  && sudo apt-get install -y ca-certificates curl libc++-dev \
  && pip install -U importlib_metadata

RUN sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN sudo apt-get -q update \
    && sudo apt-get -q -y upgrade \
    && rosdep update --include-eol-distros \
    && rosdep install -y --from-paths . --ignore-src -r --rosdistro ${ROS_DISTRO} \
    && sudo apt-get install -y ros-noetic-ompl ros-noetic-ompl-dbgsym \
    && sudo apt-get autoremove -y \
    && sudo apt-get clean -y \
    && sudo rm -rf /var/lib/apt/lists/*

RUN cd /tmp && wget https://github.com/isl-org/Open3D/releases/download/v0.19.0/open3d-devel-linux-x86_64-cxx11-abi-0.19.0.tar.xz \
    && sudo tar -xf open3d-devel-linux-x86_64-cxx11-abi-0.19.0.tar.xz \
    && sudo cp -r open3d-devel-linux-x86_64-cxx11-abi-0.19.0/include/* /usr/include \
    && sudo cp -r open3d-devel-linux-x86_64-cxx11-abi-0.19.0/lib/* /usr/lib \
    && sudo cp -r open3d-devel-linux-x86_64-cxx11-abi-0.19.0/share/* /usr/share

# Install debugging/linting Python packages
COPY --chown=$USER_UID:$USER_GID requirements-dev.txt .
RUN python3 -m pip install -r requirements-dev.txt \
    && rm -rf requirements-dev.txt
