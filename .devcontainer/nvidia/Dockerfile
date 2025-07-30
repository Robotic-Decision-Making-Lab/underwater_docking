FROM ghcr.io/robotic-decision-making-lab/underwater_docking:noetic-desktop-nvidia

# Install ROS dependencies
# This is done in a previous stage, but we include it again here in case anyone wants to
# add new dependencies during development
ENV USERNAME=ros
ENV USER_WORKSPACE=/home/$USERNAME/ws_dock
WORKDIR $USER_WORKSPACE

COPY --chown=$USER_UID:$USER_GID . src/$PROJECT_NAME

RUN ls src/$PROJECT_NAME

RUN sudo rm /etc/apt/sources.list.d/ros1-latest.list \
  && sudo rm /usr/share/keyrings/ros1-latest-archive-keyring.gpg

RUN sudo apt-get update \
  && sudo apt-get install -y ca-certificates curl

RUN sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg


RUN sudo apt-get -q update \
    && sudo apt-get -q -y upgrade \
    && rosdep update \
    && rosdep install -y --from-paths . --ignore-src -r --rosdistro ${ROS_DISTRO} \
    && sudo apt-get autoremove -y \
    && sudo apt-get clean -y \
    && sudo rm -rf /var/lib/apt/lists/*

# Install debugging/linting Python packages
COPY --chown=$USER_UID:$USER_GID requirements-dev.txt .
RUN python3 -m pip install -r requirements-dev.txt \
    && rm -rf requirements-dev.txt

# Disable the setuputils installation warning
# This prevents us from needing to pin the setuputils version (which doesn't always work)
ENV PYTHONWARNINGS="ignore"
