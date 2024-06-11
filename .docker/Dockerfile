ARG ROS_DISTRO=noetic
FROM ros:$ROS_DISTRO-ros-base as ci

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /root/ws_dock

# Update to be your project's name
ENV PROJECT_NAME=underwater_docking

COPY . src/$PROJECT_NAME

# Install apt packages
RUN apt-get -q update \
    && apt-get -q -y upgrade \
    && apt-get -q install --no-install-recommends -y \
    git \
    sudo \
    clang \
    clang-format-12 \
    clang-tidy \
    clang-tools \
    python3-pip \
    python3-dev \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    python3-catkin-tools \
    build-essential \
    nano \
    vim \
    apt-utils \
    wget \
    software-properties-common \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install all ROS dependencies needed for CI
RUN apt-get -q update \
    && apt-get -q -y upgrade \
    && rosdep update \
    && rosdep install -y --from-paths src --ignore-src --rosdistro ${ROS_DISTRO} --as-root=apt:false \
    && rm -rf src \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

FROM ci as robot

# Configure a new non-root user
ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && usermod -a -G dialout $USERNAME \
    && echo "source /usr/share/bash-completion/completions/git" >> /home/$USERNAME/.bashrc

# Switch to the non-root user
USER $USERNAME
ENV USER=$USERNAME

ENV DEBIAN_FRONTEND=noninteractive

# Install MAVROS dependencies
WORKDIR /home/$USERNAME
RUN wget https://raw.githubusercontent.com/mavlink/mavros/ros2/mavros/scripts/install_geographiclib_datasets.sh \
    && chmod +x install_geographiclib_datasets.sh \
    && sudo ./install_geographiclib_datasets.sh

# Install acados
WORKDIR /home/$USERNAME
RUN git clone https://github.com/acados/acados.git \
    && cd acados \
    && git submodule update --recursive --init \
    && mkdir -p build \
    && cd build \
    && cmake -DACADOS_WITH_QPOASES=ON -DACADOS_WITH_OSQP=ON ..\
    && make install -j4

# Install acados python interface
WORKDIR /home/$USERNAME
RUN pip install -e /home/$USERNAME/acados/interfaces/acados_template \
    && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/home/$USERNAME/acados/lib" \
    && export ACADOS_SOURCE_DIR="/home/$USERNAME/acados"


ENV USER_WORKSPACE=/home/$USERNAME/ws_dock
WORKDIR $USER_WORKSPACE
COPY --chown=$USER_UID:$USER_GID . src/$PROJECT_NAME

# Uncomment to download and install external repos used by the project
# RUN vcs import src < src/$PROJECT_NAME/ros.repos

# Install gstreamer
RUN sudo apt-get -q update \
    && sudo apt-get -q -y upgrade \
    && sudo apt-get -q install --no-install-recommends -y \
    python3-gi \
    gstreamer1.0-tools \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    && sudo apt-get autoremove -y \
    && sudo apt-get clean -y \
    && sudo rm -rf /var/lib/apt/lists/*

# Install all ROS dependencies
WORKDIR $USER_WORKSPACE
RUN sudo apt-get -q update \
    && sudo apt-get -q -y upgrade \
    && rosdep update \
    && rosdep install -y --from-paths src --ignore-src -r --rosdistro ${ROS_DISTRO} \
    && sudo apt-get autoremove -y \
    && sudo apt-get clean -y \
    && sudo rm -rf /var/lib/apt/lists/*

RUN . "/opt/ros/${ROS_DISTRO}/setup.sh" \
    && catkin_make \
    && echo "source ${USER_WORKSPACE}/devel/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "if [ -f /opt/ros/${ROS_DISTRO}/setup.bash ]; then source /opt/ros/${ROS_DISTRO}/setup.bash; fi" >> /home/$USERNAME/.bashrc

FROM robot as desktop

ENV DEBIAN_FRONTEND=noninteractive
ENV USER_WORKSPACE=/home/$USERNAME/ws_dock
WORKDIR $USER_WORKSPACE

# Install Gazebo and RViz
RUN sudo apt-get update \
    && sudo apt-get -q -y upgrade \
    && curl -sSL http://get.gazebosim.org | sh \
    && sudo apt-get -q install --no-install-recommends -y \
    ros-$ROS_DISTRO-gazebo* \
    ros-$ROS_DISTRO-rviz \
    && sudo apt-get autoremove -y \
    && sudo apt-get clean -y \
    && sudo rm -rf /var/lib/apt/lists/*

FROM desktop as desktop-nvidia

# Install NVIDIA software
RUN sudo apt-get update \
    && sudo apt-get -q -y upgrade \
    && sudo apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    && sudo apt-get autoremove -y \
    && sudo apt-get clean -y \
    && sudo rm -rf /var/lib/apt/lists/*

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
ENV QT_X11_NO_MITSHM 1
