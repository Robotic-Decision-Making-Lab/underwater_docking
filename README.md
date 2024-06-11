# Autonomous Underwater Docking using MPC

This package was developed to control a BlueROV2 to perform autonomous underwater docking. This package utilizes Model Predictive Control (MPC) to achieve optimal control of the vehicle.

## Getting Started

### Docker Installation

- Install Docker on your system by following the step-by-step guide provided [here](https://docs.docker.com/get-docker/).
- To leverage the *-desktop-nvidia image and utilize NVIDIA GPU support, follow this installation guide to set up the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- Use the guide provided [here](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry) to connect to the GitHub Container Registry.

### Development Containers

- To use Visual Studio Code within [development containers](https://github.com/Robotic-Decision-Making-Lab/underwater_docking/tree/main/.devcontainer), follow the tutorial provided [here](https://code.visualstudio.com/docs/devcontainers/tutorial).
- Next, clone and open [this](https://github.com/Robotic-Decision-Making-Lab/underwater_docking) repository in VSCode.
- Choose the option to reopen it inside a container when prompted.
- Once the container is built, you can begin your development.

## Usage

- Before proceeding further, make sure that all the steps mentioned in the guide for [Software Setup](https://bluerobotics.com/learn/bluerov2-software-setup
) has been followed.
- From a terminal, run `roslaunch bluerov2_dock mission_control.launch`
  - If the terminal initially outputs *Controller error:'joy'*, move the sticks to clear the error.
  - Press button "A" on the joystick to enable autonomous docking mode.
    - To switch back to manual mode, move either of the sticks in any direction.
