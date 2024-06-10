# Autonomous Underwater Docking using MPC

This package was developed to control a BlueROV2 to perform autonomous underwater docking. This package utilizes Model Predictive Control (MPC) to achieve optimal control of the vehicle.

## Getting Started

TODO: Add instructions here

## Usage

- Before proceeding further, make sure that all the steps mentioned in the guide for [Software Setup](https://bluerobotics.com/learn/bluerov2-software-setup
) has been followed.
- From a terminal, run `roslaunch bluerov2_dock mission_control.launch`
  - If the terminal initially outputs *Controller error:'joy'*, move the sticks to clear the error.
  - Press button "A" on the joystick to enable autonomous docking mode.
    - To switch back to manual mode, move either of the sticks in any direction.
