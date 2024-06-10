import os
import time
import numpy as np
import sys
import rospy

# sys.path.insert(0, '/home/darth/workspace/bluerov2_ws/src/bluerov2_dock/src/bluerov2_dock')

from auv_hinsdale import AUV

# from mpc_casadi import MPC
from mpc_acados import MPC


class MPControl:
    def __init__(self):
        """
        This class is used to control the AUV using the MPC controller.
        """
        cwd = os.path.dirname(__file__)

        # auv_yaml = cwd + "/../../config/auv_bluerov2.yaml"
        # mpc_yaml = cwd + "/../../config/mpc_bluerov2.yaml"

        auv_yaml = cwd + "/../../config/auv_bluerov2_heavy.yaml"
        mpc_yaml = cwd + "/../../config/mpc_bluerov2_heavy.yaml"

        # Change these values as desired
        self.tolerance = 0.05
        self.yaw_tolerance = 0.05
        self.path_length = 0.0
        self.p_times = [0]

        self.auv = AUV.load_params(auv_yaml)
        self.mpc = MPC.load_params(auv_yaml, mpc_yaml)

        self.thrusters = self.mpc.thrusters
        self.comp_time = 0.0
        self.time_id = 0
        self.dt = self.mpc.dt
        self.t_f = 3600.0
        self.t_span = np.arange(0.0, self.t_f, self.dt)
        self.mpc.reset()

        # Wave Parameters
        self.wave_height = 0.2
        self.wave_T = 2.5
        self.water_depth = 1.136
        self.g = 9.81
        self.wave_number = self.compute_wave_number(
            self.g, self.water_depth, self.wave_T
        )
        self.wave_freq = 2 * np.pi / self.wave_T
        self.wave_speed = self.wave_freq / self.wave_number
        self.wave_lambda = 2 * np.pi / self.wave_number
        self.wave_phase = 0
        self.wavemaker_offset = [16.763, 0.0, 0.568]

        self.opt_data = {}

    def compute_wave_number(self, g, h, T):
        """This function computes the wave number for a given wave period and water depth.

        Args:
            g: aceeleration due to gravity
            h: wave height
            T: wave period

        Returns:
            k: wave number
        """
        k = 1  # Initial guess for wave number
        iter_tol = 0.0000001  # Iteration tolerance
        err = 1

        while err > iter_tol:
            res = (2 * np.pi / T) ** 2 - g * k * np.tanh(k * h)
            dev_res = -g * np.tanh(k * h) - g * k * h * (1 - (np.tanh(k * h)) ** 2)
            k_new = k - res / dev_res
            err = abs(k_new - k) / k
            k = k_new

        return k

    def wrap_pi2negpi(self, angle):
        """This function wraps the angle to the range -pi to pi.

        Args:
            angle: angle to be wrapped

        Returns:
            angle: wrapped angle
        """
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

    def wrap_zeroto2pi(self, angle):
        """This function wraps the angle to the range 0 to 2pi.

        Args:
            angle: angle to be wrapped

        Returns:
            angle: wrapped angle
        """
        return angle % (2 * np.pi)

    def compute_wave_particle_vel(self, eta, t):
        """Computes the wave particle velocity at a given point in the wave field.

        Args:
            eta: vehicle pose 6 dof
            t: time

        Returns:
            nu_w: wave particle velocity
        """
        x = -eta[0][0] + self.wavemaker_offset[0]
        z = -eta[2][0] - self.wavemaker_offset[2]
        H = self.wave_height
        k = self.wave_number
        phase = self.wave_phase
        c = self.wave_speed
        T = self.wave_T
        g = self.g
        d = self.water_depth
        omega = self.wave_freq
        L = self.wave_lambda
        # L = (g * T**2 / 2 * np.pi) * ((np.tanh(omega**2 * d / g)**(3/4)))**(2/3)

        u = (g * H * np.cosh(k * (z + d)) * np.cos(k * x - omega * t + phase)) / (
            2 * c * np.cosh(k * d)
        )
        w = (g * H * np.sinh(k * (z + d)) * np.sin(k * x - omega * t + phase)) / (
            2 * c * np.cosh(k * d)
        )
        u_dot = (
            g * H * np.pi * np.cosh(k * (z + d)) * np.sin(k * x - omega * t + phase)
        ) / (L * np.cosh(k * d))
        w_dot = -(
            g * H * np.pi * np.sinh(k * (z + d)) * np.cos(k * x - omega * t + phase)
        ) / (L * np.cosh(k * d))

        nu_w = np.zeros((6, 1))
        nu_w[0][0] = -u
        nu_w[2][0] = -w
        nu_w[3][0] = -u_dot
        nu_w[5][0] = -w_dot
        return nu_w

    def run_mpc(self, x0, xr):
        """This function runs the MPC controller.

        Args:
            x0: Initial vehicle pose
            xr: Desired vehicle pose

        Returns:
            u: Control input
            wrench: Thruster forces
            done: True if the vehicle has reached the desired pose
        """
        process_t0 = time.perf_counter()
        self.distance = np.linalg.norm(x0[0:3, :] - xr[0:3, :])

        x0[3:6, :] = self.wrap_pi2negpi(x0[3:6, :])
        # xr[5, :] += np.pi
        xr[3:6, :] = self.wrap_pi2negpi(xr[3:6, :])

        self.yaw_diff = abs((((x0[5, :] - xr[5, :]) + np.pi) % (2 * np.pi)) - np.pi)[0]

        if self.distance < self.tolerance and self.yaw_diff < self.yaw_tolerance:
            return np.zeros((8, 1)), np.zeros((6, 1)), True

        else:
            nu_w = self.compute_wave_particle_vel(x0[0:6, :], self.t_span[self.time_id])
            rospy.logwarn(f"Wave Info: {nu_w}")
            self.auv.nu_w = nu_w
            u, wrench = self.mpc.run_mpc(x0, xr)

            self.comp_time = time.perf_counter() - process_t0

            print(
                f"T = {round(self.t_span[self.time_id],3)}s, Time Index = {self.time_id}"
            )
            print(f"Computation Time = {round(self.comp_time,3)}s")
            print("----------------------------------------------")
            print(f"MPC Contol Input: {np.round(u, 2).T}")
            print("----------------------------------------------")
            print(f"Axes Forces: {np.round(wrench, 2).T}")
            print("----------------------------------------------")
            print(f"Initial Vehicle Pose: {np.round(x0[0:6], 3).T}")
            print(f"Initial Vehicle Velocity: {np.round(x0[6:12], 3).T}")
            # print("----------------------------------------------")
            # print(f"Sim Vehicle Pose: {np.round(x_sim[0:6], 3).T}")
            # print(f"Sim Vehicle Velocity: {np.round(x_sim[6:12], 3).T}")
            print("----------------------------------------------")
            print(f"Dock Pose: {np.round(xr[0:6], 3).T}")
            print(f"Dock Velocity: {np.round(xr[6:12], 3).T}")
            print("----------------------------------------------")
            print(f"Path length: {np.round(self.path_length, 3)}")
            print(f"(Dock-AUV) Distance to go: {np.round(self.distance, 3)}")
            print(f"(Dock-AUV) Yaw difference: {np.round(self.yaw_diff, 3)}")
            print("----------------------------------------------")
            # print("")

            self.time_id += 1

            return u, wrench, False


if __name__ == "__main__":
    mpc = MPControl()
