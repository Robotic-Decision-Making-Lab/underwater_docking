import pdb

import numpy as np
import yaml
from casadi import (
    SX,
    cos,
    diag,
    fabs,
    if_else,
    inv,
    mtimes,
    sin,
    skew,
    tan,
    vertcat,
)


class AUV(object):
    def __init__(self, vehicle_dynamics):
        """This class is used to define the dynamics of the AUV.

        Args:
            vehicle_dynamics: A dictionary containing the vehicle dynamics parameters.
        """
        self.vehicle_length = vehicle_dynamics["vehicle_length"]
        self.vehicle_mass = vehicle_dynamics["vehicle_mass"]
        self.rb_mass = vehicle_dynamics["rb_mass"]
        self.added_mass = vehicle_dynamics["added_mass"]
        self.lin_damp = vehicle_dynamics["lin_damp"]
        self.quad_damp = vehicle_dynamics["quad_damp"]
        self.tcm = vehicle_dynamics["tcm"]
        self.inertial_terms = vehicle_dynamics["inertial_terms"]
        self.inertial_skew = vehicle_dynamics["inertial_skew"]
        self.r_gb_skew = vehicle_dynamics["r_gb_skew"]
        self.W = vehicle_dynamics["W"]
        self.B = vehicle_dynamics["B"]
        self.cog = vehicle_dynamics["cog"]
        self.cob = vehicle_dynamics["cob"]
        self.cog_to_cob = self.cog - self.cob
        self.neutral_bouy = vehicle_dynamics["neutral_bouy"]
        self.curr_timestep = 0.0
        self.ocean_current_data = []
        self.nu_w = np.zeros((6, 1))

        # Precompute the total mass matrix (w/added mass) inverted for future dynamic calls
        self.mass_inv = inv(self.rb_mass + self.added_mass)
        self.rb_mass_inv = inv(self.rb_mass)
        self.added_mass_inv = inv(self.added_mass)

    @classmethod
    def load_params(cls, filename):
        f = open(filename, "r")
        params = yaml.load(f.read(), Loader=yaml.SafeLoader)

        l = params["l"]
        m = params["m"]
        Ixx = params["Ixx"]
        Iyy = params["Iyy"]
        Izz = params["Izz"]
        I_b = SX.eye(3) * [Ixx, Iyy, Izz]

        skew_r_gb = skew(SX(params["cog"]))
        skew_I = skew([Ixx, Iyy, Izz])

        rb_mass = diag(SX([m, m, m, Ixx, Iyy, Izz]))
        rb_mass[0:3, 3:6] = -m * skew_r_gb
        rb_mass[3:6, 0:3] = m * skew_r_gb

        vehicle_dynamics = {}
        vehicle_dynamics["vehicle_length"] = l
        vehicle_dynamics["vehicle_mass"] = m
        vehicle_dynamics["rb_mass"] = rb_mass
        vehicle_dynamics["added_mass"] = -diag(SX(params["m_added"]))
        vehicle_dynamics["lin_damp"] = -diag(SX(params["d_lin"]))
        vehicle_dynamics["quad_damp"] = -diag(SX(params["d_quad"]))
        vehicle_dynamics["tcm"] = SX(params["tcm"])
        vehicle_dynamics["r_gb_skew"] = skew_r_gb
        vehicle_dynamics["inertial_skew"] = skew_I
        vehicle_dynamics["inertial_terms"] = I_b
        vehicle_dynamics["W"] = params["W"]
        vehicle_dynamics["B"] = params["B"]
        vehicle_dynamics["cog"] = SX(params["cog"])
        vehicle_dynamics["cob"] = SX(params["cob"])
        vehicle_dynamics["neutral_bouy"] = params["neutral_bouy"]

        return cls(vehicle_dynamics)

    def compute_transformation_matrix(self, x):
        """This function computes the transformation matrix from Body to NED frame
        based on Fossen's book.

        Args:
            x: vehicle pose 6 dof

        Returns:
            tf_mtx: transformation matrix
        """
        rot_mtx = SX.eye(3)
        rot_mtx[0, 0] = cos(x[5]) * cos(x[4])
        rot_mtx[0, 1] = -sin(x[5]) * cos(x[3]) + cos(x[5]) * sin(x[4]) * sin(x[3])
        rot_mtx[0, 2] = sin(x[5]) * sin(x[3]) + cos(x[5]) * cos(x[3]) * sin(x[4])
        rot_mtx[1, 0] = sin(x[5]) * cos(x[4])
        rot_mtx[1, 1] = cos(x[5]) * cos(x[3]) + sin(x[3]) * sin(x[4]) * sin(x[5])
        rot_mtx[1, 2] = -cos(x[5]) * sin(x[3]) + sin(x[4]) * sin(x[5]) * cos(x[3])
        rot_mtx[2, 0] = -sin(x[4])
        rot_mtx[2, 1] = cos(x[4]) * sin(x[3])
        rot_mtx[2, 2] = cos(x[4]) * cos(x[3])

        t_mtx = SX.eye(3)
        t_mtx[0, 1] = sin(x[3]) * tan(x[4])
        t_mtx[0, 2] = cos(x[3]) * tan(x[4])
        t_mtx[1, 1] = cos(x[3])
        t_mtx[1, 2] = -sin(x[3])
        t_mtx[2, 1] = sin(x[3]) / cos(x[4])
        t_mtx[2, 2] = cos(x[3]) / cos(x[4])

        tf_mtx = SX.eye(6)
        tf_mtx[0:3, 0:3] = rot_mtx
        tf_mtx[3:, 3:] = t_mtx

        return tf_mtx

    def compute_C_RB_force(self, v):
        """This function computes the Coriolis force due to rigid body motion
        based on Fossen's book.

        Args:
            v: vehicle velocity

        Returns:
            coriolis_force: Coriolis force matrix
        """
        v1 = v[0:3, 0]
        v2 = v[3:6, 0]

        skew_v1 = skew(v1)
        skew_v2 = skew(v2)
        skew_I_v2 = skew(mtimes(self.inertial_terms, v2))

        coriolis_force = SX.zeros((6, 6))
        coriolis_force[0:3, 0:3] = self.vehicle_mass * skew_v2
        coriolis_force[0:3, 3:6] = self.vehicle_mass * (
            -skew_v1 - mtimes(skew_v2, self.r_gb_skew)
        )
        coriolis_force[3:6, 0:3] = self.vehicle_mass * (
            -skew_v1 + mtimes(self.r_gb_skew, skew_v2)
        )
        coriolis_force[3:6, 3:6] = -skew_I_v2
        return coriolis_force

    def compute_C_A_force(self, v):
        """This function computes the Coriolis force due to added mass
        based on Fossen's book.

        Args:
            v: vehicle velocity

        Returns:
            coriolis_force: Coriolis force matrix
        """
        v1 = v[0:3, 0]
        v2 = v[3:6, 0]

        A_11 = self.added_mass[0:3, 0:3]
        A_12 = self.added_mass[0:3, 3:6]
        A_21 = self.added_mass[3:6, 0:3]
        A_22 = self.added_mass[3:6, 3:6]

        coriolis_force = SX.zeros((6, 6))
        coriolis_force[0:3, 3:6] = -skew(mtimes(A_11, v1) + mtimes(A_12, v2))
        coriolis_force[3:6, 0:3] = -skew(mtimes(A_11, v1) + mtimes(A_12, v2))
        coriolis_force[3:6, 3:6] = -skew(mtimes(A_21, v1) + mtimes(A_22, v2))
        return coriolis_force

    def compute_damping_force(self, v):
        """This function computes the damping force based on Fossen's book.

        Args:
            v: vehicle velocity

        Returns:
            damping_force: damping force matrix
        """
        damping_force = (self.quad_damp * fabs(v)) + self.lin_damp
        return damping_force

    def compute_restorive_force(self, x):
        """This function computes the restorive force based on Fossen's book.

        Args:
            x: vehicle pose 6 dof

        Returns:
            restorive_force: restorive force matrix
        """
        if self.neutral_bouy:
            restorive_force = (self.cog_to_cob * self.W) * vertcat(
                0.0, 0.0, 0.0, cos(x[4]) * sin(x[3]), sin(x[4]), 0.0
            )
        else:
            restorive_force = vertcat(
                (self.W - self.B) * sin(x[4]),
                -(self.W - self.B) * cos(x[4]) * sin(x[3]),
                -(self.W - self.B) * cos(x[4]) * cos(x[3]),
                -(self.cog[1] * self.W - self.cob[1] * self.B) * cos(x[4]) * cos(x[3])
                + (self.cog[2] * self.W - self.cob[2] * self.B) * cos(x[4]) * sin(x[3]),
                (self.cog[2] * self.W - self.cob[2] * self.B) * sin(x[4])
                + (self.cog[0] * self.W - self.cob[0] * self.B) * cos(x[4]) * cos(x[3]),
                -(self.cog[0] * self.W - self.cob[0] * self.B) * cos(x[4]) * sin(x[3])
                - (self.cog[1] * self.W - self.cob[1] * self.B) * sin(x[4]),
            )
        return restorive_force

    def compute_wave_force(self):
        """This function computes the wave force influencing the vehicle's motion.

        Returns:
            wave_force: wave force matrix
        """
        nu_w = self.nu_w[0:3, :]
        nu_w_dot = self.nu_w[3:6, :]
        wave_force = SX.zeros((6, 1))
        wave_force[0, 0] = (
            self.added_mass[0, 0] * nu_w_dot[0, 0]
            + (self.lin_damp[0, 0] + (self.quad_damp[0, 0] * fabs(nu_w[0, 0])))
            * nu_w[0, 0]
        )
        wave_force[2, 0] = (
            self.added_mass[2, 2] * nu_w_dot[2, 0]
            + (self.lin_damp[2, 2] + (self.quad_damp[2, 2] * fabs(nu_w[2, 0])))
            * nu_w[2, 0]
        )
        wave_force[4, 0] = 0
        return wave_force

    def compute_nonlinear_dynamics(
        self,
        x,
        u,
        f_B=SX.zeros((3, 1)),
        f_B_dot=SX.zeros((3, 1)),
        nu_w=SX.zeros((3, 1)),
        nu_w_dot=SX.zeros((3, 1)),
        f_est=False,
        complete_model=False,
    ):
        """This function computes the nonlinear dynamics of the AUV.

        Args:
            x: vehicle state 12 dof
            u: control input 6 dof
            f_B: flow velocity in body frame. Defaults to SX.zeros((3,1)).
            f_B_dot: flow acceleration in body frame. Defaults to SX.zeros((3,1)).
            nu_w: wave velocity. Defaults to SX.zeros((3,1)).
            nu_w_dot: wave acceleration. Defaults to SX.zeros((3,1)).
            f_est: flag for flow state estimation. Defaults to False.
            complete_model: vehicle model. Defaults to False.

        Returns:
            x_dot: vehicle state derivative
        """
        eta = x[0:6, :]
        nu_r = x[6:12, :]

        # Gets the transformation matrix to convert from Body to NED frame
        tf_mtx = self.compute_transformation_matrix(eta)
        tf_mtx_inv = inv(tf_mtx)

        # nu_c = SX.zeros(6,1)
        nu_c = vertcat(f_B, SX.zeros((3, 1)))
        # nu_c = vertcat(f, SX.zeros(3,1))
        # nu_c_dot = SX.zeros((6,1))

        # Converts ocean current disturbances to Body frame
        # nu_c = mtimes(tf_mtx_inv, nu_c_ned)

        # Computes total vehicle velocity
        nu = nu_r + nu_c

        # Computes the ocean current acceleration in Body frame
        skew_mtx = SX.eye(6)
        skew_mtx[0:3, 0:3] = -skew(nu_r[3:6])
        nu_c_dot = if_else(
            f_est, mtimes(skew_mtx, nu_c), vertcat(f_B_dot, SX.zeros((3, 1)))
        )
        # nu_c_dot = jacobian(nu_c, t)

        # Kinematic Equation
        # Convert the relative velocity from Body to NED and add it with the
        # ocean current velocity in NED to get the total velocity of the vehicle in NED

        # eta_dot = mtimes(tf_mtx, nu_r) + nu_c_ned
        eta_dot = if_else(
            complete_model, mtimes(tf_mtx, (nu_r + nu_c)), mtimes(tf_mtx, nu_r)
        )

        # eta_dot = mtimes(tf_mtx, (nu_r + nu_c))

        # Force computation
        # thruster_force = mtimes(self.tcm, u)
        thruster_force = u
        restorive_force = self.compute_restorive_force(eta)
        damping_force = self.compute_damping_force(nu_r)
        coriolis_force_rb = self.compute_C_RB_force(nu)
        coriolis_force_added = self.compute_C_A_force(nu_r)
        coriolis_force_RB_A = self.compute_C_RB_force(nu_r) + self.compute_C_A_force(
            nu_r
        )
        wave_force = self.compute_wave_force()

        # Full Body Dynamics
        nu_r_dot = if_else(
            complete_model,
            mtimes(
                self.mass_inv,
                (
                    thruster_force
                    + wave_force
                    - mtimes(self.rb_mass, nu_c_dot)
                    - mtimes(coriolis_force_rb, nu)
                    - mtimes(coriolis_force_added, nu_r)
                    - mtimes(damping_force, nu_r)
                    - restorive_force
                ),
            ),
            mtimes(
                self.mass_inv,
                (
                    thruster_force
                    + wave_force
                    - mtimes(coriolis_force_RB_A, nu_r)
                    - mtimes(damping_force, nu_r)
                    - restorive_force
                ),
            ),
        )

        x_dot = vertcat(eta_dot, nu_r_dot)
        # chi_dot = vertcat(eta_dot, nu_r_dot, nu_c_dot)

        return x_dot
