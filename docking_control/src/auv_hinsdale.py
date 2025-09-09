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
    cross,
)


class AUV(object):
    def __init__(self, vehicle_dynamics):
        self.vehicle_mass = vehicle_dynamics["vehicle_mass"]
        self.rb_mass = vehicle_dynamics["rb_mass"]
        self.added_mass = vehicle_dynamics["added_mass"]
        self.lin_damp = vehicle_dynamics["lin_damp"]
        self.quad_damp = vehicle_dynamics["quad_damp"]
        self.tam = vehicle_dynamics["tam"]
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

        # Precompute the total mass matrix (w/added mass)
        # inverted for future dynamic calls
        self.mass_inv = inv(self.rb_mass + self.added_mass)
        self.rb_mass_inv = inv(self.rb_mass)
        self.added_mass_inv = inv(self.added_mass)

    @classmethod
    def load_params(cls, filename):
        f = open(filename, "r")
        params = yaml.load(f.read(), Loader=yaml.SafeLoader)

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
        vehicle_dynamics["vehicle_mass"] = m
        vehicle_dynamics["rb_mass"] = rb_mass
        vehicle_dynamics["added_mass"] = -diag(SX(params["m_added"]))
        vehicle_dynamics["lin_damp"] = -diag(SX(params["d_lin"]))
        vehicle_dynamics["quad_damp"] = -diag(SX(params["d_quad"]))
        vehicle_dynamics["tam"] = SX(params["tam"])
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
        # Rotation about x-axis (roll)
        rot_x = SX.eye(3)
        rot_x[1, 1] = cos(x[3])
        rot_x[1, 2] = -sin(x[3])
        rot_x[2, 1] = sin(x[3])
        rot_x[2, 2] = cos(x[3])

        # Rotation about y-axis (pitch)
        rot_y = SX.eye(3)
        rot_y[0, 0] = cos(x[4])
        rot_y[0, 2] = sin(x[4])
        rot_y[2, 0] = -sin(x[4])
        rot_y[2, 2] = cos(x[4])

        # Rotation about z-axis (yaw)
        rot_z = SX.eye(3)
        rot_z[0, 0] = cos(x[5])
        rot_z[0, 1] = -sin(x[5])
        rot_z[1, 0] = sin(x[5])
        rot_z[1, 1] = cos(x[5])

        # Combined rotation matrix (ZYX convention)
        rot_mtx = mtimes(rot_z, mtimes(rot_y, rot_x))

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
        v1 = v[0:3, 0]
        v2 = v[3:6, 0]

        m11 = self.rb_mass[0:3, 0:3]
        m12 = self.rb_mass[0:3, 3:6]
        m21 = self.rb_mass[3:6, 0:3]
        m22 = self.rb_mass[3:6, 3:6]

        c1 = (m11 @ v1) + (m12 @ v2)
        c2 = (m21 @ v1) + (m22 @ v2)

        c1_skew = skew(c1)
        c2_skew = skew(c2)

        coriolis_force = SX.zeros((6, 6))
        coriolis_force[0:3, 3:6] = -c1_skew
        coriolis_force[3:6, 0:3] = -c1_skew
        coriolis_force[3:6, 3:6] = -c2_skew

        return coriolis_force

    def compute_C_A_force(self, v):
        v1 = v[0:3, 0]
        v2 = v[3:6, 0]

        a11 = self.added_mass[0:3, 0:3]
        a12 = self.added_mass[0:3, 3:6]
        a21 = self.added_mass[3:6, 0:3]
        a22 = self.added_mass[3:6, 3:6]

        c1 = (a11 @ v1) + (a12 @ v2)
        c2 = (a21 @ v1) + (a22 @ v2)

        c1_skew = skew(c1)
        c2_skew = skew(c2)

        coriolis_force = SX.zeros((6, 6))
        coriolis_force[0:3, 3:6] = -c1_skew
        coriolis_force[3:6, 0:3] = -c1_skew
        coriolis_force[3:6, 3:6] = -c2_skew

        return coriolis_force

    def compute_damping_force(self, v):
        damping_force = (self.quad_damp * fabs(v)) + self.lin_damp
        return damping_force

    def compute_restorive_force(self, x):
        if self.neutral_bouy:
            restorive_force = (self.cog_to_cob * self.W) * vertcat(
                0.0, 0.0, 0.0, cos(x[4]) * sin(x[3]), sin(x[4]), 0.0
            )
        else:
            fg = SX.zeros((3, 1))
            fg[2, 0] = self.W

            fb = SX.zeros((3, 1))
            fb[2, 0] = -self.B

            rot_t = self.compute_transformation_matrix(x)[0:3, 0:3].T

            restorive_force = SX.zeros((6, 1))
            restorive_force[0:3, 0] = rot_t @ (fg + fb)
            restorive_force[3:6, 0] = cross(self.cog, (rot_t @ fg)) + cross(
                self.cob, (rot_t @ fb)
            )
            restorive_force *= -1
        return restorive_force

    def compute_nonlinear_dynamics(
        self,
        x,
        u,
        f_B=SX.zeros((3, 1)),
        f_B_dot=SX.zeros((3, 1)),
        complete_model=False,
        env_forces=SX.zeros((6, 1)),
    ):
        eta = x[0:6, :]
        nu_r = x[6:12, :]

        # Gets the transformation matrix to convert from Body to NED frame
        tf_mtx = self.compute_transformation_matrix(eta)

        nu_c = vertcat(f_B, SX.zeros((3, 1)))
        nu_c_dot = vertcat(f_B_dot, SX.zeros((3, 1)))

        # Computes total vehicle velocity
        nu = nu_r + nu_c

        # Kinematic Equation
        # Convert the relative velocity from Body to NED and add it with the ocean
        # current velocity in NED to get the total velocity of the vehicle in NED
        eta_dot = if_else(
            complete_model, mtimes(tf_mtx, (nu_r + nu_c)), mtimes(tf_mtx, nu_r)
        )

        # Force computation
        thruster_force = u
        restorive_force = self.compute_restorive_force(eta)
        damping_force = self.compute_damping_force(nu_r)
        coriolis_force_rb = self.compute_C_RB_force(nu)
        coriolis_force_added = self.compute_C_A_force(nu_r)
        coriolis_force_RB_A = self.compute_C_RB_force(nu_r) + self.compute_C_A_force(
            nu_r
        )

        nu_r_dot = if_else(
            complete_model,
            mtimes(
                self.mass_inv,
                (
                    env_forces
                    + thruster_force
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
                    - mtimes(coriolis_force_RB_A, nu_r)
                    - mtimes(damping_force, nu_r)
                    - restorive_force
                ),
            ),
        )

        x_dot = vertcat(eta_dot, nu_r_dot)

        return x_dot
