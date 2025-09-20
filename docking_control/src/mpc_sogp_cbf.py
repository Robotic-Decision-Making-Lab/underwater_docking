import numpy as np
import yaml
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from casadi import evalf, SX, mtimes, pinv, vertcat, hessian, Function, jacobian, dot, fmin, fmax, sumsqr, cos, sin, pi
from scipy.linalg import block_diag
import sys

sys.path.insert(0, "/home/ros/ws_dock/src/underwater_docking/docking_control/src")
from auv_hinsdale import AUV  # noqa: E402


class MPC:
    def __init__(self, auv, mpc_params):
        self.auv = auv
        self.dt = mpc_params["dt"]
        self.log_quiet = mpc_params["quiet"]
        self.horizon = mpc_params["horizon"]
        self.control_axes = mpc_params["control_axes"]
        self.model_type = mpc_params["full_body"]
        self.n_thrusters = mpc_params["thrusters"]

        self.T_horizon = self.horizon * self.dt

        self.thrust_coeff_k = 4e-6  # Thrust = k * omega * |omega|

        # --- State and Control Dimensions ---
        self.vehicle_state_dim = 12
        self.aug_state_dim = (
            self.vehicle_state_dim + self.n_thrusters
        )  # [x_vehicle, omega]
        # ------------------------------------

        self.cbf_type = mpc_params.get("cbf", {}).get("type", "tube")
        if self.cbf_type not in ["tube", "frustum", "sphere"]:
            raise ValueError(f"Unsupported CBF type: {self.cbf_type}")

        self.cbf_settling_time = mpc_params.get("cbf", {}).get("settling_time", 0.05)
        self.cbf_radius = []
        if self.cbf_type == "tube":
            # get radius or set default
            self.cbf_radius.append(mpc_params["cbf"]["specs"].get("radius", 0.1))
        elif self.cbf_type == "frustum":
            # get start and end radii or set defaults
            self.cbf_radius.append(mpc_params["cbf"]["specs"].get("start_radius", 0.5))
            self.cbf_radius.append(mpc_params["cbf"]["specs"].get("end_radius", 0.05))
        elif self.cbf_type == "sphere":
            self.cbf_radius.append(mpc_params["cbf"]["specs"].get("radius", 0.1))
        else:
            raise ValueError(f"Unsupported CBF type: {self.cbf_type}")

        self.cbf_x_init = SX.zeros(12, 1)
        self.cbf_enabled = mpc_params.get("cbf", {}).get("enabled", False)
        self.cbf_flag = (
            False  # Flag to indicate if CBF has been initialized appropriately
        )
        self.cbf_penalty = mpc_params.get("cbf", {}).get("penalty", 1000.0)

        self.Q_eta = float(mpc_params["noises"]["Q_eta"]) * np.eye(6)
        self.Q_nu_r = float(mpc_params["noises"]["Q_nu_r"]) * np.eye(6)
        self.Q_f = float(mpc_params["noises"]["Q_f"]) * np.eye(3)
        self.Q_auv_state = block_diag(self.Q_eta, self.Q_nu_r)

        self.R_att = float(mpc_params["noises"]["R_att"]) * np.eye(3)
        self.R_linvel = float(mpc_params["noises"]["R_linvel"]) * np.eye(3)
        self.R_angvel = float(mpc_params["noises"]["R_angvel"]) * np.eye(3)
        self.R_linacc = float(mpc_params["noises"]["R_linacc"]) * np.eye(3)
        self.R_xy = float(mpc_params["noises"]["R_xy"]) * np.eye(2)
        self.R_z = float(mpc_params["noises"]["R_z"])
        self.R_dr = float(mpc_params["noises"]["R_dr"])
        self.R_auv_meas = block_diag(
            self.R_xy,
            self.R_z,
            self.R_att,
            self.R_linvel,
            self.R_angvel,  # , self.R_linacc
        )
        self.R_f_meas = block_diag(self.R_linacc, self.R_dr)

        self.P = np.diag(mpc_params["penalty"]["P"])
        self.Q = np.diag(mpc_params["penalty"]["Q"])
        self.R = np.diag(mpc_params["penalty"]["R"])
        self.xmin = np.array(mpc_params["bounds"]["xmin"], dtype=np.float64).reshape(
            -1, 1
        )
        self.xmax = np.array(mpc_params["bounds"]["xmax"], dtype=np.float64).reshape(
            -1, 1
        )
        self.umin = np.array(mpc_params["bounds"]["umin"], dtype=np.float64).reshape(
            -1, 1
        )
        self.umax = np.array(mpc_params["bounds"]["umax"], dtype=np.float64).reshape(
            -1, 1
        )
        self.dumin = np.array(mpc_params["bounds"]["dumin"], dtype=np.float64).reshape(
            -1, 1
        )
        self.dumax = np.array(mpc_params["bounds"]["dumax"], dtype=np.float64).reshape(
            -1, 1
        )

        self.previous_control = np.zeros((self.control_axes, 1))

        self.acados_ocp = AcadosOcp()
        self.acados_model = AcadosModel()
        self.auv_model()
        self.create_ocp_solver_description()
        self.acados_ocp_solver = AcadosOcpSolver(
            self.acados_ocp,
            json_file="acados_nmpc_" + self.acados_ocp.model.name + ".json",
        )

    @classmethod
    def load_params(cls, auv_filename, mpc_filename):
        auv = AUV.load_params(auv_filename)

        f = open(mpc_filename, "r")
        mpc_params = yaml.load(f.read(), Loader=yaml.SafeLoader)

        return cls(auv, mpc_params)

    def reset(self):
        self.previous_control = None
        self.previous_state = None

    def initialize_cbf(self):
        x = SX.sym("x", (12, 1))
        xr = SX.sym("xr", (12, 1))
        u = SX.sym("u", (self.control_axes, 1))
        residual = SX.sym("residual", (12, 1))

        # Define a casadi function for computing the CBF constraints
        cbf_constraints = self.compute_cbf_constraints(x, xr, u, residual)
        self.cbf_constraints = Function(
            "cbf_constraints", [x, xr, u, residual], [cbf_constraints]
        )
        self.cbf_flag = True

    def compute_cbf_constraints(self, x, xr, wrench, residual):
        """
        Compute the constraints for the control barrier function

        u => 8x1 vector of control inputs
        x => 12x1 vector of states
        xr => 12x1 vector of reference states
        wrench => 6x1 vector of forces and torques
        f => 12x1 vector of dynamics
        g => 12x12 matrix of control input dynamics
        w => 12x1 vector of disturbances
        """
        f = (
            self.auv.compute_nonlinear_dynamics(x, 0.0, complete_model=True)[0:12, :]
            + residual
        )
        g = vertcat(SX.zeros(6, 6), self.auv.mass_inv)

        # --- Automatic CBF Parameter Tuning via Pole Placement ---
        # 1. Calculate the pole 'lambda' from the desired settling time.
        barrier_settling_time = self.cbf_settling_time
        lam = 4.0 / barrier_settling_time

        # 2. Set c1 and c2 for a critically damped response.
        c1 = lam
        c2 = lam
        # --------------------------------------------------------

        def frustum_barrier(
            x,
            xr,
            p_start,
            start_radius,
            end_radius,
            orientation_weight,
            max_align_angle_rad,
        ):
            """
            CBF to keep the AUV within a frustum-shaped corridor while also
            aligning its orientation with a target orientation.

            Args:
                x (ca.MX): The AUV's current state vector (12x1).
                xr (ca.MX): The AUV's reference state vector (12x1), including target position and orientation.
                p_start (ca.MX): The start point of the frustum's axis (3x1).
                start_radius (float): The radius of the corridor at the start point.
                end_radius (float): The radius of the corridor at the end point.
                orientation_weight (float): Tuning parameter to scale the orientation constraint's importance.
                max_align_angle_rad (float): The maximum allowed angle (in radians) between the AUV's
                                            forward vector and the target's forward vector.

            Returns:
                ca.MX: The value of the combined barrier function.
            """
            # --- 1. Positional Barrier (Unchanged) ---
            p_auv = x[0:3]
            p_end = xr[0:3]  # Extract target position from reference state
            v = p_end - p_start
            w = p_auv - p_start
            t = dot(w, v) / (dot(v, v) + 1e-9)
            t_clamped = fmax(0, fmin(1, t))
            p_closest = p_start + t_clamped * v
            d_sq = sumsqr(p_auv - p_closest)
            R_t = start_radius * (1 - t_clamped) + end_radius * t_clamped
            b_position = R_t**2 - d_sq

            # --- 2. Orientation Barrier (Aligns AUV frame with Target frame) ---
            # AUV's current orientation
            phi_auv, theta_auv, psi_auv = x[3], x[4], x[5]

            # AUV's forward vector (body x-axis) in the world frame
            v_auv_x = cos(psi_auv) * cos(theta_auv)
            v_auv_y = sin(psi_auv) * cos(theta_auv)
            v_auv_z = -sin(theta_auv)
            v_auv = vertcat(v_auv_x, v_auv_y, v_auv_z)

            # Target's orientation from the reference state
            phi_target, theta_target, psi_target = xr[3], xr[4], xr[5]

            # Target's forward vector (body x-axis) in the world frame
            v_target_orient_x = cos(psi_target) * cos(theta_target)
            v_target_orient_y = sin(psi_target) * cos(theta_target)
            v_target_orient_z = -sin(theta_target)
            v_target_orient = vertcat(
                v_target_orient_x, v_target_orient_y, v_target_orient_z
            )

            # Cosine of the alignment error angle between the two forward vectors
            cos_angle_error = dot(v_auv, v_target_orient)

            # The cosine of the maximum allowed alignment angle
            cos_max_angle = cos(max_align_angle_rad)

            # The orientation barrier is positive when the frames are aligned within the allowed cone
            b_orientation = cos_angle_error - cos_max_angle

            # --- 3. Combined Barrier ---
            b_combined = b_position  # + orientation_weight * b_orientation

            return b_combined

        def tube_barrier(x, p_start, p_end, tube_radius):
            """
            CBF to keep the AUV within a cylindrical tube between two points.

            Args:
                x (ca.MX): The AUV's state vector.
                p_start (ca.MX): The start point of the tube's axis [3x1].
                p_end (ca.MX): The end point of the tube's axis [3x1].
                tube_radius (float): The radius of the tube.

            Returns:
                ca.MX: The value of the barrier function.
            """
            # Extract the AUV's 3D position from the state vector
            p_auv = x[0:3]

            # Define the vector for the tube's central axis
            v = p_end - p_start
            # Define the vector from the start of the tube to the AUV
            w = p_auv - p_start

            # Calculate the projection parameter 't'.
            # This determines the closest point on the infinite line.
            # Add a small epsilon to prevent division by zero if start and end points are the same.
            t = dot(w, v) / (dot(v, v) + 1e-9)

            # Clamp 't' to the range [0, 1] to stay on the line *segment*.
            # This is the crucial step for creating a finite tube.
            t_clamped = fmax(0, fmin(1, t))

            # Calculate the closest point on the line segment
            p_closest = p_start + t_clamped * v

            # Calculate the squared distance from the AUV to the closest point
            d_sq = sumsqr(p_auv - p_closest)

            # The barrier function: b(x) is positive when inside the tube.
            b = tube_radius**2 - d_sq

            return b

        def sphere_barrier(x, xr, radius):
            # Define r
            x_diff = x[0:3, 0] - xr[0:3, 0]

            # Define b(x)
            b = (radius**2) - sumsqr(x_diff)
            return b

        if self.cbf_type == "frustum":
            b = frustum_barrier(
                x=x,
                xr=xr,
                p_start=self.cbf_x_init[0:3, 0],
                start_radius=self.cbf_radius[0],
                end_radius=self.cbf_radius[1],
                orientation_weight=5.0,
                max_align_angle_rad=45 * (pi / 180),
            )
        elif self.cbf_type == "tube":
            b = tube_barrier(
                x=x,
                p_start=self.cbf_x_init[0:3, 0],
                p_end=xr[0:3, 0],
                tube_radius=self.cbf_radius[0],
            )
        elif self.cbf_type == "sphere":
            b = sphere_barrier(x, xr, radius=self.cbf_radius[0])

        # # Compute the gradient (Jacobian of b with respect to x)
        # b_jac = jacobian(b, x)  # 1x12
        # # Compute the Hessian (second derivatives of b)
        # b_hessian = hessian(b, x)[0]  # 12x12
        # # b_hessian = jacobian(b_jac, x) # 12x12
        # # Compute the Jacobian of f (df/dx)
        # f_jac = jacobian(f, x)  # 12x12

        # # Compute the first derivative: ḃ = ∇b ⋅ f
        # b_dot = b_jac @ f  # 1x1
        # # Compute the second derivative: b̈ = ∇b ⋅ df/dx ⋅ f + fᵀ ⋅ Hessian(b) ⋅ f
        # b_ddot = b_jac @ f_jac @ f + f.T @ b_hessian @ f  # 1x1

        # hocbf = b_ddot + (c1 + c2) * b_dot + (c1 * c2) * b

        # Calculate Lie derivatives
        b_jac = jacobian(b, x)  # 1x12
        Lf_b = b_jac @ f  # Lie derivative of b with respect to f
        Lf2_b = jacobian(Lf_b, x) @ f  # Second Lie derivative of b with respect to f
        LgLf_b = (
            jacobian(Lf_b, x) @ g
        )  # Mixed Lie derivative of b with respect to f and g

        hocbf = Lf2_b + LgLf_b @ wrench + (c1 + c2) * Lf_b + (c1 * c2) * b

        return hocbf[0, 0]

    def auv_model(self):
        x = SX.sym("x", (12, 1))
        x_dot = SX.sym("x_dot", (12, 1))
        u = SX.sym("u", (self.control_axes, 1))
        p = SX.sym("p", (2 * self.vehicle_state_dim, 1))
        residual = p[self.vehicle_state_dim :]

        f_expl_expr = self.nonlinear_dynamics(x, u, self.model_type)[0:12] + residual
        f_impl_expr = x_dot - f_expl_expr

        self.acados_model.name = "bluerov2"
        self.acados_model.f_expl_expr = f_expl_expr
        self.acados_model.f_impl_expr = f_impl_expr
        self.acados_model.x = x
        self.acados_model.xdot = x_dot
        self.acados_model.u = u
        self.acados_model.p = p

    def nonlinear_dynamics(self, x, u, full_body):
        x_dot = self.auv.compute_nonlinear_dynamics(x, u, complete_model=full_body)
        return x_dot

    def create_ocp_solver_description(self):
        self.acados_ocp.model = self.acados_model
        x = self.acados_model.x
        u = self.acados_model.u
        nx = self.acados_model.x.shape[0]
        nu = self.acados_model.u.shape[0]
        self.acados_ocp.dims.N = self.horizon
        self.acados_ocp.parameter_values = np.zeros(2 * self.vehicle_state_dim)

        xr = self.acados_model.p[0:self.vehicle_state_dim]
        residual = self.acados_model.p[self.vehicle_state_dim:]

        self.acados_ocp.cost.cost_type = "NONLINEAR_LS"
        self.acados_ocp.cost.cost_type_e = "NONLINEAR_LS"

        # Terminal cost: Penalize terminal state error with matrix P
        self.acados_ocp.cost.W_e = self.P
        self.acados_ocp.model.cost_y_expr_e = x
        self.acados_ocp.cost.yref_e = np.zeros(nx)

        # Path cost: Penalize state and control effort with Q and R
        # The cost expression is a concatenation of state and control
        self.acados_ocp.cost.W = block_diag(self.Q, self.R)
        self.acados_ocp.model.cost_y_expr = vertcat(x, u)
        self.acados_ocp.cost.yref = np.zeros(nx + nu)

        self.acados_ocp.constraints.lbu = self.umin
        self.acados_ocp.constraints.ubu = self.umax
        self.acados_ocp.constraints.idxbu = np.arange(nu)

        self.acados_ocp.constraints.lbx = self.xmin
        self.acados_ocp.constraints.ubx = self.xmax
        self.acados_ocp.constraints.idxbx = np.arange(nx)

        if self.cbf_flag and self.cbf_enabled:
            cbf_val = self.cbf_constraints(x, xr, u, residual)

            # Inequality constraint cbf_val >= 0
            self.acados_model.con_h_expr = cbf_val
            self.acados_ocp.constraints.lh = 0.0
            self.acados_ocp.constraints.uh = np.inf

            # Soften the lower bound
            self.acados_ocp.constraints.lsh = 0.0
            self.acados_ocp.constraints.ush = np.inf
            self.acados_ocp.cost.Zl = np.array([self.cbf_penalty])
            self.acados_ocp.cost.zl = np.array([0.0])

        self.acados_ocp.constraints.x0 = np.zeros(nx)

        self.acados_ocp.solver_options.N_horizon = self.horizon
        self.acados_ocp.solver_options.tf = self.T_horizon

        # self.acados_ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        self.acados_ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        # self.acados_ocp.solver_options.qp_solver = "FULL_CONDENSING_DAQP"
        # self.acados_ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
        # self.acados_ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_OSQP"

        self.acados_ocp.solver_options.hessian_approx = (
            "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
        )

        self.acados_ocp.solver_options.integrator_type = "ERK"
        # self.acados_ocp.solver_options.integrator_type = "IRK"
        # self.acados_ocp.solver_options.sim_method_num_stages = 4
        # self.acados_ocp.solver_options.sim_method_num_steps = 1

        self.acados_ocp.solver_options.nlp_solver_type = "SQP"
        # self.acados_ocp.solver_options.nlp_solver_type = "SQP_RTI"

    def run_mpc(self, x, x_ref, residual):
        return self.optimize(x, x_ref, residual)

    def optimize(self, x, x_ref, residual):
        xr = x_ref[0:12, :]
        x0 = x[0:12, :]
        N = self.horizon

        nx = self.acados_ocp.model.x.shape[0]
        nu = self.acados_ocp.model.u.shape[0]

        # set initial state constraint
        self.acados_ocp_solver.set(0, "lbx", x0)
        self.acados_ocp_solver.set(0, "ubx", x0)

        ur = np.zeros((nu, 1))
        p = np.vstack((xr, residual))

        # Set the reference for the path cost
        path_y_ref = np.concatenate((xr, ur))

        for k in range(N):
            self.acados_ocp_solver.set(k, "p", p)
            self.acados_ocp_solver.set(k, "yref", path_y_ref)

        # Set the reference for the terminal cost (state only)
        terminal_y_ref = xr
        self.acados_ocp_solver.set(N, "yref", terminal_y_ref)
        self.acados_ocp_solver.set(N, "p", p)

        status = self.acados_ocp_solver.solve()
        if status != 0:
            print("acados returned status {}".format(status))

        wrench = self.acados_ocp_solver.get(0, "u").reshape(-1, 1)
        self.previous_control = wrench

        u_next = evalf(mtimes(pinv(self.auv.tam), wrench)).full()

        cost = self.acados_ocp_solver.get_cost()

        return u_next, wrench, cost
