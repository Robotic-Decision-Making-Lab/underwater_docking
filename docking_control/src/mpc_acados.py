import numpy as np
import yaml
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from casadi import evalf, SX, mtimes, pinv, vertcat
from scipy.linalg import block_diag
import sys

sys.path.insert(0, "/home/ros/ws_dock/src/underwater_docking/docking_control/src")
from auv_hinsdale import AUV  # noqa: E402


class MPC:
    def __init__(self, auv, mpc_params):
        """This class is used to control the AUV using the MPC controller.

        Args:
            auv: Vehicle object
            mpc_params: Dictionary containing the MPC parameters
        """
        self.auv = auv
        self.dt = mpc_params["dt"]
        self.log_quiet = mpc_params["quiet"]
        self.control_axes = mpc_params["control_axes"]
        self.horizon = mpc_params["horizon"]
        self.T_horizon = self.horizon * self.dt
        self.model_type = mpc_params["full_body"]
        self.P = np.diag(mpc_params["penalty"]["P"])
        self.Q = np.diag(mpc_params["penalty"]["Q"])
        self.R = np.diag(mpc_params["penalty"]["R"])
        self.xmin = np.array(mpc_params["bounds"]["xmin"], dtype=np.float)
        self.xmax = np.array(mpc_params["bounds"]["xmax"], dtype=np.float)
        self.umin = np.array(mpc_params["bounds"]["umin"], dtype=np.float)
        self.umax = np.array(mpc_params["bounds"]["umax"], dtype=np.float)

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

    def auv_model(self):
        """Create the AUV model for the MPC controller."""
        x = SX.sym("x", (12, 1))
        x_dot = SX.sym("x_dot", (12, 1))
        u = SX.sym("u", (self.control_axes, 1))

        f_expl_expr = self.nonlinear_dynamics(x, u, self.model_type)
        f_impl_expr = x_dot - f_expl_expr

        p = []

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
        """Create the optimal control problem (OCP) solver description"""
        self.acados_ocp.model = self.acados_model
        x = self.acados_model.x
        u = self.acados_model.u
        nx = x.shape[0]
        nu = u.shape[0]

        self.acados_ocp.dims.N = self.horizon
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

        # Set constraints
        # Control input constraints
        self.acados_ocp.constraints.lbu = self.umin
        self.acados_ocp.constraints.ubu = self.umax
        self.acados_ocp.constraints.idxbu = np.arange(nu)

        # State constraints
        self.acados_ocp.constraints.lbx = self.xmin
        self.acados_ocp.constraints.ubx = self.xmax
        self.acados_ocp.constraints.idxbx = np.arange(nx)

        # Initial state constraint will be set in the solver loop
        self.acados_ocp.constraints.x0 = np.zeros(nx)

        # set options
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

        # self.acados_ocp.solver_options.nlp_solver_type = "SQP"
        self.acados_ocp.solver_options.nlp_solver_type = "SQP_RTI"
        # self.acados_ocp.solver_options.nlp_solver_type = "DDP"

    def reset(self):
        """Reset the previous control to None."""
        self.previous_control = None

    def run_mpc(self, x, x_ref):
        return self.optimize(x, x_ref)

    def optimize(self, x, x_ref):
        """Optimize the control input using the MPC controller.

        Args:
            x: Initial vehicle state
            x_ref: Desired vehicle state

        Returns:
            u_next: Control input
        """
        xr = x_ref[0:12, :]
        x0 = x[0:12, :]
        N = self.horizon

        self.acados_ocp.model.x.shape[0]
        nu = self.acados_ocp.model.u.shape[0]

        self.acados_ocp_solver.set(0, "lbx", x0)
        self.acados_ocp_solver.set(0, "ubx", x0)

        # The control reference is usually zero (minimize control effort)
        u_ref = np.zeros((nu, 1))

        # Set the reference for the path cost
        path_y_ref = np.concatenate((xr, u_ref))

        for k in range(N):
            self.acados_ocp_solver.set(k, "yref", path_y_ref)

        # Set the reference for the terminal cost (state only)
        terminal_y_ref = xr
        self.acados_ocp_solver.set(N, "yref", terminal_y_ref)

        # Solve the OCP
        status = self.acados_ocp_solver.solve()

        if status != 0:
            print(f"acados solver returned status {status}")

        wrench = self.acados_ocp_solver.get(0, "u")

        # dist_err = abs(x0[0, :] - xr[0, :])
        # yaw_err = abs((((x0[5, :] - xr[5, :]) + np.pi) % (2 * np.pi)) - np.pi)[0]

        # if dist_err < 0.30 and yaw_err > 0.05:
        #     wrench[0:5] = 0.0

        # if dist_err < 0.15 and yaw_err < 0.035:
        #     wrench[1:6] = 0.0

        u_next = evalf(mtimes(pinv(self.auv.tam), wrench)).full()

        return u_next, wrench
