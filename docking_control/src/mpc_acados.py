import numpy as np
import yaml
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from casadi import evalf, SX, mtimes, pinv, hessian, vertcat
import sys

sys.path.insert(0, "/home/ros/ws_dock/src/underwater_docking/docking_control/src")
from auv_hinsdale import AUV  # noqa: E402


class MPC:
    def __init__(self, auv, mpc_params):
        self.auv = auv
        self.dt = mpc_params["dt"]
        self.log_quiet = mpc_params["quiet"]
        self.thrusters = mpc_params["thrusters"]
        self.horizon = mpc_params["horizon"]
        self.T_horizon = self.horizon * self.dt
        self.model_type = mpc_params["full_body"]
        self.P = np.diag(mpc_params["penalty"]["P"])
        self.Q = np.diag(mpc_params["penalty"]["Q"])
        self.R = np.diag(mpc_params["penalty"]["R"])
        self.xmin = np.array(mpc_params["bounds"]["xmin"], dtype=np.float64)
        self.xmax = np.array(mpc_params["bounds"]["xmax"], dtype=np.float64)
        self.umin = np.array(mpc_params["bounds"]["umin"], dtype=np.float64)
        self.umax = np.array(mpc_params["bounds"]["umax"], dtype=np.float64)
        self.dumin = np.array(mpc_params["bounds"]["dumin"], dtype=np.float64)
        self.dumax = np.array(mpc_params["bounds"]["dumax"], dtype=np.float64)

        # self.previous_control = np.random.uniform(-0.1, 0.1, size=(self.thrusters,1))
        # self.previous_control = np.zeros((self.thrusters, 1))

        self.previous_control = []
        self.previous_states = []
        self.flag = False

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
        x = SX.sym("x", (12, 1))
        x_dot = SX.sym("x_dot", (12, 1))
        u = SX.sym("u", (self.thrusters, 1))
        p = SX.sym("p", 18, 1)

        f_expl_expr = self.nonlinear_dynamics(x, u, self.model_type)
        f_impl_expr = x_dot - f_expl_expr

        # p[:] = vertcat(xr, u_prev)
        # p = [xr, u_prev]
        # p = np.arange(0, 18)
        # p = []

        self.acados_model.name = "bluerov2"
        self.acados_model.f_expl_expr = f_expl_expr
        self.acados_model.f_impl_expr = f_impl_expr
        # self.acados_model.disc_dyn_expr = f_expl_expr
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
        self.acados_ocp.parameter_values = np.zeros(18)
        xr = self.acados_model.p[0:12]
        u_prev = self.acados_model.p[12:]

        self.acados_ocp.cost.cost_type = "EXTERNAL"
        self.acados_ocp.cost.cost_type_e = "EXTERNAL"

        # self.acados_ocp.model.cost_expr_ext_cost = x.T @ self.P @ x + u.T @ self.R @ u
        # self.acados_ocp.model.cost_expr_ext_cost_e = x.T @ self.P @ x

        cost = (xr - x).T @ self.P @ (xr - x) + (u - u_prev).T @ self.R @ (u - u_prev)
        # cost = (xr - x).T @ self.P @ (xr - x) + u.T @ self.R @ u
        cost_e = (xr - x).T @ self.P @ (xr - x)

        hessian_cost, grad_cost = hessian(cost, vertcat(u, x))
        hessian_cost_e, grad_cost_e = hessian(cost_e, x)

        # pdb.set_trace()
        self.acados_ocp.model.p = vertcat(xr, u_prev)

        self.acados_ocp.model.cost_expr_ext_cost = cost
        self.acados_ocp.model.cost_expr_ext_cost_e = cost_e

        self.acados_ocp.model.cost_expr_ext_cost_custom_hess = hessian_cost
        self.acados_ocp.model.cost_expr_ext_cost_custom_hess_e = hessian_cost_e

        self.acados_ocp.constraints.lbu = self.umin
        self.acados_ocp.constraints.ubu = self.umax
        self.acados_ocp.constraints.idxbu = np.arange(nu)

        # nonlinear_constraint = u - u_prev
        # self.acados_model.con_h_expr = nonlinear_constraint
        # self.acados_ocp.dims.nh = nu
        # self.acados_ocp.constraints.lh = self.dumin
        # self.acados_ocp.constraints.uh = self.dumax

        self.acados_ocp.constraints.lbx_0 = self.xmin
        self.acados_ocp.constraints.ubx_0 = self.xmax
        self.acados_ocp.constraints.idxbx_0 = np.arange(nx)

        self.acados_ocp.constraints.lbx = self.xmin
        self.acados_ocp.constraints.ubx = self.xmax
        self.acados_ocp.constraints.idxbx = np.arange(nx)

        self.acados_ocp.constraints.lbx_e = self.xmin
        self.acados_ocp.constraints.ubx_e = self.xmax
        self.acados_ocp.constraints.idxbx_e = np.arange(nx)

        # self.acados_ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        # self.acados_ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_OSQP"
        # self.acados_ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
        self.acados_ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"

        self.acados_ocp.solver_options.hessian_approx = "EXACT"
        # self.acados_ocp.solver_options.hessian_approx = "GAUSS_NEWTON"

        # self.acados_ocp.solver_options.regularize_method = "CONVEXIFY"

        # self.acados_ocp.solver_options.integrator_type = "DISCRETE"
        # self.acados_ocp.solver_options.integrator_type = "ERK"
        self.acados_ocp.solver_options.integrator_type = "IRK"
        self.acados_ocp.solver_options.sim_method_num_stages = 4
        self.acados_ocp.solver_options.sim_method_num_steps = 1

        self.acados_ocp.solver_options.nlp_solver_type = "SQP"
        # self.acados_ocp.solver_options.nlp_solver_type = "SQP_RTI"
        # self.acados_ocp.solver_options.nlp_solver_type = "DDP"

        self.acados_ocp.solver_options.exact_hess_cost = 1

        # self.acados_ocp.solver_options.nlp_solver_max_iter = 500
        self.acados_ocp.solver_options.nlp_solver_max_iter = 1000
        # self.acados_ocp.solver_options.levenberg_marquardt = 1e-5
        # self.acados_ocp.solver_options.line_search_use_sufficient_descent = 1
        # self.acados_ocp.solver_options.globalization = "MERIT_BACKTRACKING"

        self.acados_ocp.solver_options.qp_solver_warm_start = 1

        self.acados_ocp.solver_options.qp_solver_cond_N = self.horizon
        self.acados_ocp.solver_options.tf = self.T_horizon

    def reset(self):
        """Reset the previous control to None."""
        self.previous_control = []
        self.previous_states = []
        self.flag = False

    def warm_start(self, x0):
        # self.previous_control = self.acados_ocp_solver.solve_for_x0(x0, fail_on_nonzero_status=False)
        # self.previous_control = np.array([self.previous_control]).T
        self.acados_ocp_solver.set(0, "lbx", x0)
        self.acados_ocp_solver.set(0, "ubx", x0)

        status = self.acados_ocp_solver.solve()
        if status != 0:
            print("acados returned status {}".format(status))

        self.previous_control = []
        self.previous_states = []

        for k in range(self.horizon):
            wrench = self.acados_ocp_solver.get(k, "u")
            self.previous_control.append(wrench)

        for k in range(self.horizon + 1):
            state = self.acados_ocp_solver.get(k, "x")
            self.previous_states.append(state)

        self.previous_control = np.array(self.previous_control).T
        self.previous_states = np.array(self.previous_states).T

    def run_mpc(self, x, x_ref):
        return self.optimize(x, x_ref)

    def optimize(self, x, x_ref):
        xr = x_ref[0:12, :]
        x0 = x[0:12, :]
        N = self.horizon

        if not self.flag:
            print("Warm starting")
            self.warm_start(x0)

            self.flag = True

        # initialize solver
        for stage in range(N + 1):
            self.acados_ocp_solver.set(stage, "x", self.previous_states[:, stage])
        for stage in range(N):
            self.acados_ocp_solver.set(stage, "u", self.previous_control[:, stage])

        # set initial state constraint
        self.acados_ocp_solver.set(0, "lbx", x0)
        self.acados_ocp_solver.set(0, "ubx", x0)

        for k in range(N):
            if k == 0:
                u_prev = np.array([self.previous_control[:, k]]).T
            else:
                u_prev = np.array([self.acados_ocp_solver.get(k-1, "u")]).T

            self.acados_ocp_solver.set(k, "p", np.vstack((xr, u_prev)))
        self.acados_ocp_solver.set(N, "p", np.vstack((xr, u_prev)))

        self.acados_ocp_solver.options_set("qp_warm_start", 1)

        status = self.acados_ocp_solver.solve()
        if status != 0:
            print("acados returned status {}".format(status))

        # Store the control inputs and states for the next iteration
        self.previous_control = []
        self.previous_states = []

        for k in range(self.horizon):
            wrench = self.acados_ocp_solver.get(k, "u")
            self.previous_control.append(wrench)

        for k in range(self.horizon + 1):
            state = self.acados_ocp_solver.get(k, "x")
            self.previous_states.append(state)

        self.previous_control = np.array(self.previous_control).T
        self.previous_states = np.array(self.previous_states).T

        wrench_next = self.acados_ocp_solver.get(0, "u")

        # dist_err = abs(x0[0, :] - xr[0, :])
        # yaw_err = abs((((x0[5, :] - xr[5, :]) + np.pi) % (2 * np.pi)) - np.pi)[0]

        # if dist_err < 0.25 and yaw_err > 0.035:
        #     wrench[0:5] = 0.0

        # if dist_err < 0.15 and yaw_err < 0.035:
        #     wrench[1:6] = 0.0

        u_next = evalf(mtimes(pinv(self.auv.tcm), wrench_next)).full()

        return u_next, wrench_next
