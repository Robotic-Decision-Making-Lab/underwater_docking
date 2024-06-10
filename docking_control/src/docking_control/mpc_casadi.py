import yaml
import numpy as np
from casadi import DM, SX, Function, integrator, Opti, evalf, mtimes, vertcat, exp, pinv
import sys

sys.path.insert(
    0, "/home/darth/workspace/bluerov2_ws/src/bluerov2_dock/src/bluerov2_dock"
)

from auv_hinsdale import AUV


class MPC(object):
    def __init__(self, auv, mpc_params):
        """This class is used to control the AUV using the MPC controller.

        Args:
            auv: Vehicle object
            mpc_params: Dictionary containing the MPC parameters
        """
        self.auv = auv
        self.dt = mpc_params["dt"]
        self.log_quiet = mpc_params["quiet"]
        self.horizon = mpc_params["horizon"]
        self.thrusters = mpc_params["thrusters"]
        self.model_type = mpc_params["full_body"]

        self.P = np.diag(mpc_params["penalty"]["P"])
        self.Q = np.diag(mpc_params["penalty"]["Q"])
        self.R = np.diag(mpc_params["penalty"]["R"])

        self.xmin = np.array(mpc_params["bounds"]["xmin"], dtype=np.float)
        self.xmax = np.array(mpc_params["bounds"]["xmax"], dtype=np.float)

        # self.umin = -1 * np.ones((self.thrusters, 1), dtype=np.float)
        # self.umax = np.ones((self.thrusters, 1), dtype=np.float)

        self.umin = np.array(mpc_params["bounds"]["umin"], dtype=np.float)
        self.umax = np.array(mpc_params["bounds"]["umax"], dtype=np.float)

        # self.dumin = np.array(mpc_params["bounds"]["dumin"], dtype=np.float)
        # self.dumax = np.array(mpc_params["bounds"]["dumax"], dtype=np.float)

        self.wec_states = None

        self.reset()
        self.initialize_optimizer()

    @classmethod
    def load_params(cls, auv_filename, mpc_filename):
        auv = AUV.load_params(auv_filename)

        f = open(mpc_filename, "r")
        mpc_params = yaml.load(f.read(), Loader=yaml.SafeLoader)

        return cls(auv, mpc_params)

    def reset(self):
        """Reset the previous control and state values"""
        self.previous_control = None
        self.previous_state = None

    def initialize_optimizer(self):
        """Initialize the MPC optimizer"""
        x = SX.sym("x", (12, 1))
        # f_B = SX.sym('f_B', (3,1))
        # f_B_dot = SX.sym('f_B_dot', (3,1))
        u = SX.sym("u", (self.thrusters, 1))
        full_body = SX.sym("full_body")

        x_dot = self.auv.compute_nonlinear_dynamics(x, u, complete_model=full_body)
        f = Function("f", [x, u, full_body], [x_dot])

        T = self.dt * self.horizon
        N = self.horizon
        intg_options = {"tf": T / N, "simplify": True, "number_of_finite_elements": 4}

        dae = {"x": x, "p": vertcat(u, full_body), "ode": f(x, u, full_body)}
        intg = integrator("intg", "rk", dae, intg_options)
        x_next = intg(x0=x, p=vertcat(u, full_body))["xf"]
        self.forward_dynamics = Function("func", [x, u, full_body], [x_next])

    def run_mpc(self, x, x_ref):
        return self.optimize(x, x_ref)

    def optimize(self, x, x_ref):
        """Optimize the control inputs using the MPC controller

        Args:
            x: Initial state of the vehicle
            x_ref: Desired state of the vehicle

        Returns:
            u_next: Control inputs for the next time step
            force_axes: Thrust forces in the body frame
        """
        eta = x[0:6, :]

        # f_B = DM.zeros((3,1))
        # f_B_dot = DM.zeros((3,1))

        tf_B2I = self.auv.compute_transformation_matrix(eta)
        R_B2I = evalf(tf_B2I[0:3, 0:3])

        # xr = x_ref[0:3, :]
        # xr = x_ref[0:6, :]
        # x0 = x[0:6, :]
        xr = x_ref[0:12, :]
        x0 = x[0:12, :]
        cost = 0

        opt = Opti()

        X = opt.variable(12, self.horizon + 1)
        U = opt.variable(self.thrusters, self.horizon + 1)
        # X0 = opt.parameter(6, 1)
        X0 = opt.parameter(12, 1)

        # flow_bf = opt.parameter(3, 1)
        # flow_acc_bf = opt.parameter(3, 1)

        # Only do horizon rolling if our horizon is greater than 1
        if (self.horizon > 1) and (self.previous_control is not None):
            # Shift all commands one over, since we executed the first control action
            initial_guess_control = np.roll(self.previous_control, -1, axis=1)
            initial_guess_state = np.roll(self.previous_state, -1, axis=1)

            # Set the final column to the same as the second to last column
            initial_guess_control[:, -1] = initial_guess_control[:, -2]
            initial_guess_state[:, -1] = initial_guess_state[:, -2]

            opt.set_initial(U, initial_guess_control)
            opt.set_initial(X, initial_guess_state)

        for k in range(self.horizon):
            # cost += (X[0:3, k] - xr).T @ self.P @ (X[0:3, k] - xr)
            # cost += (X[0:6, k] - xr).T @ self.P @ (X[0:6, k] - xr)
            cost += (X[0:12, k] - xr).T @ self.P @ (X[0:12, k] - xr)
            cost += (U[:, k + 1] - U[:, k]).T @ self.R @ (U[:, k + 1] - U[:, k])
            # cost += (U[:, k]).T @ self.Q @ (U[:, k])

            opt.subject_to(
                X[:, k + 1] == self.forward_dynamics(X[:, k], U[:, k], self.model_type)
            )
            opt.subject_to(opt.bounded(self.xmin, X[0:12, k], self.xmax))
            opt.subject_to(opt.bounded(self.umin, U[:, k], self.umax))
            # opt.subject_to(opt.bounded(self.dumin, (U[:, k+1] - U[:, k]), self.dumax))

        # cost += (X[0:3, -1] - xr).T @ self.P @ (X[0:3, -1] - xr)
        # cost += (X[0:6, -1] - xr).T @ self.P @ (X[0:6, -1] - xr)
        cost += (X[0:12, -1] - xr).T @ self.P @ (X[0:12, -1] - xr)

        opt.subject_to(opt.bounded(self.xmin, X[0:12, -1], self.xmax))
        # opt.subject_to(X[0:6, 0] == X0)
        opt.subject_to(X[0:12, 0] == X0)

        opt.set_value(X0, x0)
        # opt.set_value(flow_bf, f_B)
        # opt.set_value(flow_acc_bf, f_B_dot)

        opt.minimize(cost)

        options = {"ipopt": {}}

        if self.log_quiet:
            options["ipopt"]["print_level"] = 0
            options["print_time"] = 0
        else:
            options["show_eval_warnings"] = True

        opt.solver("ipopt", options)
        sol = opt.solve()

        force_axes = sol.value(U)[:, 0:1]
        # x_next = sol.value(X)[:,1:2]

        self.previous_control = sol.value(U)
        self.previous_state = sol.value(X)

        u_next = evalf(mtimes(pinv(self.auv.tcm), force_axes)).full()

        return u_next, force_axes
