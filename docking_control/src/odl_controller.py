import os
import time
import numpy as np
import sys
import yaml
from typing import Dict, List, Tuple, Optional
import logging

sys.path.insert(0, "/home/ros/ws_dock/src/underwater_docking/docking_control/src")

from auv_hinsdale import AUV  # noqa: E402
from sogp import SOGP  # noqa: E402
from mpc_sogp_cbf import MPC  # noqa: E402


class ODL:
    def __init__(self):
        """
        This class is used to control the AUV using the MPC controller.
        """
        cwd = os.path.dirname(__file__)

        # auv_yaml = cwd + "/../config/auv_bluerov2.yaml"
        # mpc_yaml = cwd + "/../config/mpc_bluerov2.yaml"

        auv_yaml = cwd + "/../config/auv_bluerov2_heavy.yaml"
        mpc_yaml = cwd + "/../config/mpc_bluerov2_heavy.yaml"
        test_params_yaml = cwd + "/../config/test_params.yaml"

        self.test_params = self.load_yaml_params(test_params_yaml)
        gp_params = self.test_params.get("gp", {})
        self.gp_enabled: bool = bool(gp_params.get("enabled", False))
        self.gp_kernel_type: str = str(gp_params.get("kernel_type", "RBF_ISO")).upper()
        self.gp_hyperparams: Optional[Dict[str, float]] = gp_params.get(
            "hyperparameters", None
        )
        self.gp_input_dim: Optional[int] = None
        self.gp_output_dim: Optional[int] = None
        self.gp_max_bvs: int = int(gp_params.get("max_bvs", 50))
        self.gp_threshold: float = float(gp_params.get("threshold", 1e-3))
        self.gp_tuning_max_iters: int = int(gp_params.get("tuning_max_iters", 100))
        self.init_tuning_flag: bool = bool(gp_params.get("init_tuning_flag", False))
        self.training_samples: int = int(gp_params.get("training_samples", 50))

        self.tuned_hyperparams_list: Optional[List[Tuple]] = None
        self.sogp_models: Optional[List[SOGP]] = None

        self.online_tuning_flag: bool = bool(gp_params.get("online_tuning_flag", False))
        self.rho_sogp_list: Optional[List[float]] = None
        self.rho_threshold_sogp_tuning: float = float(
            gp_params.get("rho_threshold", 0.8)
        )
        self.sogp_tuning_min_bv_increment: int = int(
            gp_params.get("min_bv_for_rho_tune", 10)
        )
        self.sogp_hpo_buffer_size: int = int(
            gp_params.get("sogp_internal_hpo_buffer_size", 30)
        )
        self.sogp_cg_max_iter: int = int(gp_params.get("sogp_cg_max_iter", 10))
        self.sogp_ls_max_iter: int = int(gp_params.get("sogp_ls_max_iter", 10))

        # Change these values as desired
        self.tolerance = self.test_params["tolerance"]
        self.yaw_tolerance = self.test_params["yaw_tolerance"]
        self.path_length = 0.0
        self.p_times = [0]

        self.auv = AUV.load_params(auv_yaml)
        self.mpc = MPC.load_params(auv_yaml, mpc_yaml)

        self.control_axes = self.mpc.control_axes
        self.comp_time = 0.0
        self.time_id = 0
        self.dt = self.mpc.dt
        self.t_f = 3600.0
        self.t_span = np.arange(0.0, self.t_f, self.dt)
        self.mpc.reset()

        self.opt_data = {}
        self._initialize_sogp_models()

    def _initialize_sogp_models(self) -> None:
        """Initializes SOGP models based on self.gp_kernel_type."""
        self.gp_input_dim = 12 + self.mpc.control_axes + 1
        self.gp_output_dim = 12

        self.rho_sogp_list = [0.0] * self.gp_output_dim

        hyperparams: Optional[Tuple] = None

        default_hyperparams = {
            "length_scale": 50.0,
            "variance": 1.0,
            "noise_variance": 0.1,
            "period": 15.0,
        }

        loaded_params = self.gp_hyperparams or {}

        if not isinstance(loaded_params, dict):
            logging.warning(
                "YAML 'hyperparameters' format is invalid "
                f"(type: {type(loaded_params)}). "
                "Using all default values."
            )
            loaded_params = {}

        if not loaded_params:
            logging.warning(
                "No SOGP hyperparameters in YAML. Using all default values."
            )
        else:
            logging.info(
                "Loading SOGP hyperparameters from YAML, using "
                "defaults for any missing keys."
            )

        length_scale = loaded_params.get(
            "length_scale", default_hyperparams["length_scale"]
        )
        variance = loaded_params.get("variance", default_hyperparams["variance"])
        noise_variance = loaded_params.get(
            "noise_variance", default_hyperparams["noise_variance"]
        )
        period = loaded_params.get("period", default_hyperparams["period"])

        # 4. Convert values to the required log-space format for the SOGP model.
        log_ls = np.log(length_scale)
        log_signal_var = np.log(variance)
        log_noise_var = np.log(noise_variance)

        # 5. Construct the final hyperparameter tuple based on the kernel type.
        if self.gp_kernel_type == "RBF_ISO":
            hyperparams = (log_ls, log_signal_var, log_noise_var)
        elif self.gp_kernel_type == "RBF_ARD":
            log_ls_array = np.full(self.gp_input_dim, log_ls)
            hyperparams = (log_ls_array, log_signal_var, log_noise_var)
        elif self.gp_kernel_type == "PERIODIC":
            # The period is used directly, not in log space.
            hyperparams = (log_ls, log_signal_var, log_noise_var, period)
        elif self.gp_kernel_type == "COMBINATION":
            log_periodic_ls = np.log(loaded_params.get("periodic_length_scale", 50.0))
            log_periodic_signal_var = np.log(
                loaded_params.get("periodic_variance", 1.0)
            )
            log_periodic_noise_var = np.log(
                loaded_params.get("periodic_noise_variance", 0.1)
            )
            hyperparams = (
                log_ls,
                log_signal_var,
                log_noise_var,
                log_periodic_ls,
                log_periodic_signal_var,
                log_periodic_noise_var,
                period,
            )
        else:
            raise ValueError(f"Unsupported SOGP kernel type: {self.gp_kernel_type}")

        logging.info(f"Using SOGP hyperparameters: {hyperparams}")

        if not self.init_tuning_flag:
            self.tuned_hyperparams_list = [hyperparams] * self.gp_output_dim
            self.sogp_models = [
                SOGP(
                    input_dim=self.gp_input_dim,
                    hyperparams=hp_tuple,
                    kernel=self.gp_kernel_type,
                    max_bvs=self.gp_max_bvs,
                    threshold=self.gp_threshold,
                    hpo_buffer_size=self.sogp_hpo_buffer_size,
                )
                for hp_tuple in self.tuned_hyperparams_list
            ]
        else:
            self.sogp_models = []
            self.tuned_hyperparams_list = []

    def _update_and_predict_sogp(
        self,
        x_auv_pred: np.ndarray,
        wrench_mpc: np.ndarray,
        x_dot_true: np.ndarray,
        x_dot_nominal: np.ndarray,
        current_time: float,
    ) -> np.ndarray:
        """Updates SOGP models, predicts residual, handles rho-tuning."""
        gp_input = np.hstack((x_auv_pred.flatten(), wrench_mpc.flatten(), current_time))
        gp_output_target = (
            x_dot_true[0 : self.gp_output_dim, :]
            - x_dot_nominal[0 : self.gp_output_dim, :]
        )
        # print(f"GP Output Target: {gp_output_target.flatten()}")
        predicted_residual = np.zeros((self.gp_output_dim, 1))

        for j in range(self.gp_output_dim):
            if not self.sogp_models or j >= len(self.sogp_models):
                continue

            sogp_model_j = self.sogp_models[j]

            if sogp_model_j.num_bvs > 0:
                pred_mean, _ = sogp_model_j.predict(gp_input)
                predicted_residual[j] = pred_mean

            sogp_model_j.update(gp_input, gp_output_target[j, 0])

        return predicted_residual

    def load_yaml_params(self, filename: str):
        """
        Load parameters from a YAML file.

        Args:
            filename: Path to the YAML file.

        Returns:
            A dictionary containing the parameters.
        """
        with open(filename, "r") as f:
            params = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return params

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

    def run_mpc(self, x0, xr, residual):
        """This function runs the MPC controller.

        Args:
            x0: Initial vehicle pose
            xr: Desired vehicle pose
            residual: Residual error

        Returns:
            u: Control input
            wrench: Thruster forces
            done: True if the vehicle has reached the desired pose
        """

        if self.gp_enabled and np.any(np.abs(residual) > 1e2):
            logging.warning(f"Large GP residual {residual.T}, resetting to zero.")
            residual = np.zeros_like(residual)

        current_gp_res_for_mpc = (
            residual if self.gp_enabled else np.zeros_like(residual)
        )

        if abs(xr[0, 0]) < 0.30:
            xr[3:6, :] = 0.0
            xr[9:12, :] = 0.0

        process_t0 = time.perf_counter()
        self.distance = np.linalg.norm(x0[0:3, :] - xr[0:3, :])

        # x0[3:6, :] = self.wrap_pi2negpi(x0[3:6, :])
        # xr[5, :] += np.pi
        # xr[3:6, :] = self.wrap_pi2negpi(xr[3:6, :])

        # x0[6:12, :] = np.clip(
        #     x0[6:12, 0], self.mpc.xmin[6:12], self.mpc.xmax[6:12]
        # ).reshape(-1, 1)

        self.yaw_diff = abs((((x0[5, :] - xr[5, :]) + np.pi) % (2 * np.pi)) - np.pi)[0]
        self.ang_diff = np.linalg.norm(
            (((x0[3:6, :] - xr[3:6, :]) + np.pi) % (2 * np.pi)) - np.pi
        )

        if self.distance <= self.tolerance and self.yaw_diff <= self.yaw_tolerance:
            # if self.distance < self.tolerance and self.ang_diff < self.yaw_tolerance:
            return np.zeros((8, 1)), np.zeros((6, 1)), True

        else:
            # x0[3:5, :] = 0.0
            # x0[9:11, :] = 0.0
            u, wrench, _ = self.mpc.run_mpc(x0, xr, current_gp_res_for_mpc)

            # x_dot_sim = self.auv.compute_nonlinear_dynamics(x=x0, u=wrench)
            # x_sim = x0 + evalf(x_dot_sim).full() * self.dt

            # if xr is close to the dock, send the self.distance back to the controller
            # else send a large number
            if abs(xr[0, 0]) < 0.25:
                distance_to_dock = self.distance
            else:
                distance_to_dock = 100.0

            self.comp_time = time.perf_counter() - process_t0

            print(f"T = {round(self.t_span[self.time_id],3)}s")
            print(f"Computation Time = {round(self.comp_time,3)}s")
            print("----------------------------------------------")
            print(f"MPC Control Input: {np.round(u, 2).T}")
            print("----------------------------------------------")
            print(f"Axes Forces: {np.round(wrench, 2).T}")
            print("----------------------------------------------")
            print(f"GP Residual: {np.round(residual, 3).T}")
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
            print(f"(Dock-AUV) Angle difference: {np.round(self.ang_diff, 3)}")
            print("----------------------------------------------")
            # print("")

            self.time_id += 1



            return u, wrench, False, distance_to_dock


if __name__ == "__main__":
    mpc = ODL()
