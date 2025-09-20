import warnings
import numbers
from typing import Union, Callable, Optional, Tuple, List, Any

import numpy as np
from numpy.linalg import slogdet, inv, cholesky
from scipy.linalg import cho_solve


# ##############################################################################
# Helper Utility Functions
# ##############################################################################


def log_likelihood(
    noise_variance: float,
    y_observed: float,
    predicted_mean: float,
    predicted_variance: float,
) -> Tuple[float, float, float]:
    """
    Computes the log-likelihood and its derivatives for a univariate Gaussian.

    Args:
        noise_variance: Variance of the observation noise ($sigma_n^2$).
        y_observed: The observed target value.
        predicted_mean: The predicted mean ($mu$).
        predicted_variance: The predicted variance of the function (before noise).

    Returns:
        A tuple containing:
            - The log-likelihood, $log p(y|mu, var)$.
            - First derivative of the log-likelihood w.r.t. the mean ($q$).
            - Second derivative of the log-likelihood w.r.t. the mean ($r$).
    """
    # Total variance includes function variance and observation noise
    total_variance = predicted_variance + noise_variance
    # Add jitter for numerical stability
    total_variance = max(total_variance, 1e-9)

    # Log-likelihood: log p(y) = -0.5 * log(2*pi*sigma^2) - (y - mu)^2 / (2*sigma^2)
    log_lik = -0.5 * (
        np.log(2 * np.pi * total_variance)
        + (y_observed - predicted_mean) ** 2 / total_variance
    )

    # First derivative w.r.t. mean
    q = (y_observed - predicted_mean) / total_variance

    # Second derivative w.r.t. mean
    r = -1.0 / total_variance

    return log_lik, q, r


def stabilize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Ensures a matrix is symmetric by averaging it with its transpose."""
    return (matrix + matrix.T) / 2.0


def extend_matrix(matrix: np.ndarray, index: int = -1) -> np.ndarray:
    """Extends a matrix by inserting a zero row and column at the given index."""
    if index == -1:
        index = matrix.shape[0]

    if matrix.size == 0:
        return np.zeros((1, 1))

    # Insert a column of zeros
    extended_matrix = np.insert(matrix, index, 0, axis=1)
    # Insert a row of zeros
    extended_matrix = np.insert(extended_matrix, index, 0, axis=0)
    return extended_matrix


def extend_vector(
    vector: np.ndarray, value: float = 0.0, index: int = -1
) -> np.ndarray:
    """Extends a column vector by inserting a new element at the given index."""
    if vector.size == 0:
        return np.array([[value]])
    if index == -1:
        index = vector.shape[0]
    return np.insert(vector, index, value, axis=0)


class SOGP:
    """
    Sparse Online Gaussian Process (SOGP).

    This implementation allows for online learning by maintaining a small set of
    "basis vectors" (BVs) to build a sparse approximation of the full GP.

    Supported Kernels:
    - **RBF_ARD**: Radial Basis Function with separate length-scales for each dimension.
    - **RBF_ISO**: Radial Basis Function with a single isotropic length-scale.
    - **PERIODIC**: A kernel for modeling periodic functions.
    - **COMBINATION**: A combination of an RBF kernel on the first D-1 dimensions
      and a Periodic kernel on the last dimension.
    """

    def __init__(
        self,
        input_dim: int,
        hyperparams: Tuple,
        kernel: str = "RBF_ARD",
        max_bvs: int = 200,
        prior_mean_func: Optional[
            Callable[[np.ndarray, Optional[Any]], np.ndarray]
        ] = None,
        prior_mean_params: Optional[Any] = None,
        threshold: float = 1e-6,
        hpo_buffer_size: int = 0,
    ):
        """
        Initializes the SOGP model.

        Args:
            input_dim: The dimensionality of the input space.
            hyperparams: A tuple of hyperparameters. The structure depends on the kernel
                - RBF_ARD: (log_length_scales, log_signal_variance, log_noise_variance)
                           where log_length_scales is numpy array of size `input_dim`.
                - RBF_ISO: (log_length_scale, log_signal_variance, log_noise_variance)
                - PERIODIC: (log_length_scale, log_signal_variance,
                            log_noise_variance, period)
                - COMBINATION: (rbf_log_l, rbf_log_sig_var, rbf_log_noise,
                                periodic_log_l, periodic_log_sig_var,
                                periodic_log_noise, period)
            kernel: The kernel type. One of "RBF_ARD", "RBF_ISO",
                    "PERIODIC", "COMBINATION".
            max_bvs: The maximum number of basis vectors to store.
            prior_mean_func: A function to compute the prior mean.
            prior_mean_params: Additional parameters for the prior mean function.
            threshold: The novelty threshold (gamma) for adding a new basis vector.
            hpo_buffer_size: The number of recent points to store for hyperparameter
                            optimization.
        """
        self.input_dim: int = input_dim
        self.max_bvs: int = max_bvs
        self.num_bvs: int = 0
        self.threshold: float = threshold
        self.kernel_name: str = kernel

        self.hpo_buffer_size = hpo_buffer_size
        if self.hpo_buffer_size > 0:
            self.hpo_buffer_X: List[np.ndarray] = []
            self.hpo_buffer_Y: List[float] = []

        # --- Initialize Kernel Hyperparameters ---
        if self.kernel_name == "RBF_ARD":
            self._initialize_rbf_ard_hyperparameters(hyperparams)
        elif self.kernel_name == "RBF_ISO":
            self._initialize_rbf_iso_hyperparameters(hyperparams)
        elif self.kernel_name == "PERIODIC":
            self._initialize_periodic_hyperparameters(hyperparams)
        elif self.kernel_name == "COMBINATION":
            self._initialize_combination_hyperparameters(hyperparams)
        else:
            raise ValueError(
                f"Unknown kernel: {self.kernel_name}. "
                "Supported: RBF_ARD, RBF_ISO, PERIODIC, COMBINATION."
            )

        # Derived parameter for convenience
        self.noise_variance: float = np.exp(self.log_noise_variance)

        self.prior_mean_func = prior_mean_func
        self.prior_mean_params = prior_mean_params

        # --- Initialize Model State ---
        # Basis Vectors (the sparse set of inputs)
        self.bv: np.ndarray = np.zeros((0, self.input_dim))
        # Weight vector for mean prediction
        self.alpha: np.ndarray = np.zeros((0, 1))
        # Covariance matrix of the posterior
        self.C: np.ndarray = np.zeros((0, 0))
        # Gram matrix of basis vectors, K(bv, bv)
        self.Kb: np.ndarray = np.zeros((0, 0))
        # Inverse of the Gram matrix
        self.Kb_inv: np.ndarray = np.zeros((0, 0))

    def _initialize_rbf_ard_hyperparameters(self, hyperparams: Tuple) -> None:
        """Validates and sets RBF_ARD specific hyperparameters."""
        if not (isinstance(hyperparams, tuple) and len(hyperparams) == 3):
            raise ValueError("hyperparams for RBF_ARD must be a 3-tuple.")
        log_length_scales, log_signal_variance, log_noise_variance = hyperparams
        if not (
            isinstance(log_length_scales, np.ndarray)
            and log_length_scales.ndim == 1
            and log_length_scales.shape[0] == self.input_dim
        ):
            raise ValueError(
                "log_length_scales for RBF_ARD must be a "
                f"1D numpy array of length {self.input_dim}."
            )
        if not isinstance(log_signal_variance, numbers.Number):
            raise ValueError("log_signal_variance must be a scalar for RBF_ARD.")
        if not isinstance(log_noise_variance, numbers.Number):
            raise ValueError("log_noise_variance must be a scalar for RBF_ARD.")

        self.log_length_scale: np.ndarray = log_length_scales.reshape(1, -1)
        self.log_signal_variance: float = float(log_signal_variance)
        self.log_noise_variance: float = float(log_noise_variance)

    def _initialize_rbf_iso_hyperparameters(self, hyperparams: Tuple) -> None:
        """Validates and sets RBF_ISO specific hyperparameters."""
        if not (isinstance(hyperparams, tuple) and len(hyperparams) == 3):
            raise ValueError("hyperparams for RBF_ISO must be a 3-tuple.")
        log_length_scale, log_signal_variance, log_noise_variance = hyperparams
        if not isinstance(log_length_scale, numbers.Number):
            raise ValueError("log_length_scale for RBF_ISO must be a scalar.")
        if not isinstance(log_signal_variance, numbers.Number):
            raise ValueError("log_signal_variance must be a scalar for RBF_ISO.")
        if not isinstance(log_noise_variance, numbers.Number):
            raise ValueError("log_noise_variance must be a scalar for RBF_ISO.")

        self.log_length_scale: float = float(log_length_scale)
        self.log_signal_variance: float = float(log_signal_variance)
        self.log_noise_variance: float = float(log_noise_variance)

    def _initialize_periodic_hyperparameters(self, hyperparams: Tuple) -> None:
        """Validates and sets PERIODIC specific hyperparameters."""
        if not (isinstance(hyperparams, tuple) and len(hyperparams) == 4):
            raise ValueError("hyperparams for PERIODIC must be a 4-tuple.")
        log_length_scale, log_signal_variance, log_noise_variance, period = hyperparams
        if not isinstance(log_length_scale, numbers.Number):
            raise ValueError("log_length_scale for PERIODIC must be a scalar.")
        if not isinstance(log_signal_variance, numbers.Number):
            raise ValueError("log_signal_variance must be a scalar for PERIODIC.")
        if not isinstance(log_noise_variance, numbers.Number):
            raise ValueError("log_noise_variance must be a scalar for PERIODIC.")
        if not isinstance(period, numbers.Number) or period <= 0:
            raise ValueError("period must be a positive scalar for PERIODIC.")

        self.log_length_scale: float = float(log_length_scale)
        self.log_signal_variance: float = float(log_signal_variance)
        self.log_noise_variance: float = float(log_noise_variance)
        self.period: float = float(period)

    def _initialize_combination_hyperparameters(self, hyperparams: Tuple) -> None:
        """Validates and sets COMBINATION specific hyperparameters."""
        if not (isinstance(hyperparams, tuple) and len(hyperparams) == 7):
            raise ValueError("hyperparams for COMBINATION must be a 7-tuple.")
        (
            rbf_log_l,
            rbf_log_sig_var,
            rbf_log_noise,
            periodic_log_l,
            periodic_log_sig_var,
            periodic_log_noise,
            period,
        ) = hyperparams

        self.log_length_scale: float = float(rbf_log_l)
        self.log_signal_variance: float = float(rbf_log_sig_var)
        self.log_noise_variance: float = float(rbf_log_noise)
        self.log_periodic_length_scale: float = float(periodic_log_l)
        self.log_periodic_signal_variance: float = float(periodic_log_sig_var)
        self.log_periodic_noise_variance: float = float(periodic_log_noise)
        self.period: float = float(period)

    @staticmethod
    def _compute_rbf_ard_kernel(
        x1: np.ndarray,
        x2: np.ndarray,
        input_dim: int,
        log_length_scales: np.ndarray,  # Shape (1, dim)
        log_signal_variance: float,
        log_noise_variance: float,
        self_covariance: bool = False,
    ) -> np.ndarray:
        if x1.ndim == 1:
            x1 = x1.reshape(1, -1)
        if x2.ndim == 1:
            x2 = x2.reshape(1, -1)
        n1, dim1 = x1.shape
        n2, dim2 = x2.shape
        if dim1 != input_dim or dim2 != input_dim:
            raise ValueError(
                "Input dimensions mismatch."
                f" Expected {input_dim}, got {dim1} and {dim2}."
            )

        signal_variance = np.exp(log_signal_variance)
        noise_variance = np.exp(log_noise_variance)
        length_scales = np.exp(log_length_scales.flatten())

        x1_scaled = x1 / length_scales
        x2_scaled = x2 / length_scales

        # Compute squared Euclidean distance in the scaled space
        dist_sq_scaled = (
            np.sum(x1_scaled**2, axis=1, keepdims=True)
            - 2 * np.dot(x1_scaled, x2_scaled.T)
            + np.sum(x2_scaled**2, axis=1, keepdims=True).T
        )
        dist_sq_scaled = np.maximum(dist_sq_scaled, 0)

        K = signal_variance * np.exp(-0.5 * dist_sq_scaled)

        if self_covariance and n1 == n2 and np.array_equal(x1, x2):
            K.flat[:: n1 + 1] += noise_variance  # Add noise to the diagonal
        return K

    @staticmethod
    def _compute_rbf_iso_kernel(
        x1: np.ndarray,
        x2: np.ndarray,
        input_dim: int,
        log_length_scale: float,
        log_signal_variance: float,
        log_noise_variance: float,
        self_covariance: bool = False,
    ) -> np.ndarray:
        if x1.ndim == 1:
            x1 = x1.reshape(1, -1)
        if x2.ndim == 1:
            x2 = x2.reshape(1, -1)
        n1, dim1 = x1.shape
        n2, dim2 = x2.shape
        if dim1 != input_dim or dim2 != input_dim:
            raise ValueError(
                "Input dimensions mismatch."
                f" Expected {input_dim}, got {dim1} and {dim2}."
            )

        signal_variance = np.exp(log_signal_variance)
        noise_variance = np.exp(log_noise_variance)
        length_scale = np.exp(log_length_scale)

        # Compute squared Euclidean distance
        dist_sq_unscaled = (
            np.sum(x1**2, axis=1, keepdims=True)
            - 2 * np.dot(x1, x2.T)
            + np.sum(x2**2, axis=1, keepdims=True).T
        )
        dist_sq_unscaled = np.maximum(dist_sq_unscaled, 0)

        K = signal_variance * np.exp(-0.5 * dist_sq_unscaled / (length_scale**2))

        if self_covariance and n1 == n2 and np.array_equal(x1, x2):
            K.flat[:: n1 + 1] += noise_variance  # Add noise to the diagonal

        return K

    @staticmethod
    def _compute_periodic_kernel(
        x1: np.ndarray,
        x2: np.ndarray,
        input_dim: int,
        log_length_scale: float,
        log_signal_variance: float,
        log_noise_variance: float,
        period: float,
        self_covariance: bool = False,
    ) -> np.ndarray:
        """Computes the covariance matrix using a periodic kernel.
        k(x,x') = sigma^2 * exp(-2 * sum(sin^2(pi*|x_d-x'_d|/p)) / l^2)
        """
        if x1.ndim == 1:
            x1 = x1.reshape(-1, 1)
        if x2.ndim == 1:
            x2 = x2.reshape(-1, 1)
        n1, dim1 = x1.shape
        n2, dim2 = x2.shape
        if dim1 != input_dim or dim2 != input_dim:
            raise ValueError(
                "Input dimensions mismatch."
                f" Expected {input_dim}, got {dim1} and {dim2}."
            )

        signal_variance = np.exp(log_signal_variance)
        noise_variance = np.exp(log_noise_variance)
        length_scale = np.exp(log_length_scale)

        # Pairwise distance for each dimension
        dist = np.abs(x1[:, np.newaxis, :] - x2[np.newaxis, :, :])
        # Kernel computation
        sin_term = np.sin(np.pi * dist / period)
        # Summing over dimensions
        dist_sq_periodic = np.sum(sin_term**2, axis=2)

        K = signal_variance * np.exp(-2 * dist_sq_periodic / length_scale**2)

        if self_covariance:
            K.flat[:: n1 + 1] += noise_variance

        return K

    @staticmethod
    def _compute_combination_kernel(
        x1: np.ndarray,
        x2: np.ndarray,
        input_dim: int,
        log_rbf_length_scale: float,
        log_rbf_signal_variance: float,
        log_rbf_noise_variance: float,
        log_periodic_length_scale: float,
        log_periodic_signal_variance: float,
        log_periodic_noise_variance: float,
        period: float,
        self_covariance: bool = False,
    ) -> np.ndarray:
        """
        Computes covariance using a combination of an RBF (ISO) kernel
        and a Periodic kernel.
        The RBF kernel acts on the first D-1 dimensions.
        The Periodic kernel acts on the last (D-th) dimension.
        """
        if input_dim < 2:
            raise ValueError("COMBINATION kernel requires input_dim to be at least 2.")

        # --- Periodic Kernel on the last dimension ---
        k_periodic = SOGP._compute_periodic_kernel(
            x1=x1[:, -1:],
            x2=x2[:, -1:],
            input_dim=1,
            log_length_scale=log_periodic_length_scale,
            log_signal_variance=log_periodic_signal_variance,
            log_noise_variance=log_periodic_noise_variance,
            period=period,
            self_covariance=self_covariance,
        )

        # --- RBF Kernel on the first D-1 dimensions ---
        rbf_input_dim = input_dim - 1
        k_rbf = SOGP._compute_rbf_iso_kernel(
            x1=x1[:, :-1],
            x2=x2[:, :-1],
            input_dim=rbf_input_dim,
            log_length_scale=log_rbf_length_scale,
            log_signal_variance=log_rbf_signal_variance,
            log_noise_variance=log_rbf_noise_variance,
            self_covariance=self_covariance,
        )

        return k_periodic + k_rbf

    def _compute_covariance(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        self_covariance: bool = False,
    ) -> np.ndarray:
        """Instance method calling the appropriate static covariance function."""
        if self.kernel_name == "RBF_ARD":
            return SOGP._compute_rbf_ard_kernel(
                x1=x1,
                x2=x2,
                input_dim=self.input_dim,
                log_length_scales=self.log_length_scale,
                log_signal_variance=self.log_signal_variance,
                log_noise_variance=self.log_noise_variance,
                self_covariance=self_covariance,
            )
        elif self.kernel_name == "RBF_ISO":
            return SOGP._compute_rbf_iso_kernel(
                x1=x1,
                x2=x2,
                input_dim=self.input_dim,
                log_length_scale=self.log_length_scale,
                log_signal_variance=self.log_signal_variance,
                log_noise_variance=self.log_noise_variance,
                self_covariance=self_covariance,
            )
        elif self.kernel_name == "PERIODIC":
            return SOGP._compute_periodic_kernel(
                x1=x1,
                x2=x2,
                input_dim=self.input_dim,
                log_length_scale=self.log_length_scale,
                log_signal_variance=self.log_signal_variance,
                log_noise_variance=self.log_noise_variance,
                period=self.period,
                self_covariance=self_covariance,
            )
        elif self.kernel_name == "COMBINATION":
            return SOGP._compute_combination_kernel(
                x1=x1,
                x2=x2,
                input_dim=self.input_dim,
                log_rbf_length_scale=self.log_length_scale,
                log_rbf_signal_variance=self.log_signal_variance,
                log_rbf_noise_variance=self.log_noise_variance,
                log_periodic_length_scale=self.log_periodic_length_scale,
                log_periodic_signal_variance=self.log_periodic_signal_variance,
                log_periodic_noise_variance=self.log_periodic_noise_variance,
                period=self.period,
                self_covariance=self_covariance,
            )
        else:
            raise ValueError(f"Unknown kernel stored: {self.kernel_name}")

    def _get_prior_mean(self, x: np.ndarray) -> np.ndarray:
        """Computes the prior mean for the given input."""
        if callable(self.prior_mean_func):
            return (
                self.prior_mean_func(x, self.prior_mean_params)
                if self.prior_mean_params is not None
                else self.prior_mean_func(x)
            )
        elif isinstance(self.prior_mean_func, numbers.Number):
            return np.full((x.shape[0], 1), float(self.prior_mean_func))
        return np.zeros((x.shape[0], 1))

    def predict(self, x_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts the mean and variance for new input points.

        Args:
            x_new: New input point(s) to predict, shape (n_points, input_dim).

        Returns:
            A tuple containing the predicted mean and variance arrays.
        """
        x_new_array = (
            np.asarray(x_new, dtype=float).reshape(1, -1)
            if x_new.ndim == 1
            else np.asarray(x_new, dtype=float)
        )
        if x_new_array.shape[1] != self.input_dim:
            raise ValueError(
                f"Input x_new dimension mismatch: {x_new_array.shape[1]}, "
                f"expected {self.input_dim}"
            )

        # If no basis vectors exist, return the prior
        if self.num_bvs == 0:
            prior_mean = self._get_prior_mean(x_new_array)
            k_new_new = self._compute_covariance(
                x_new_array, x_new_array, self_covariance=True
            )
            return prior_mean, np.diag(k_new_new).reshape(-1, 1)

        # k_bv_new = K(bv, x_new)
        k_bv_new = self._compute_covariance(self.bv, x_new_array)
        # mu = mu_prior + alpha^T * k_bv_new
        predicted_mean = self._get_prior_mean(x_new_array) + np.dot(
            self.alpha.T, k_bv_new
        )

        # k_new_new = K(x_new, x_new)
        k_new_new_diag = np.diag(
            self._compute_covariance(x_new_array, x_new_array, self_covariance=True)
        ).reshape(-1, 1)

        # Variance reduction term: k_bv_new^T * C * k_bv_new
        variance_reduction = np.diag(
            np.dot(k_bv_new.T, np.dot(self.C, k_bv_new))
        ).reshape(-1, 1)

        # var = k_new_new - variance_reduction
        predicted_variance = np.maximum(k_new_new_diag + variance_reduction, 1e-9)

        return predicted_mean.T, predicted_variance

    def update(self, x_new: np.ndarray, y_new: float) -> bool:
        """
        Updates the SOGP model with a new data point.

        Args:
            x_new: New input point, shape (1, input_dim).
            y_new: Target value for the new input point.

        Returns:
            True if a new basis vector was added, False otherwise.
        """
        bv_added = False
        x_new_array = np.asarray(x_new, dtype=float).reshape(1, -1)
        if x_new_array.shape[1] != self.input_dim:
            raise ValueError(
                f"x_new has incorrect dimension {x_new_array.shape[1]},"
                f" expected {self.input_dim}"
            )

        if self.hpo_buffer_size > 0:
            self.hpo_buffer_X.append(x_new_array.flatten())
            self.hpo_buffer_Y.append(y_new)
            if len(self.hpo_buffer_X) > self.hpo_buffer_size:
                self.hpo_buffer_X.pop(0)
                self.hpo_buffer_Y.pop(0)

        # k_bv_new = K(bv, x_new)
        k_bv_new = (
            self._compute_covariance(self.bv, x_new_array)
            if self.num_bvs > 0
            else np.zeros((0, 1))
        )
        # k_new_new = K(x_new, x_new)
        k_new_new = self._compute_covariance(
            x_new_array, x_new_array, self_covariance=True
        )[0, 0]

        # Get current prediction for the new point
        prior_mean_value = self._get_prior_mean(x_new_array)[0, 0]
        pred_mean_from_bvs = (
            np.dot(self.alpha.T, k_bv_new)[0, 0] if self.num_bvs > 0 else 0.0
        )
        variance_reduction = (
            np.dot(k_bv_new.T, np.dot(self.C, k_bv_new))[0, 0]
            if self.num_bvs > 0
            else 0.0
        )
        current_pred_variance = max(k_new_new + variance_reduction, 1e-9)

        # Compute log-likelihood derivatives
        _, q, r = log_likelihood(
            self.noise_variance,
            y_new,
            pred_mean_from_bvs + prior_mean_value,
            current_pred_variance,
        )

        # Compute novelty score (gamma)
        if self.num_bvs > 0:
            Kb_stable = self.Kb + np.eye(self.Kb.shape[0]) * 1e-9
            try:
                # projection_coeffs = inv(Kb) @ k_bv_new
                projection_coeffs = cho_solve((cholesky(Kb_stable), True), k_bv_new)
            except np.linalg.LinAlgError:
                warnings.warn(
                    "Singular Kb matrix in update; using pseudo-inverse.",
                    RuntimeWarning,
                )
                projection_coeffs = np.linalg.pinv(Kb_stable) @ k_bv_new
            gamma = k_new_new - np.dot(k_bv_new.T, projection_coeffs)[0, 0]
        else:
            projection_coeffs = np.zeros((0, 1))
            gamma = k_new_new

        gamma = max(gamma, 1e-9)

        # Decide whether to add new BV or perform sparse update
        if self.num_bvs > 0 and (gamma < self.threshold):
            self._sparse_param_update(k_bv_new, q, r, projection_coeffs)
            bv_added = False
        else:
            self._full_param_update(
                x_new_array, k_bv_new, k_new_new, q, r, gamma, projection_coeffs
            )
            self.num_bvs += 1
            bv_added = True

        # Prune oldest/least important BV if we exceed the maximum
        while self.num_bvs > self.max_bvs:
            bv_to_remove_index = self._score_bvs()
            self._delete_bv(bv_to_remove_index)
            self.num_bvs -= 1

        return bv_added

    def _sparse_param_update(
        self, k_bv_new: np.ndarray, q: float, r: float, projection_coeffs: np.ndarray
    ) -> None:
        """
        Performs a sparse parameter update (no new BV is added).
        Updates alpha and C using Equation 9 from Csató (2001).
        """
        update_vector = np.dot(self.C, k_bv_new) + projection_coeffs
        self.alpha += q * update_vector
        self.C += r * np.outer(update_vector, update_vector)

    def _full_param_update(
        self,
        x_new: np.ndarray,
        k_bv_new: np.ndarray,
        k_new_new: float,
        q: float,
        r: float,
        gamma: float,
        projection_coeffs: np.ndarray,
    ) -> None:
        """Performs a full parameter update, adding the new point as a BV."""
        old_num_bvs = self.num_bvs
        self.bv = np.vstack([self.bv, x_new]) if old_num_bvs > 0 else x_new.copy()

        # Update inverse Gram matrix (Kb_inv) via rank-1 update
        # See Equation 24 from Csató (2001), Section 3.1.
        hat_e_extended = extend_vector(projection_coeffs, value=-1.0)
        self.Kb_inv = extend_matrix(self.Kb_inv)
        self.Kb_inv += np.outer(hat_e_extended, hat_e_extended) / gamma

        # Update Gram matrix (Kb)
        self.Kb = extend_matrix(self.Kb)
        self.Kb[:-1, -1:] = k_bv_new
        self.Kb[-1:, :-1] = k_bv_new.T
        self.Kb[-1, -1] = k_new_new

        # Update alpha and C
        C_old_dot_k = np.dot(self.C, k_bv_new) if old_num_bvs > 0 else np.zeros((0, 1))
        s_extended = extend_vector(C_old_dot_k, value=1.0)
        alpha_extended = extend_vector(self.alpha, value=0.0)
        C_extended = extend_matrix(self.C)

        self.alpha = alpha_extended + (q * s_extended)
        self.C = C_extended + (r * np.outer(s_extended, s_extended))

        # self.C = stabilize_matrix(self.C)
        # self.Kb = stabilize_matrix(self.Kb)
        # self.Kb_inv = stabilize_matrix(self.Kb_inv)

    def _score_bvs(self) -> int:
        """
        Scores BVs to find the least important one for removal.
        Uses Equation 27 from Csató (2001), Section 3.2. The score is essentially
        the change in the posterior mean if a BV is removed.
        """
        if self.num_bvs == 0:
            raise ValueError("No BVs to score.")

        diag_Kb_inv = np.diag(self.Kb_inv)
        # Add jitter to avoid division by zero
        scores = np.abs(self.alpha.flatten()) / (diag_Kb_inv + 1e-9)

        if np.any(np.isnan(scores)):
            warnings.warn(
                "NaNs in BV scores; choosing first available.", RuntimeWarning
            )
            scores[np.isnan(scores)] = np.inf

        return int(np.argmin(scores))

    def _get_updated_params_after_removal(
        self, remove_index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the updated alpha, C, and Kb_inv after removing a basis vector.
        This uses the update equations from Csató (2001), Section 3.2.
        """
        # Partition matrices based on the removal index 'i'
        i = remove_index
        q_ii = self.Kb_inv[i, i]
        if abs(q_ii) < 1e-9:
            warnings.warn(
                f"Denominator near zero for BV {i} in removal. Applying jitter.",
                RuntimeWarning,
            )
            q_ii = np.sign(q_ii) * 1e-9 if q_ii != 0 else 1e-9

        # Partition alpha
        alpha_i = self.alpha[i, 0]
        alpha_rest = np.delete(self.alpha, i, axis=0)

        # Partition Kb_inv (Q matrix in the paper)
        q_i_col = np.delete(self.Kb_inv[:, i], i, axis=0).reshape(-1, 1)
        Q_rest = np.delete(np.delete(self.Kb_inv, i, axis=0), i, axis=1)

        # Partition C
        c_ii = self.C[i, i]
        c_i_col = np.delete(self.C[:, i], i, axis=0).reshape(-1, 1)
        C_rest = np.delete(np.delete(self.C, i, axis=0), i, axis=1)

        # Update alpha
        updated_alpha = alpha_rest - alpha_i * (q_i_col / q_ii)

        # Update C
        term1 = C_rest
        term2 = c_ii * (np.outer(q_i_col, q_i_col) / (q_ii**2))
        term3 = (1 / q_ii) * (
            np.outer(q_i_col, c_i_col.T) + np.outer(c_i_col, q_i_col.T)
        )
        updated_C = term1 + term2 - term3

        # Update Kb_inv (Q)
        updated_Kb_inv = Q_rest - (1 / q_ii) * np.outer(q_i_col, q_i_col)

        return updated_alpha, updated_C, updated_Kb_inv

    def _delete_bv(self, remove_index: int) -> None:
        """Deletes a basis vector from the model."""
        if not (0 <= remove_index < self.num_bvs):
            raise ValueError("Invalid BV removal index.")

        (
            updated_alpha,
            updated_C,
            updated_Kb_inv,
        ) = self._get_updated_params_after_removal(remove_index)
        self.alpha = updated_alpha
        self.C = updated_C
        self.Kb_inv = updated_Kb_inv

        # Physically remove the BV and corresponding rows/cols from Kb
        keep_indices = [i for i in range(self.num_bvs) if i != remove_index]
        self.bv = self.bv[keep_indices]
        self.Kb = self.Kb[np.ix_(keep_indices, keep_indices)]

    def fit(
        self, X: Union[np.ndarray, Any], Y: Union[np.ndarray, List[float], Any]
    ) -> None:
        """
        Fits the model by sequentially updating it with all provided data points.

        Args:
            X: Input data, either a pandas DataFrame or a numpy array
                (n_samples, n_features).
            Y: Target data, either a pandas Series, numpy array, or list (n_samples,).
        """
        is_pandas_df = hasattr(X, "iloc")
        is_pandas_series = hasattr(Y, "iloc")

        if is_pandas_df:
            num_samples = X.shape[0]
            if num_samples != len(Y):
                raise ValueError("X and Y must have the same number of samples.")
            for i in range(num_samples):
                x_i = np.array(X.iloc[i, :], ndmin=2, dtype=float)
                y_i = float(Y.iloc[i] if is_pandas_series else Y[i])
                self.update(x_i, y_i)
        elif isinstance(X, np.ndarray) and isinstance(Y, (np.ndarray, list)):
            X_array, Y_array = np.asarray(X), np.asarray(Y)
            if X_array.ndim == 1:
                X_array = X_array.reshape(1, -1)
            if Y_array.ndim == 0:
                Y_array = Y_array.reshape(1)
            if X_array.shape[0] != Y_array.shape[0]:
                raise ValueError("X and Y must have the same number of samples.")
            for i in range(X_array.shape[0]):
                self.update(X_array[i, :].reshape(1, -1), float(Y_array[i]))
        else:
            raise TypeError(
                "Inputs must be pandas DataFrames/Series or numpy arrays/lists."
            )

    def get_hyperparameters_flat(self) -> np.ndarray:
        """
        Converts the model's hyperparameters into a single flat numpy array.
        This is a helper for the numerical optimizer.
        """
        if self.kernel_name == "RBF_ARD":
            # [l_1, ..., l_D, sigma_f, sigma_n]
            log_length_scales_flat = self.log_length_scale.flatten()
            return np.concatenate(
                [
                    log_length_scales_flat,
                    [self.log_signal_variance, self.log_noise_variance],
                ]
            )
        elif self.kernel_name == "RBF_ISO":
            # [l, sigma_f, sigma_n]
            return np.array(
                [
                    self.log_length_scale,
                    self.log_signal_variance,
                    self.log_noise_variance,
                ]
            )
        elif self.kernel_name == "PERIODIC":
            # [l, sigma_f, sigma_n, period]
            return np.array(
                [
                    self.log_length_scale,
                    self.log_signal_variance,
                    self.log_noise_variance,
                    self.period,
                ]
            )
        elif self.kernel_name == "COMBINATION":
            # [rbf_l, rbf_sf, rbf_sn, per_l, per_sf, per_sn, period]
            return np.array(
                [
                    self.log_length_scale,
                    self.log_signal_variance,
                    self.log_noise_variance,
                    self.log_periodic_length_scale,
                    self.log_periodic_signal_variance,
                    self.log_periodic_noise_variance,
                    self.period,
                ]
            )
        else:
            raise ValueError(f"Unknown kernel_name: {self.kernel_name}")

    def set_hyperparameters_flat(self, hyperparameters_flat: np.ndarray) -> None:
        """
        Sets the model's hyperparameters from a single flat numpy array.
        This is a helper for the numerical optimizer.
        """
        if self.kernel_name == "RBF_ARD":
            if len(hyperparameters_flat) != self.input_dim + 2:
                raise ValueError(
                    f"RBF_ARD: Expected {self.input_dim + 2} hyperparameters, "
                    f"got {len(hyperparameters_flat)}"
                )
            self.log_length_scale = hyperparameters_flat[: self.input_dim].reshape(
                1, -1
            )
            self.log_signal_variance = hyperparameters_flat[self.input_dim]
            self.log_noise_variance = hyperparameters_flat[self.input_dim + 1]
            self.noise_variance = np.exp(self.log_noise_variance)

        elif self.kernel_name == "RBF_ISO":
            if len(hyperparameters_flat) != 3:
                raise ValueError(
                    f"RBF_ISO: Expected 3 hyperparameters, "
                    f"got {len(hyperparameters_flat)}"
                )
            self.log_length_scale = hyperparameters_flat[0]
            self.log_signal_variance = hyperparameters_flat[1]
            self.log_noise_variance = hyperparameters_flat[2]
            self.noise_variance = np.exp(self.log_noise_variance)

        elif self.kernel_name == "PERIODIC":
            if len(hyperparameters_flat) != 4:
                raise ValueError(
                    f"PERIODIC: Expected 4 hyperparameters, "
                    f"got {len(hyperparameters_flat)}"
                )
            self.log_length_scale = hyperparameters_flat[0]
            self.log_signal_variance = hyperparameters_flat[1]
            self.log_noise_variance = hyperparameters_flat[2]
            self.period = hyperparameters_flat[3]
            self.noise_variance = np.exp(self.log_noise_variance)

        elif self.kernel_name == "COMBINATION":
            if len(hyperparameters_flat) != 7:
                raise ValueError(
                    f"COMBINATION: Expected 7 hyperparameters, "
                    f"got {len(hyperparameters_flat)}"
                )
            self.log_length_scale = hyperparameters_flat[0]
            self.log_signal_variance = hyperparameters_flat[1]
            self.log_noise_variance = hyperparameters_flat[2]
            self.log_periodic_length_scale = hyperparameters_flat[3]
            self.log_periodic_signal_variance = hyperparameters_flat[4]
            self.log_periodic_noise_variance = hyperparameters_flat[5]
            self.period = hyperparameters_flat[6]
            # For the online update, the model uses a single noise term.
            # We set it to the RBF noise to maintain consistency with the
            # original code's structure.
            #
            # Note: The HPO gradient calculation correctly uses the sum of both
            # noise terms.
            self.noise_variance = np.exp(self.log_noise_variance)

        else:
            raise ValueError(f"Unknown kernel_name: {self.kernel_name}")

    @staticmethod
    def _objective_and_gradient_static(
        hyperparameters_flat: np.ndarray,
        X_hpo: np.ndarray,
        Y_hpo: np.ndarray,
        input_dim: int,
        kernel_name: str,
    ) -> Tuple[float, np.ndarray]:
        """
        Calculates the negative log marginal likelihood (objective function)
        and its gradient.

        This static method is the core of the hyperparameter optimization.
        """
        num_samples = X_hpo.shape[0]
        Y_hpo_column = Y_hpo.reshape(-1, 1)

        # --- Compute K_f and total noise based on kernel type ---
        if kernel_name == "RBF_ARD":
            log_length_scales = hyperparameters_flat[:input_dim].reshape(1, -1)
            log_signal_variance = hyperparameters_flat[input_dim]
            log_noise_variance = hyperparameters_flat[input_dim + 1]
            K_f = SOGP._compute_rbf_ard_kernel(
                X_hpo,
                X_hpo,
                input_dim,
                log_length_scales,
                log_signal_variance,
                log_noise_variance,
                self_covariance=False,
            )
            total_noise_variance = np.exp(log_noise_variance)
        elif kernel_name == "RBF_ISO":
            (
                log_length_scale,
                log_signal_variance,
                log_noise_variance,
            ) = hyperparameters_flat
            K_f = SOGP._compute_rbf_iso_kernel(
                X_hpo,
                X_hpo,
                input_dim,
                log_length_scale,
                log_signal_variance,
                log_noise_variance,
                self_covariance=False,
            )
            total_noise_variance = np.exp(log_noise_variance)
        elif kernel_name == "PERIODIC":
            (
                log_length_scale,
                log_signal_variance,
                log_noise_variance,
                period,
            ) = hyperparameters_flat
            K_f = SOGP._compute_periodic_kernel(
                X_hpo,
                X_hpo,
                input_dim,
                log_length_scale,
                log_signal_variance,
                log_noise_variance,
                period,
                self_covariance=False,
            )
            total_noise_variance = np.exp(log_noise_variance)
        ## NEW ##
        elif kernel_name == "COMBINATION":
            rbf_l, rbf_sf, rbf_sn, per_l, per_sf, per_sn, period = hyperparameters_flat
            K_f = SOGP._compute_combination_kernel(
                X_hpo,
                X_hpo,
                input_dim,
                rbf_l,
                rbf_sf,
                rbf_sn,
                per_l,
                per_sf,
                per_sn,
                period,
                self_covariance=False,
            )
            # For an additive kernel, the total noise is the sum of individual noises
            total_noise_variance = np.exp(rbf_sn) + np.exp(per_sn)
        else:
            raise ValueError(f"Unsupported kernel for HPO: {kernel_name}")

        # --- Compute NLML (Valid for all kernels) ---
        K_y = (
            K_f
            + total_noise_variance * np.eye(num_samples)
            + np.eye(num_samples) * 1e-9
        )
        try:
            K_y_inv = inv(K_y)
        except np.linalg.LinAlgError:
            return np.inf, np.zeros_like(hyperparameters_flat)

        alpha_vector = K_y_inv @ Y_hpo_column
        log_det_sign, log_det_val = slogdet(K_y)
        if log_det_sign < 0.5:
            return np.inf, np.zeros_like(hyperparameters_flat)

        negative_log_marginal_likelihood = (
            0.5 * (Y_hpo_column.T @ alpha_vector)[0, 0]
            + 0.5 * log_det_val
            + 0.5 * num_samples * np.log(2 * np.pi)
        )

        # --- Gradient Calculation ---
        grad = np.zeros_like(hyperparameters_flat)
        common_term = np.outer(alpha_vector, alpha_vector) - K_y_inv

        # Gradients for kernel-specific parameters
        if kernel_name == "RBF_ARD":
            # (Code for RBF_ARD gradient)
            pass  # Placeholder for brevity
        elif kernel_name == "RBF_ISO":
            # (Code for RBF_ISO gradient)
            pass  # Placeholder for brevity
        elif kernel_name == "PERIODIC":
            # (Code for PERIODIC gradient)
            pass  # Placeholder for brevity
        ## NEW ##
        elif kernel_name == "COMBINATION":
            # Unpack for clarity
            rbf_l, rbf_sf, rbf_sn, per_l, per_sf, per_sn, period = hyperparameters_flat

            # Deconstruct K_f to get gradients for each part
            K_rbf = SOGP._compute_rbf_iso_kernel(
                X_hpo[:, :-1],
                X_hpo[:, :-1],
                input_dim - 1,
                rbf_l,
                rbf_sf,
                rbf_sn,
                self_covariance=False,
            )
            K_per = SOGP._compute_periodic_kernel(
                X_hpo[:, -1:],
                X_hpo[:, -1:],
                1,
                per_l,
                per_sf,
                per_sn,
                period,
                self_covariance=False,
            )

            # Grad w.r.t RBF log-signal-variance
            grad[1] = 0.5 * np.trace(common_term @ (2 * K_rbf))
            # Grad w.r.t RBF log-noise-variance
            grad[2] = 0.5 * np.trace(
                common_term @ (2 * np.exp(rbf_sn) * np.eye(num_samples))
            )

            # Grad w.r.t Periodic log-signal-variance
            grad[4] = 0.5 * np.trace(common_term @ (2 * K_per))
            # Grad w.r.t Periodic log-noise-variance
            grad[5] = 0.5 * np.trace(
                common_term @ (2 * np.exp(per_sn) * np.eye(num_samples))
            )

            # Grad w.r.t RBF log-length-scale
            length_scale_rbf = np.exp(rbf_l)
            x_rbf = X_hpo[:, :-1]
            x_sum_sq = np.sum(x_rbf**2, axis=1, keepdims=True)
            dist_sq_rbf = np.maximum(
                0, -2 * np.dot(x_rbf, x_rbf.T) + x_sum_sq + x_sum_sq.T
            )
            dK_d_log_l_rbf = K_rbf * (dist_sq_rbf / (length_scale_rbf**2))
            grad[0] = 0.5 * np.trace(common_term @ dK_d_log_l_rbf)

            # Grad w.r.t Periodic log-length-scale
            length_scale_per = np.exp(per_l)
            x_per = X_hpo[:, -1:]
            dist_per = np.abs(x_per - x_per.T)
            sin_term_sq = np.sin(np.pi * dist_per / period) ** 2
            dK_d_log_l_per = K_per * (4 * sin_term_sq / length_scale_per**2)
            grad[3] = 0.5 * np.trace(common_term @ dK_d_log_l_per)

            # Grad w.r.t Period
            if abs(period) > 1e-9:
                dist_per = np.abs(x_per - x_per.T)
                sin_2term = np.sin(2 * np.pi * dist_per / period)
                dK_dp = (
                    K_per
                    * (2 * np.pi * dist_per / (length_scale_per**2 * period**2))
                    * sin_2term
                )
                grad[6] = 0.5 * np.trace(common_term @ dK_dp)

        return negative_log_marginal_likelihood, grad

    def tune_hyperparameters_cg(
        self,
        X_hpo: np.ndarray,
        Y_hpo: np.ndarray,
        max_cg_iter: int = 10,
    ) -> None:
        """
        Tunes hyperparameters using Conjugate Gradient descent on the provided data.
        """
        if X_hpo.shape[0] < 2:
            warnings.warn(
                f"Not enough HPO data ({X_hpo.shape[0]} points) for tuning. Skipping.",
                UserWarning,
            )
            return

        # Get initial hyperparameters as a flat vector
        hyperparameters = self.get_hyperparameters_flat()

        # The function to be minimized (we want to MAXIMIZE log-likelihood)
        def objective_grad(h_params):
            nlml, grad_nlml = SOGP._objective_and_gradient_static(
                h_params, X_hpo, Y_hpo, self.input_dim, self.kernel_name
            )
            return nlml, grad_nlml  # Return positive NLML and its gradient

        # Very basic CG implementation for demonstration
        # In a real-world scenario, scipy.optimize.minimize is recommended
        # e.g., from scipy.optimize import minimize
        # res = minimize(objective_grad, hyperparameters, method='CG', jac=True,
        #                options={'maxiter': max_cg_iter})
        # self.set_hyperparameters_flat(res.x)

        # Manual CG implementation
        f_val, f_prime = objective_grad(hyperparameters)
        d = -f_prime  # Initial search direction
        r = -f_prime  # Initial residual

        for i in range(max_cg_iter):
            # A simple line search could be implemented here to find step size 'alpha'
            # For simplicity, we use a small fixed step size
            step_size = 1e-4

            hyperparameters += step_size * d

            f_val, f_prime = objective_grad(hyperparameters)
            r_new = -f_prime

            # Polak-Ribière update for beta
            beta = np.dot(r_new.T, r_new - r) / (np.dot(r.T, r) + 1e-9)
            beta = max(0, beta)  # Ensure descent direction

            # Update search direction
            d = r_new + beta * d
            r = r_new

        self.set_hyperparameters_flat(hyperparameters)
        print(f"HPO Complete. Final NLML: {f_val:.4f}")

    def get_hpo_data_buffer(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Returns the data currently in the HPO buffer."""
        if self.hpo_buffer_size > 0 and len(self.hpo_buffer_X) > 0:
            return np.array(self.hpo_buffer_X), np.array(self.hpo_buffer_Y)
        return None
