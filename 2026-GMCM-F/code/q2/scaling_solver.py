"""Bounded scaling-law solvers with analytic derivatives and variable projection.

The objective, physical parameter bounds and prediction family match v6.
No validation row is used to initialise a training fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares, lsq_linear


@dataclass(frozen=True)
class ScalingData:
    log_n: np.ndarray
    log_d: np.ndarray
    log_q: np.ndarray
    y: np.ndarray
    sqrt_weight: np.ndarray

    @classmethod
    def from_frame(cls, frame, *, quality=False, prediction=False):
        n, d = (frame[key].to_numpy(float) for key in ("N_params_B", "D_tokens_B"))
        y = np.zeros(len(n)) if prediction else frame["val_loss"].to_numpy(float)
        q = frame["Q_score"].to_numpy(float) if quality else np.ones(len(y))
        if not len(y) or not all(np.all(np.isfinite(v)) for v in (n, d, q, y)):
            raise ValueError("Scaling data must be nonempty and finite")
        if any(np.any(v <= 0) for v in (n, d, q)):
            raise ValueError("N, D and Q must be positive")
        return cls(np.log(n), np.log(d), np.log(q), y, np.ones(len(y)))

    def take(self, index):
        return ScalingData(*(getattr(self, key)[index] for key in self.__dataclass_fields__))

    def weighted(self, count):
        """Frequency weights are exactly equivalent to repeated bootstrap rows."""
        count = np.asarray(count, dtype=float)
        if count.shape != self.y.shape or np.any(count < 0) or not np.all(np.isfinite(count)):
            raise ValueError("Invalid frequency weights")
        keep = count > 0
        if not keep.any():
            raise ValueError("At least one row must have positive weight")
        return ScalingData(self.log_n[keep], self.log_d[keep], self.log_q[keep],
                           self.y[keep], self.sqrt_weight[keep] * np.sqrt(count[keep]))


@dataclass
class ScalingFit:
    theta: np.ndarray
    objective: float
    optimality: float
    nfev: int
    evaluations: int
    elapsed_seconds: float
    method: str
    status: int
    active_bounds: list[str]

    def record(self):
        return dict(theta=self.theta.tolist(), objective=self.objective,
                    optimality=self.optimality, nfev=self.nfev,
                    evaluations=self.evaluations, elapsed_seconds=self.elapsed_seconds,
                    method=self.method, status=self.status, active_bounds=self.active_bounds)


def prediction_jacobian(theta, data):
    """Full unweighted prediction and exact Jacobian in physical parameters."""
    e, a, b, alpha, beta = theta[:5]
    gamma = theta[5] if len(theta) == 6 else 0.0
    u = np.exp(-alpha * data.log_n)
    v = np.exp(-beta * (data.log_d + gamma * data.log_q))
    columns = [np.ones(len(u)), u, v, -a * data.log_n * u,
               -b * (data.log_d + gamma * data.log_q) * v]
    if len(theta) == 6:
        columns.append(-b * beta * data.log_q * v)
    return e + a * u + b * v, np.column_stack(columns)


def predict(theta, data, psi=0.0, mix_lambda=0.0):
    e, a, b, alpha, beta = theta[:5]
    gamma = theta[5] if len(theta) == 6 else 0.0
    exponent = -beta * (data.log_d + gamma * data.log_q - mix_lambda * np.asarray(psi))
    return e + a * np.exp(-alpha * data.log_n) + b * np.exp(exponent)


def _bounded_linear(matrix, target):
    lower, upper = np.array([0., 1e-8, 1e-8]), np.array([10., 100., 100.])
    coef = np.linalg.lstsq(matrix, target, rcond=None)[0]
    if np.all(coef >= lower) and np.all(coef <= upper):
        return coef, np.zeros(3, dtype=int)
    result = lsq_linear(matrix, target, bounds=(lower, upper), method="bvls", tol=1e-12)
    if not result.success:
        raise RuntimeError("Bounded linear coefficient fit failed")
    return result.x, result.active_mask


class ProfileResidual:
    """Exact variable-projection residual/Jacobian on each active-set region."""

    def __init__(self, data, template, nonlinear_indices):
        self.data, self.template = data, template.copy()
        self.indices = nonlinear_indices
        self.last_z = None
        self.evaluations = 0

    def evaluate(self, z):
        if self.last_z is not None and np.array_equal(z, self.last_z):
            return self.residual, self.jacobian
        theta = self.template.copy()
        theta[self.indices] = z
        alpha, beta = theta[3:5]
        gamma = theta[5] if len(theta) == 6 else 0.
        u = np.exp(-alpha * self.data.log_n)
        v = np.exp(-beta * (self.data.log_d + gamma * self.data.log_q))
        matrix = np.column_stack([np.ones(len(u)), u, v]) * self.data.sqrt_weight[:, None]
        target = self.data.y * self.data.sqrt_weight
        coef, active = _bounded_linear(matrix, target)
        residual = matrix @ coef - target
        free = active == 0
        # Use an SVD-based pseudoinverse, avoiding squared conditioning from A.T A.
        inverse = np.linalg.pinv(matrix[:, free])
        free_matrix = matrix[:, free]
        derivatives = []
        for idx in self.indices:
            dm = np.zeros_like(matrix)
            if idx == 3:
                dm[:, 1] = -self.data.log_n * matrix[:, 1]
            elif idx == 4:
                dm[:, 2] = -(self.data.log_d + gamma * self.data.log_q) * matrix[:, 2]
            else:
                dm[:, 2] = -beta * self.data.log_q * matrix[:, 2]
            direct = dm @ coef
            derivative = direct - free_matrix @ (inverse @ direct)
            derivative -= inverse.T @ (dm[:, free].T @ residual)
            derivatives.append(derivative)
        self.last_z = np.array(z, copy=True)
        self.residual = residual
        self.jacobian = np.column_stack(derivatives) if derivatives else np.empty((len(u), 0))
        theta[:3] = coef
        self.theta, self.active = theta, active
        self.evaluations += 1
        return self.residual, self.jacobian


def fit_scaling(data, *, quality=False, method="analytic", initial=None,
                fixed_exponents=None, fixed_gamma=None, starts=None):
    """Fit the v6 bounded least-squares objective; reject failed optimisations."""
    if method not in {"analytic", "profile"}:
        raise ValueError("method must be analytic or profile")
    if fixed_gamma is not None and not quality:
        raise ValueError("fixed_gamma requires quality=True")
    begin = perf_counter()
    lower = np.array([0., 1e-8, 1e-8, .005, .005] + ([0.] if quality else []))
    upper = np.array([10., 100., 100., 1.5, 1.5] + ([5.] if quality else []))
    if starts is None:
        if quality:
            starts = [[.4, .5, 2., .28, .08, 1.]]
        else:
            starts = [[1., .5, 1., .3, .2], [1.5, 1., .5, .4, .25], [.5, 2., 2., .2, .1]]
    if initial is not None:
        initial = np.asarray(initial, dtype=float)
        if initial.size == len(lower):
            starts = [initial]
        elif fixed_exponents is not None and initial.size == (4 if quality else 3):
            starts = [np.r_[initial[:3], fixed_exponents, initial[3:]]]
        else:
            raise ValueError("Initial parameter vector has the wrong size")
    if fixed_exponents is not None and initial is None:
        starts = [starts[0]]
    fixed = ({3: fixed_exponents[0], 4: fixed_exponents[1]}
             if fixed_exponents is not None else {})
    if fixed_gamma is not None:
        fixed[5] = float(fixed_gamma)
    free = [idx for idx in range(len(lower)) if idx not in fixed]
    nonlinear = [idx for idx in free if idx >= 3]
    solutions = []
    total_evaluations, total_nfev = 0, 0
    for start in starts:
        template = np.asarray(start, dtype=float).copy()
        for idx, value in fixed.items():
            template[idx] = value
        if method == "profile" or not nonlinear:
            evaluator = ProfileResidual(data, template, nonlinear)
            if nonlinear:
                result = least_squares(lambda z: evaluator.evaluate(z)[0], template[nonlinear],
                    jac=lambda z: evaluator.evaluate(z)[1], bounds=(lower[nonlinear], upper[nonlinear]),
                    x_scale="jac", max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                evaluator.evaluate(result.x)
                if result.success:
                    solutions.append((float(np.dot(evaluator.residual, evaluator.residual)),
                                      evaluator.theta.copy(), result.optimality, result.status))
                total_nfev += result.nfev
            else:
                evaluator.evaluate(np.empty(0))
                solutions.append((float(np.dot(evaluator.residual, evaluator.residual)),
                                  evaluator.theta.copy(), 0., 1))
                total_nfev += 1
            total_evaluations += evaluator.evaluations
        else:
            state = dict(last=None, evaluations=0)

            def evaluate(z):
                if state["last"] is None or not np.array_equal(z, state["last"]):
                    theta = template.copy()
                    theta[free] = z
                    values, jac = prediction_jacobian(theta, data)
                    state.update(last=z.copy(), residual=(values-data.y)*data.sqrt_weight,
                                 jac=jac[:, free]*data.sqrt_weight[:, None],
                                 evaluations=state["evaluations"]+1)
                return state["residual"], state["jac"]

            result = least_squares(lambda z: evaluate(z)[0], template[free], jac=lambda z: evaluate(z)[1],
                bounds=(lower[free], upper[free]), max_nfev=40000,
                ftol=1e-12, xtol=1e-12, gtol=1e-12)
            theta = template.copy()
            theta[free] = result.x
            if result.success:
                solutions.append((float(2*result.cost), theta, result.optimality, result.status))
            total_nfev += result.nfev
            total_evaluations += state["evaluations"]
    if not solutions:
        raise RuntimeError("All scaling-law starts failed to converge")
    objective, theta, optimality, status = min(solutions, key=lambda s: s[0])
    names = ["E", "A", "B", "alpha", "beta", "gamma_Q"]
    active = [names[i] for i in free if np.isclose(theta[i], lower[i], atol=1e-7, rtol=0)
              or np.isclose(theta[i], upper[i], atol=1e-7, rtol=0)]
    return ScalingFit(theta, objective, float(optimality), total_nfev, total_evaluations,
                      perf_counter()-begin, method, int(status), active)
