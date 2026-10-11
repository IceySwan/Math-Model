"""Separable quality laws in direct loss-slope coordinates.

The curved family bounds the quality/token elasticity at both ends of the
declared support. All outer constraints are linear. Outer minimization is
nonconvex; multistart and independent checks do not prove global optimality.
"""
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.optimize import minimize, nnls

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scaling_solver import ScalingData, _bounded_linear

U_MIN = float(np.log(.1))
NAMES = ["E", "A", "B", "alpha", "beta", "delta", "delta_min"]


def shape_basis(data, model):
    u = data.log_q
    if model == "power":
        return np.column_stack([-u])
    if model != "curved":
        raise ValueError(model)
    return np.column_stack([-u + u*u/(2*U_MIN), -u*u/(2*U_MIN)])


def kappa(theta):
    return (theta[5]-theta[6])/(2*U_MIN) if len(theta) == 7 else 0.


def log_quality(theta, log_q):
    return -theta[5]*np.asarray(log_q) + kappa(theta)*np.asarray(log_q)**2


def quality_slope(theta, q):
    return theta[5] - 2*kappa(theta)*np.log(np.asarray(q))


def prediction_jacobian(theta, data):
    e, a, b, alpha, beta = theta[:5]
    model = "curved" if len(theta) == 7 else "power"
    h = shape_basis(data, model)
    un = np.exp(-alpha*data.log_n)
    vd = np.exp(-beta*data.log_d + h @ theta[5:])
    jac = np.column_stack([np.ones(len(un)), un, vd, -a*data.log_n*un,
                           -b*data.log_d*vd, b*vd[:, None]*h])
    return e+a*un+b*vd, jac


def predict(theta, data, psi=0., mix_lambda=0.):
    values = prediction_jacobian(theta, data)[0]
    if np.any(np.asarray(psi) != 0) and mix_lambda != 0:
        term = theta[2]*np.exp(-theta[4]*data.log_d + log_quality(theta, data.log_q))
        values = values + term*np.expm1(theta[4]*mix_lambda*np.asarray(psi))
    return values


class ProjectedResidual:
    def __init__(self, data, model):
        self.data, self.model = data, model
        self.h = shape_basis(data, model)
        self.last = None
        self.evaluations = 0

    def evaluate(self, z):
        if self.last is not None and np.array_equal(z, self.last):
            return self.residual, self.jac
        d = self.data
        un = np.exp(-z[0]*d.log_n)
        vd = np.exp(-z[1]*d.log_d + self.h @ z[2:])
        matrix = np.column_stack([np.ones(len(un)), un, vd])*d.sqrt_weight[:, None]
        target = d.y*d.sqrt_weight
        coef, active = _bounded_linear(matrix, target)
        residual = matrix @ coef - target
        free = active == 0
        mf = matrix[:, free]
        inv = np.linalg.pinv(mf)
        jac = []
        for j in range(len(z)):
            dm = np.zeros_like(matrix)
            if j == 0:
                dm[:, 1] = -d.log_n*matrix[:, 1]
            elif j == 1:
                dm[:, 2] = -d.log_d*matrix[:, 2]
            else:
                dm[:, 2] = self.h[:, j-2]*matrix[:, 2]
            direct = dm @ coef
            jac.append(direct - mf @ (inv @ direct) - inv.T @ (dm[:, free].T @ residual))
        self.residual, self.jac = residual, np.column_stack(jac)
        self.theta = np.r_[coef, z]
        self.linear_active = active
        self.last = z.copy()
        self.evaluations += 1
        return self.residual, self.jac


@dataclass
class Fit:
    theta: np.ndarray
    objective: float
    kkt_residual: float
    elapsed_seconds: float
    evaluations: int
    successful_starts: int
    start_objective_range: float
    min_constraint_margin: float

    def record(self):
        return {**dict(zip(NAMES, map(float, self.theta))), "kappa": float(kappa(self.theta)),
                **{k: v for k,v in vars(self).items() if k != "theta"}}


def fit(data, model, *, warm=None, fixed=None, starts=None):
    begin = perf_counter()
    size = 3 if model == "power" else 4
    lower = np.array([.005, .005]+[0.]*(size-2))
    upper = np.array([1.5, 1.5]+[7.5]*(size-2))
    cmat = np.zeros((size-2, size))
    cmat[:, 1] = 5.
    for j in range(size-2):
        cmat[j, j+2] = -1.
    fixed = fixed or {}
    free = np.array([j for j in range(size) if j not in fixed], dtype=int)
    if starts is None:
        # Preset starts contain no held-out information.
        starts = [np.array([.28,.08,.08]), np.array([.4,.25,.30]), np.array([.2,.05,.12])]
        if model == "curved":
            starts = [np.r_[s,s[2]] for s in starts]
            starts += [np.array([.28,.08,.14,.03]), np.array([.28,.08,.03,.14])]
    if warm is not None:
        w = np.asarray(warm)[3:]
        if len(w) == 3 and size == 4:
            w = np.r_[w,w[2]]
        starts = [w, *starts]
    solutions, evaluations = [], 0
    failures = []
    for raw in starts:
        z = np.clip(np.array(raw, dtype=float), lower, upper)
        for j,value in fixed.items():
            z[j] = value
        # Profiles in delta need beta sufficiently large; reject impossible profiles.
        if 1 not in fixed:
            z[1] = max(z[1], max(z[2:])/5)
        for j in range(2,size):
            if j not in fixed:
                z[j] = min(z[j], 5*z[1])
        if np.min(cmat@z) < -1e-10:
            continue
        evaluator = ProjectedResidual(data, model)
        def full(x):
            zz = z.copy()
            zz[free] = x
            return zz
        def fun(x):
            r,_ = evaluator.evaluate(full(x))
            return .5*float(r@r)
        def jac(x):
            r,j = evaluator.evaluate(full(x))
            return (j.T@r)[free]
        result = minimize(fun, z[free], jac=jac, method="SLSQP",
            bounds=list(zip(lower[free],upper[free])),
            constraints={"type":"ineq", "fun":lambda x: cmat@full(x),
                         "jac":lambda x: cmat[:, free]},
            options={"ftol":1e-13,"maxiter":800})
        zz = full(result.x)
        r,j = evaluator.evaluate(zz)
        margin = min(float(np.min(zz-lower)), float(np.min(upper-zz)),float(np.min(cmat@zz)))
        if result.success and margin >= -1e-8:
            normals = [cmat[k,free] for k in range(len(cmat)) if cmat[k]@zz < 1e-7]
            for k,idx in enumerate(free):
                if zz[idx]-lower[idx] < 1e-7:
                    normals.append(np.eye(len(free))[k])
                if upper[idx]-zz[idx] < 1e-7:
                    normals.append(-np.eye(len(free))[k])
            grad = (j.T@r)[free]
            if normals:
                mat = np.array(normals).T
                multipliers = nnls(mat,grad)[0]
                grad = grad-mat@multipliers
            solutions.append((float(r@r), evaluator.theta.copy(), float(np.linalg.norm(grad,np.inf)), margin))
        else:
            failures.append(str(result.message))
        evaluations += evaluator.evaluations
    if not solutions:
        raise RuntimeError(f"All {model} fits failed: {failures}; fixed={fixed}")
    objective,theta,kkt,margin = min(solutions,key=lambda r:r[0])
    return Fit(theta,objective,kkt,perf_counter()-begin,evaluations,len(solutions),
               max(s[0] for s in solutions)-objective,margin)


def parameter_dict(theta):
    result = dict(zip(NAMES,map(float,theta)))
    result["delta_min"] = float(theta[6] if len(theta)==7 else theta[5])
    result["kappa"] = float(kappa(theta))
    result["gamma_at_Q1"] = float(theta[5]/theta[4])
    result["gamma_at_Qmin"] = float(result["delta_min"]/theta[4])
    return result


