"""Budget optimization with analytic isolation of every quality branch.

All resources are in billions, budget in total FLOPs. No I/O or fitting here.
The three supported cost laws admit finite derivative partitions; no quality
grid is used to discover stationary points. See reports/ALGORITHM_OPTIMIZATION.md.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Callable

import numpy as np
from scipy.optimize import brentq

ETA = 2e-4
FLOPS_SCALE = 1e18
COST_LABELS = {"exponential": "指数型", "power": "幂函数型", "logarithmic": "对数渐进型"}


@dataclass(frozen=True)
class Parameters:
    E: float
    A: float
    B: float
    alpha: float
    beta: float
    gamma: float

    def __post_init__(self):
        values = asdict(self)
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("Parameters must be finite")
        if self.E < 0 or self.gamma < 0 or min(self.A, self.B, self.alpha, self.beta) <= 0:
            raise ValueError("Require E,gamma >= 0 and A,B,alpha,beta > 0")

    @classmethod
    def from_mapping(cls, data):
        return cls(**{k: float(data['gamma_Q' if k == 'gamma' and 'gamma_Q' in data else k])
                      for k in cls.__dataclass_fields__})


def unique(values, tol=2e-13):
    out = []
    for x in sorted(float(x) for x in values):
        if not out or abs(x - out[-1]) > tol:
            out.append(x)
    return out


def monotone_roots(f: Callable[[float], float], lo: float, hi: float, turns=(), tol=1e-12):
    """Root isolation on certified monotone intervals, including touching roots."""
    points = unique([lo, hi] + [x for x in turns if lo < x < hi])
    vals = [float(f(x)) for x in points]
    roots = [x for x, y in zip(points, vals) if abs(y) <= tol]
    for a, b, fa, fb in zip(points[:-1], points[1:], vals[:-1], vals[1:]):
        if (fa < 0 < fb) or (fb < 0 < fa):
            roots.append(brentq(f, a, b, xtol=5e-15, rtol=1e-14))
    return unique(roots)


def real_polynomial_roots(coefficients, lo, hi):
    c = np.trim_zeros(np.asarray(coefficients, dtype=float), 'f')
    if len(c) <= 1:
        return []
    return unique([float(x.real) for x in np.roots(c)
                   if abs(x.imag) < 1e-10 and lo < x.real < hi])


@dataclass(frozen=True)
class QualityCost:
    kind: str
    q0: float
    scale: float = 1.0

    def __post_init__(self):
        if self.kind not in COST_LABELS:
            raise ValueError(f"Unsupported cost law: {self.kind}")
        if not (math.isfinite(self.q0) and 0 < self.q0 <= 1):
            raise ValueError("Q0 must be in (0,1]")
        if not (math.isfinite(self.scale) and self.scale >= 0):
            raise ValueError("Cost scale must be finite and nonnegative")

    def h(self, q):
        # expm1/log1p and polynomial factorization avoid cancellation near Q0.
        if self.kind == 'exponential':
            return self.scale * .01 * math.exp(6*self.q0) * math.expm1(6*(q-self.q0))
        if self.kind == 'power':
            return self.scale * 5*(q-self.q0)*(q+self.q0)*(q*q+self.q0*self.q0)
        return self.scale * 2*math.log1p(10*(q-self.q0)/(1+10*self.q0))

    def hp(self, q):
        if self.kind == 'exponential':
            return self.scale * .06*math.exp(6*q)
        if self.kind == 'power':
            return self.scale * 20*q**3
        return self.scale * 20/(1+10*q)

    def hpp(self, q):
        if self.kind == 'exponential':
            return self.scale * .36*math.exp(6*q)
        if self.kind == 'power':
            return self.scale * 60*q*q
        return -self.scale * 200/(1+10*q)**2


class Solver:
    def __init__(self, params: Parameters, context: int, kind: str, q0: float,
                 *, psi=0.0, lambda_p=0.0, cost_scale=1.0):
        if not math.isfinite(context) or context <= 0 or int(context) != context:
            raise ValueError("Context must be a positive integer")
        if not (math.isfinite(psi) and math.isfinite(lambda_p) and lambda_p >= 0):
            raise ValueError("Invalid mixture scenario")
        self.p, self.context = params, int(context)
        self.cost = QualityCost(kind, q0, cost_scale)
        self.k = 6 + ETA*context
        self.log_k = math.log(self.k)
        self.log_Bp = math.log(params.B) + params.beta*lambda_p*psi
        self.psi, self.lambda_p = psi, lambda_p
        self.stats = {'fixed_q_calls': 0, 'n_root_calls': 0, 'q_root_calls': 0}
        self.turns, self.validity_roots, self.branches = [], [], []
        if params.gamma > 0 and cost_scale > 0 and q0 < 1:
            self._build_branches()

    def r(self, q):
        return q*self.cost.hp(q)/self.p.gamma - self.cost.h(q)

    def log_critical_cbar(self, q):
        r = self.r(q)
        if r <= 0:
            return -math.inf
        p = self.p
        t = q*self.cost.hp(q)/p.gamma
        return (math.log(p.beta/(p.alpha*p.A)) + self.log_Bp - p.alpha*self.log_k
                -p.beta*p.gamma*math.log(q) + (p.alpha+1)*math.log(r)
                +(p.beta-1)*math.log(t))/p.beta

    def _build_branches(self):
        a, b, g = self.p.alpha, self.p.beta, self.p.gamma
        q0, kind = self.cost.q0, self.cost.kind
        if kind == 'power':
            lead = (4-g)/g * ((a+b)*4-b*g)
            constant = (b*(4-g)-4)*q0**4
            if lead != 0 and -constant/lead > 0:
                self.turns = [(-constant/lead)**.25]
            r_turns = []  # r is monotone, or constant when gamma=4.
        elif kind == 'exponential':
            x0, x1 = 6*q0, 6.
            poly = np.array([a+b, a+b-g*(a+2*b), g*(1+b*(g-1))])
            dp, ddp = np.polyder(poly), np.polyder(poly, 2)
            def f(x):
                return math.exp(x-x0)*np.polyval(poly, x) + g*((b-1)*(1+x)-b*g)
            def df(x):
                return math.exp(x-x0)*(np.polyval(poly, x)+np.polyval(dp, x)) + g*(b-1)
            ddfpoly = np.polyadd(np.polyadd(poly, 2*dp), ddp)
            df_turns = real_polynomial_roots(ddfpoly, x0, x1)
            f_turns = monotone_roots(df, x0, x1, df_turns)
            self.turns = [x/6 for x in monotone_roots(f, x0, x1, f_turns)]
            r_turns = [(g-1)/6]
        else:
            x0, x1, m = 1+10*q0, 11., a+b
            aa, bb = b*g*g, g*(1-b)
            def f(x):
                return (x-1)*(m-(m+1)*g*x) + (aa*x*x+bb*x)*math.log(x/x0)
            def df(x):
                z = math.log(x/x0)
                return -2*g*(m+1)*x + m+g*(m+1) + aa*x*(2*z+1)+bb*(z+1)
            def ddf(x):
                return -2*g*(m+1) + aa*(2*math.log(x/x0)+3) + bb/x
            # f'''(x)=(2*aa*x-bb)/x^2: at most one zero.
            ddf_turns = [bb/(2*aa)]
            df_turns = monotone_roots(ddf, x0, x1, ddf_turns)
            f_turns = monotone_roots(df, x0, x1, df_turns)
            self.turns = [(x-1)/10 for x in monotone_roots(f, x0, x1, f_turns)]
            r_turns = [(1/g-1)/10]
        self.turns = unique(q for q in self.turns if q0 < q < 1)
        self.validity_roots = monotone_roots(self.r, q0, 1., r_turns, tol=1e-14)
        points = unique([q0, 1.] + self.turns + self.validity_roots)
        for lo, hi in zip(points[:-1], points[1:]):
            if self.r((lo+hi)/2) > 0:
                # The limit at r=0 is -infinity. Store that explicitly.
                lv = -math.inf if any(abs(lo-x)<2e-13 for x in self.validity_roots) else self.log_critical_cbar(lo)
                hv = -math.inf if any(abs(hi-x)<2e-13 for x in self.validity_roots) else self.log_critical_cbar(hi)
                self.branches.append((lo, hi, lv, hv))

    def stationary_qualities(self, budget):
        target = math.log(budget/FLOPS_SCALE)
        roots = []
        for lo, hi, lv, hv in self.branches:
            if abs(lv-target) <= 5e-12:
                roots.append(lo)
            if abs(hv-target) <= 5e-12:
                roots.append(hi)
            if min(lv, hv) < target < max(lv, hv):
                def f(q):
                    if q == lo:
                        return lv-target
                    if q == hi:
                        return hv-target
                    return self.log_critical_cbar(q)-target
                self.stats['q_root_calls'] += 1
                roots.append(brentq(f, lo, hi, xtol=5e-15, rtol=1e-14))
        return [q for q in unique(roots) if self.cost.q0 < q < 1]

    def fixed_q(self, q, budget):
        if not (math.isfinite(budget) and budget > 0):
            raise ValueError("Budget must be finite and positive")
        if not self.cost.q0 <= q <= 1:
            raise ValueError("Quality is outside the feasible interval")
        self.stats['fixed_q_calls'] += 1
        p, h = self.p, self.cost.h(q)
        lc = math.log(budget/FLOPS_SCALE)
        lq = math.log(q)
        center = (math.log(p.alpha*p.A/p.beta)-self.log_Bp
                  +p.beta*(lc-self.log_k+p.gamma*lq))/(p.alpha+p.beta)
        if h == 0:
            ln = center
        else:
            lh = math.log(h)
            def gap(ln):
                ld = float(np.logaddexp(self.log_k+ln, lh))
                return (math.log(p.alpha*p.A)-(p.alpha+1)*ln-math.log(p.beta)
                        -self.log_Bp-self.log_k+p.beta*lc+p.beta*p.gamma*lq-(p.beta-1)*ld)
            width = 2.
            while gap(center-width) < 0 or gap(center+width) > 0:
                width *= 2
                if width > 4096:
                    raise ArithmeticError("Could not bracket the unique scale root")
            self.stats['n_root_calls'] += 1
            ln = brentq(gap, center-width, center+width, xtol=5e-14, rtol=1e-14)
        n = math.exp(ln)
        d = budget/FLOPS_SCALE/(self.k*n+h)
        loss = p.E+p.A*math.exp(-p.alpha*ln)+math.exp(self.log_Bp-p.beta*(math.log(d)+p.gamma*lq))
        return {'n_B': n, 'd_B': d, 'Q': float(q), 'loss': loss, 'h': h, 'k': self.k}

    def evaluate(self, n, q, budget):
        """Loss after spending the budget; accepts vector n/q for diagnostics."""
        d = (budget/FLOPS_SCALE)/(self.k*n+self.cost.h(float(q)))
        return self.p.E+self.p.A*n**(-self.p.alpha)+math.exp(self.log_Bp)*(d*q**self.p.gamma)**(-self.p.beta)

    def solve(self, budget):
        q0, p = self.cost.q0, self.p
        if p.gamma == 0 or q0 == 1:
            candidates = [self.fixed_q(q0, budget)]
        elif self.cost.scale == 0:
            candidates = [self.fixed_q(1., budget)]
        else:
            candidates = [self.fixed_q(q0, budget), self.fixed_q(1., budget)]
            for q in self.stationary_qualities(budget):
                n = self.r(q)/self.k
                d = budget/FLOPS_SCALE/(self.k*n+self.cost.h(q))
                candidates.append({'n_B': n, 'd_B': d, 'Q': q, 'loss': self.evaluate(n,q,budget),
                                   'h': self.cost.h(q), 'k': self.k})
        best = min(candidates, key=lambda row: row['loss'])
        n, d, q, h = (best[k] for k in ('n_B','d_B','Q','h'))
        cbar = budget/FLOPS_SCALE
        train, attention, quality = 6*n*d, ETA*self.context*n*d, d*h
        log_lhs = math.log(p.alpha*p.A)-(p.alpha+1)*math.log(n)
        log_rhs = math.log(p.beta)+self.log_Bp+self.log_k-p.beta*math.log(cbar)+(p.beta-1)*math.log(self.k*n+h)-p.beta*p.gamma*math.log(q)
        hp, qr = self.cost.hp(q), p.gamma*(self.k*n+h)/q
        qgap = (hp-qr)/max(1e-300, hp+qr)
        active = 'lower' if q == q0 else ('upper' if q == 1 else 'interior')
        q_residual = max(0., -qgap) if active=='lower' else (max(0., qgap) if active=='upper' else abs(qgap))
        if q0 == 1:
            q_residual = 0.
        return {'budget_FLOPs': float(budget), 'context_tokens': self.context,
                'cost_model': self.cost.kind, 'cost_model_label': COST_LABELS[self.cost.kind],
                'N_B': n, 'D_B': d, 'Q0': q0, 'Q_star': q, 'loss_star': best['loss'],
                'training_share': train/cbar, 'quality_share': quality/cbar, 'attention_share': attention/cbar,
                'share_sum': (train+quality+attention)/cbar,
                'budget_slack_FLOPs': (cbar-train-attention-quality)*FLOPS_SCALE,
                'quality_regime': {'lower':'基线质量态','upper':'质量饱和态','interior':'质量内点态'}[active],
                'active_boundary': active, 'attention_to_training': ETA*self.context/6, 'k_context': self.k,
                'N_within_B6_support': .07 <= n <= 11.97, 'D_within_B6_support': 10 <= d <= 600,
                'Q_within_B6_support': .1 <= q <= 1, 'budget_relative_residual': abs((train+quality+attention)/cbar-1),
                'n_stationarity_log_residual': abs(log_lhs-log_rhs), 'q_kkt_normalized_residual': q_residual,
                'candidate_count': len(candidates), 'monotone_branches': len(self.branches),
                'psi': self.psi, 'lambda_p': self.lambda_p, 'lambda_p_identified': False,
                'quality_cost_scale': self.cost.scale}

    def saturation_transition(self, scan_lo=1e19, scan_hi=1e27):
        row = {'context_tokens': self.context, 'cost_model': self.cost.kind,
               'transition_budget_FLOPs': None, 'boundary_candidate_FLOPs': None}
        if self.p.gamma == 0 or self.cost.q0 == 1 or self.cost.scale == 0:
            row['transition_type'] = ('quality_interval_singleton' if self.cost.q0 == 1 else
                                      'no_quality_benefit' if self.p.gamma == 0 else 'free_quality')
            row['scan_status'] = 'no_finite_transition'
            return row
        if self.r(1.) > 0:
            lb = self.log_critical_cbar(1.)
            if -690 < lb < 660:
                candidate = FLOPS_SCALE*math.exp(lb)
                row['boundary_candidate_FLOPs'] = candidate
                end = self.fixed_q(1., candidate)
                global_solution = self.solve(candidate)
                row['boundary_global_loss_gap'] = end['loss']-global_solution['loss_star']
                increasing = self.log_critical_cbar(1.) > self.log_critical_cbar(1.-1e-7)
                if increasing and row['boundary_global_loss_gap'] < 1e-10:
                    row['transition_budget_FLOPs'] = candidate
                    row['transition_type'] = 'continuous_global_boundary'
        if row['transition_budget_FLOPs'] is None:
            # Endpoint may win by a discontinuous branch switch. Never label
            # an absent positive KKT candidate as "no transition" automatically.
            lo, hi = -30., 70.
            if self.solve(10.**lo)['active_boundary']=='upper' or self.solve(10.**hi)['active_boundary']!='upper':
                row['transition_type'] = 'unresolved_outside_extended_search'
                row['scan_status'] = 'unresolved'
                return row
            def branch_gap(lb):
                budget=10.**lb
                upper=self.fixed_q(1.,budget)['loss']
                others=[self.fixed_q(self.cost.q0,budget)['loss']]
                for q in self.stationary_qualities(budget):
                    others.append(self.evaluate(self.r(q)/self.k,q,budget))
                return upper-min(others)
            root=brentq(branch_gap,lo,hi,xtol=1e-11,rtol=1e-12)
            row['transition_budget_FLOPs'] = 10.**root
            row['transition_type'] = 'global_branch_switch_numeric'
        value = row['transition_budget_FLOPs']
        row['scan_status'] = ('before_scan' if value < scan_lo else 'after_scan' if value > scan_hi else 'within_scan')
        return row

    def near_optimal_profiles(self, budget, tolerance=.001):
        """Exact 1D profile sublevel sets, not a box for the joint feasible set."""
        best = self.solve(budget)
        level = best['loss_star']+tolerance
        def qgap(q):
            return self.fixed_q(q,budget)['loss']-level
        qs = [self.cost.q0,1.]+self.stationary_qualities(budget)
        crossings = monotone_roots(qgap,self.cost.q0,1.,qs,tol=1e-13)
        points = unique(qs[:2]+crossings)
        intervals = [(lo,hi) for lo,hi in zip(points[:-1],points[1:]) if qgap((lo+hi)/2)<=0]
        qstar, lnstar = best['Q_star'],math.log(best['N_B'])
        def ngap(ln):
            return self.evaluate(math.exp(ln),qstar,budget)-level
        lo,hi = lnstar-1.,lnstar+1.
        while ngap(lo)<0: lo-=1.
        while ngap(hi)<0: hi+=1.
        nlo=math.exp(brentq(ngap,lo,lnstar,xtol=1e-13))
        nhi=math.exp(brentq(ngap,lnstar,hi,xtol=1e-13))
        return {'budget_FLOPs':budget,'context_tokens':self.context,'cost_model':self.cost.kind,
                'absolute_loss_tolerance':tolerance,'Q_star':qstar,'Q_profile_intervals':intervals,
                'Q_profile_min': min((a for a,b in intervals),default=qstar),
                'Q_profile_max': max((b for a,b in intervals),default=qstar),
                'N_star_B':best['N_B'],'N_at_fixed_Q_min_B':nlo,'N_at_fixed_Q_max_B':nhi,
                'D_at_fixed_Q_min_B':budget/FLOPS_SCALE/(self.k*nhi+self.cost.h(qstar)),
                'D_at_fixed_Q_max_B':budget/FLOPS_SCALE/(self.k*nlo+self.cost.h(qstar))}
