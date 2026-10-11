"""Curved quality response, convex scale solve, and bounded quality search.

Bounds are analytic in exact arithmetic. The reported certificates include
conservative floating-point padding, not formal interval-arithmetic proof.
"""
from dataclasses import dataclass
import heapq
import math
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import brentq, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

sys.path.insert(0, str(paths.CODE / "q3"))
from q3_core import QualityCost, COST_LABELS, FLOPS_SCALE, ETA


@dataclass(frozen=True)
class Params:
    E: float
    A: float
    B: float
    alpha: float
    beta: float
    delta: float
    kappa: float=0.

    def __post_init__(self):
        if not all(math.isfinite(x) for x in vars(self).values()):
            raise ValueError("Nonfinite parameter")
        if self.E<0 or min(self.A,self.B,self.alpha,self.beta)<=0:
            raise ValueError("Invalid physical parameters")
        if min(self.slope(.1),self.slope(1)) < -1e-9:
            raise ValueError("Quality response is not monotone on the declared support")

    @classmethod
    def from_dict(cls,d):
        if 'gamma_Q' in d and 'delta' not in d:
            d={**d,'delta':d['beta']*d['gamma_Q'],'kappa':0.}
        return cls(**{k:float(d.get(k,0.)) for k in cls.__dataclass_fields__})

    def g(self,q):
        u=math.log(q)
        return -self.delta*u+self.kappa*u*u

    def slope(self,q):
        return self.delta-2*self.kappa*math.log(q)


class Problem:
    def __init__(self,params,context,kind,q0,*,cost_scale=1.,psi=0.,mix_lambda=0.,
                 offsets=None):
        self.params=list(params)
        if not self.params: raise ValueError("No scenarios")
        if not isinstance(context,(int,np.integer)) or context<=0:
            raise ValueError("Context must be a positive integer")
        if not .1<=q0<=1: raise ValueError("Q0 must remain within Q2 support [.1,1]")
        self.context=int(context)
        self.k=6+ETA*context
        self.cost=QualityCost(kind,q0,cost_scale)
        self.q0=q0
        self.p=np.array([[p.E,p.A,p.B,p.alpha,p.beta,p.delta,p.kappa] for p in params]).T
        self.e,self.a,self.b,self.alpha,self.beta,self.delta,self.kappa=self.p
        self.logb=np.log(self.b)+self.beta*mix_lambda*psi
        self.offsets=np.zeros(len(params)) if offsets is None else np.array(offsets,float)
        if self.offsets.shape!=(len(params),): raise ValueError("Bad offset count")
        self.calls=0

    def g(self,q):
        u=math.log(q)
        return -self.delta*u+self.kappa*u*u

    def values(self,z,q,budget,*,h_override=None,g_override=None):
        h=self.cost.h(q) if h_override is None else h_override
        g=self.g(q) if g_override is None else g_override
        kn=self.k*math.exp(z)
        s=kn+h
        nt=self.a*np.exp(-self.alpha*z)
        dt=np.exp(self.logb+self.beta*(math.log(s)-math.log(budget/FLOPS_SCALE))+g)
        val=self.e+nt+dt-self.offsets
        ratio=kn/s
        grad=-self.alpha*nt+self.beta*dt*ratio
        hess=self.alpha**2*nt+self.beta*dt*ratio*(1+(self.beta-1)*ratio)
        return val,grad,hess

    def explicit_loss(self,n,d,q):
        return self.e+self.a*n**(-self.alpha)+np.exp(self.logb-self.beta*math.log(d)+self.g(q))

    def scale(self,q,budget,*,h_override=None,g_override=None,tol=1e-11):
        """Convex maximum on R. Opposing supporting lines give a lower bound."""
        self.calls+=1
        if len(self.params)==1:
            p=self.params[0]
            h=self.cost.h(q) if h_override is None else h_override
            gg=p.g(q) if g_override is None else float(g_override[0])
            offset=float(self.offsets[0])
            ld=float(self.logb[0])+gg-p.beta*math.log(budget/FLOPS_SCALE)
            def one(z):
                kn=self.k*math.exp(z);s=kn+h;r=kn/s
                nt=p.A*math.exp(-p.alpha*z);dt=math.exp(ld+p.beta*math.log(s))
                return p.E+nt+dt-offset,-p.alpha*nt+p.beta*dt*r
            lo,hi=-4.,4.
            while one(lo)[1]>0:lo-=8.
            while one(hi)[1]<0:hi+=8.
            z=brentq(lambda x:one(x)[1],lo,hi,xtol=2e-14,rtol=1e-14)
            v=one(z)[0]
            # Use actual opposing tangents, rather than treating a small
            # stationarity residual alone as a rigorous objective bound.
            a,b=z-1e-6,z+1e-6
            fa,ga=one(a);fb,gb=one(b)
            if not ga<0<gb:raise RuntimeError("Single-scenario tangent bracket failed")
            cross=(fa-ga*a-fb+gb*b)/(gb-ga)
            low=fa+ga*(cross-a)-2e-12*(1+abs(fa)+abs(fb))
            return dict(value=v,lower=min(low,v),z=z,inner_gap=max(0.,v-low),iterations=1)
        def at(z):
            v,g,h=self.values(z,q,budget,h_override=h_override,g_override=g_override)
            i=int(np.argmax(v))
            return float(v[i]),float(g[i]),float(h[i])
        # Coercivity and increasing subgradients guarantee eventual bracketing.
        lo,hi=-4.,4.
        fl,gl,_=at(lo);fh,gh,_=at(hi)
        for _ in range(16):
            if gl<=0<=gh: break
            if gl>0: hi,fh,gh=lo,fl,gl;lo-=8.;fl,gl,_=at(lo)
            if gh<0: lo,fl,gl=hi,fh,gh;hi+=8.;fh,gh,_=at(hi)
        else: raise RuntimeError("Could not bracket convex scale minimum")
        best=(fl,lo) if fl<fh else (fh,hi)
        lb=-math.inf
        for iteration in range(90):
            # Both lines lie below the maximum for every real z.
            if gh-gl>0:
                cross=(fl-gl*lo-fh+gh*hi)/(gh-gl)
                lb=fl+gl*(cross-lo)
            else: lb=min(fl,fh)
            pad=2e-12*(1+abs(fl)+abs(fh))
            lb-=pad
            if best[0]-lb<=tol or hi-lo<2e-12: break
            mid=.5*(lo+hi)
            fm,gm,hm=at(mid)
            if fm<best[0]:best=(fm,mid)
            if gm<0:lo,fl,gl=mid,fm,gm
            else:hi,fh,gh=mid,fm,gm
        else: raise RuntimeError("Convex scale solve did not reach tolerance")
        # Smooth single-scenario polish makes stored stationarity meaningful.
        if len(self.params)==1:
            z=brentq(lambda x:at(x)[1],lo,hi,xtol=5e-14,rtol=1e-14) if gl<0<gh else best[1]
            f=at(z)[0]
            if f<best[0]+1e-13:best=(f,z)
        return {'value':best[0],'lower':min(lb,best[0]),'z':best[1],
                'inner_gap':max(0.,best[0]-lb),'iterations':iteration+1}

    def node(self,a,b,budget):
        # h is nondecreasing and each g(log Q) is nonincreasing.
        # Splitting them at opposite endpoints gives a global interval lower bound.
        return self.scale(a,budget,h_override=self.cost.h(a),g_override=self.g(b))['lower']

    def solve(self,budget,*,tol=1e-7,max_nodes=100000,refine=True):
        if not math.isfinite(budget) or budget<=0:raise ValueError("Invalid budget")
        count0=self.calls
        points={}
        def at(q):
            q=float(q)
            if q not in points:points[q]=self.scale(q,budget)
            return points[q]['value']
        def lower_bound(a,b):
            if len(self.params)==1 and self.params[0].beta<=1:
                # Each fixed-Q scale optimum increases with Q when beta<=1.
                # Bounds on its profile derivative can certify monotonicity.
                at(a);at(b)
                p=self.params[0]
                na=math.exp(points[a]['z']);nb=math.exp(points[b]['z'])
                hpmin=min(self.cost.hp(a),self.cost.hp(b))
                hpmax=max(self.cost.hp(a),self.cost.hp(b))
                smin=min(p.slope(a),p.slope(b));smax=max(p.slope(a),p.slope(b))
                upper=p.beta*hpmax/(self.k*na+self.cost.h(a))-smin/b
                lower=p.beta*hpmin/(self.k*nb+self.cost.h(b))-smax/a
                if upper < -1e-11:return points[b]['lower']
                if lower > 1e-11:return points[a]['lower']
            return self.node(a,b,budget)
        # Feasible upper candidates; no optimality claim rests on this mesh.
        grid=np.linspace(self.q0,1.,9)
        for q in grid:at(q)
        if len(self.params)==1 and self.q0<1:
            p=self.params[0]
            def qcondition(q):
                at(q)
                n=math.exp(points[float(q)]['z'])
                return p.beta*self.cost.hp(q)/(self.k*n+self.cost.h(q))-p.slope(q)/q
            for a,b in zip(grid[:-1],grid[1:]):
                if qcondition(a)*qcondition(b)<0:
                    at(brentq(qcondition,a,b,xtol=2e-14,rtol=1e-14))
        if refine:
            for i in range(1,len(grid)-1):
                if at(grid[i])<=at(grid[i-1]) and at(grid[i])<=at(grid[i+1]):
                    opt=minimize_scalar(at,bounds=(grid[i-1],grid[i+1]),method='bounded',
                                        options={'xatol':2e-12})
                    at(opt.x)
        qb=min(points,key=lambda q:points[q]['value']);best=points[qb]['value']
        heap=[];serial=0
        for a,b in zip(grid[:-1],grid[1:]):
            heapq.heappush(heap,(lower_bound(a,b),serial,float(a),float(b)));serial+=1
        iterations=0
        while heap and best-heap[0][0]>tol:
            if iterations>=max_nodes:raise RuntimeError("Quality bound search exhausted max_nodes")
            lb,_,a,b=heapq.heappop(heap)
            mid=.5*(a+b);v=at(mid)
            if v<best:best,qb=v,mid
            for aa,bb in ((a,mid),(mid,b)):
                low=lower_bound(aa,bb)
                if low<=best:
                    heapq.heappush(heap,(low,serial,aa,bb));serial+=1
            iterations+=1
        lower=min(best,heap[0][0]) if heap else best
        # Polish the incumbent locally without replacing the global lower bound.
        if refine and self.q0+1e-9<qb<1-1e-9:
            opt=minimize_scalar(at,bounds=(max(self.q0,qb-.003),min(1.,qb+.003)),
                                method='bounded',options={'xatol':1e-13})
            if at(opt.x)<best:best,qb=at(opt.x),float(opt.x)
        z=points[qb]['z'];n=math.exp(z);h=self.cost.h(qb)
        d=budget/FLOPS_SCALE/(self.k*n+h)
        active='lower' if qb==self.q0 else 'upper' if qb==1. else 'interior'
        row=dict(budget_FLOPs=float(budget),context_tokens=self.context,cost_model=self.cost.kind,
            cost_scale=self.cost.scale,Q0=self.q0,N_B=n,D_B=d,Q_star=qb,objective=best,
            objective_lower=lower,optimality_gap=max(0.,best-lower),
            active_boundary=active,quality_share=d*h/(budget/FLOPS_SCALE),
            training_share=6*n*d/(budget/FLOPS_SCALE),attention_share=ETA*self.context*n*d/(budget/FLOPS_SCALE),
            budget_relative_residual=abs(d*(self.k*n+h)/(budget/FLOPS_SCALE)-1),
            scale_calls=self.calls-count0,interval_splits=iterations,scenario_count=len(self.params),
            N_within_B6_support=.07<=n<=11.97,D_within_B6_support=10<=d<=600)
        if len(self.params)==1:
            p=self.params[0];kn=self.k*n;s=kn+h
            nt=p.A*n**(-p.alpha);dt=math.exp(self.logb[0]-p.beta*math.log(d)+p.g(qb))
            gn=-p.alpha*nt+p.beta*dt*kn/s
            gq=dt*(p.beta*self.cost.hp(qb)/s-p.slope(qb)/qb)
            qkkt=max(0.,-gq) if active=='lower' else max(0.,gq) if active=='upper' else abs(gq)
            row.update(loss_star=best,n_gradient=gn,q_gradient=gq,q_kkt_violation=qkkt)
        return row


def local_boundary_budget(p,context,cost):
    """Q=1 KKT candidate only; separate global checking is required."""
    gamma=p.slope(1)/p.beta
    if gamma<=0 or cost.scale==0:return None
    k=6+ETA*context
    s=cost.hp(1)/gamma
    r=s-cost.h(1)
    if r<=0:return None
    n=r/k
    logc=(math.log(p.beta*p.B*k/(p.alpha*p.A))+(p.alpha+1)*math.log(n)
          +(p.beta-1)*math.log(s)+p.g(1))/p.beta
    return FLOPS_SCALE*math.exp(logc)


def sufficient_saturation_bound(p,context,cost):
    """Conservative computable bound for Q=1 globally optimal above Cbar.

    Bound M using max Q, max h', min gamma and h>=0; then bound C(M,Q)
    by elementary extrema. No grid supremum is substituted for a theorem.
    """
    gamma_min=min(p.slope(cost.q0),p.slope(1))/p.beta
    if gamma_min<=0:return None
    k=6+ETA*context
    hpmax=max(cost.hp(cost.q0),cost.hp(1))
    M=(hpmax/gamma_min+1e-6)/k
    smin=k*M;smax=smin+cost.h(1)
    sm=smax if p.beta>=1 else smin
    logc=(math.log(p.beta*p.B*k/(p.alpha*p.A))+(p.alpha+1)*math.log(M)
          +(p.beta-1)*math.log(sm)+p.g(cost.q0))/p.beta
    return dict(M_N_B=M,sufficient_budget_FLOPs=FLOPS_SCALE*math.exp(logc),gamma_min=gamma_min)


