"""Convex, raw-score joint quantile QP with an auditable global lower bound."""
from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.optimize import minimize
from protocol import BOX,QS,GROUPS,PROTOCOL

def publisher_weights(frame,counts=None):
    n=frame.groupby('org').org.transform('size').to_numpy(float)
    counts=np.ones(len(frame)) if counts is None else np.asarray(counts,float)
    w=counts/n
    if w.sum()<=0:raise ValueError('Empty publisher mass')
    return w/w.sum()

def raw_features(frame,model):
    names=['logN','tau'] if 'size_time' in model else ['tau']
    return frame[names].to_numpy(float),names

def domain_vertices(names):return np.array(list(product(*(BOX[n] for n in names))),float)

@dataclass
class JointFit:
    model:str
    coef:np.ndarray
    center:np.ndarray
    scale:np.ndarray
    names:list
    aux_slope:np.ndarray
    diagnostics:dict

    def matrix(self,frame):
        a=frame[self.names].to_numpy(float)
        if not np.isfinite(a).all():raise ValueError('Nonfinite prediction input')
        for j,n in enumerate(self.names):
            if (a[:,j]<BOX[n][0]-1e-10).any() or (a[:,j]>BOX[n][1]+1e-10).any():
                raise ValueError('Prediction outside declared domain: '+n)
        z=(a-self.center)/self.scale
        slope=z@self.coef[:len(self.names)]
        offsets=self.coef[len(self.names):].reshape(2,3)
        group=np.array([GROUPS.index(g) for g in frame.group])
        return 100*(slope[:,None]+offsets[group])

    def to_dict(self):
        return dict(model=self.model,coef=self.coef.tolist(),center=self.center.tolist(),
            scale=self.scale.tolist(),names=self.names,aux_slope=self.aux_slope.tolist(),diagnostics=self.diagnostics)

def fit_joint(aux,train,model='joint_size_time',penalty=1.,aux_counts=None,strict_counts=None):
    wa=publisher_weights(aux,aux_counts);w=publisher_weights(train,strict_counts)
    a,names=raw_features(aux,model);xx,_=raw_features(train,model)
    center=np.average(a,axis=0,weights=wa)
    scale=np.sqrt(np.average((a-center)**2,axis=0,weights=wa));scale[scale<1e-8]=1.
    az=(a-center)/scale;z=(xx-center)/scale;nv=len(names)
    # The auxiliary intercept and type effect are discarded. Only slopes shrink.
    ax=np.column_stack([np.ones(len(aux)),az,aux.group.eq('posttrained')])
    reg=np.diag([0.]+[PROTOCOL['auxiliary_ridge']]*(nv+1))
    ca=np.linalg.solve(ax.T@(wa[:,None]*ax)+reg,ax.T@(wa*aux.S.to_numpy()/100))
    target=ca[1:1+nv]
    n=len(train);p=nv+6;nr=3*n
    design=np.zeros((nr,p));yy=np.repeat(train.S.to_numpy()/100,3)
    levels=np.tile(QS,n);ww=np.repeat(w/3,3)
    for i,g in enumerate(train.group):
        for j in range(3):
            row=3*i+j;design[row,:nv]=z[i]
            design[row,nv+3*GROUPS.index(g)+j]=1
    vertices=(domain_vertices(names)-center)/scale
    vdesign=[];orders=[]
    for g in range(2):
        for v in vertices:
            for j in range(3):
                row=np.zeros(p);row[:nv]=v;row[nv+g*3+j]=1;vdesign.append(row)
        for j in range(2):
            row=np.zeros(p);row[nv+g*3+j+1]=1;row[nv+g*3+j]=-1;orders.append(row)
    vd=np.array(vdesign);od=np.array(orders)
    # D x >= b: t >= tau*(y-pred); t >= (tau-1)*(y-pred).
    dim=p+nr
    D=np.vstack([np.column_stack([levels[:,None]*design,np.eye(nr)]),
        np.column_stack([(levels-1)[:,None]*design,np.eye(nr)]),
        np.column_stack([vd,np.zeros((len(vd),nr))]),
        np.column_stack([-vd,np.zeros((len(vd),nr))]),
        np.column_stack([od,np.zeros((len(od),nr))])])
    b=np.r_[levels*yy,(levels-1)*yy,np.zeros(len(vd)),-np.ones(len(vd)),np.zeros(len(od))]
    lower=np.r_[np.full(p,-10.),np.zeros(nr)];upper=np.r_[np.full(p,10.),np.ones(nr)]
    def fun(x):return float(ww@x[p:]+penalty*np.sum((x[:nv]-target)**2))
    def jac(x):return np.r_[2*penalty*(x[:nv]-target),np.zeros(6),ww]
    x0=np.zeros(dim)
    # A constant across types is an always-feasible ordered initial model.
    empirical=np.quantile(train.S.to_numpy()/100,QS)
    x0[nv:p]=np.tile(empirical,2)
    residual=yy-design@x0[:p];x0[p:]=np.maximum(levels*residual,(levels-1)*residual)+1e-7
    for refinement,ftol in enumerate([2e-12,1e-14,1e-15]):
        rr=minimize(fun,x0,jac=jac,method='SLSQP',bounds=list(zip(lower,upper)),
            constraints={'type':'ineq','fun':lambda x:D@x-b,'jac':lambda x:D},
            options={'ftol':ftol,'maxiter':400})
        x=rr.x;slack=D@x-b;primal=float(fun(x));mu=np.maximum(rr.multipliers,0.)
        residual_grad=jac(x)-D.T@mu
        # Minimize the complete quadratic Lagrangian over the declared box.
        movement=np.where(residual_grad>=0,lower-x,upper-x)
        movement[:nv]=np.clip(-residual_grad[:nv]/(2*penalty),(lower-x)[:nv],(upper-x)[:nv])
        dual=float(primal-mu@slack+residual_grad@movement+penalty*np.sum(movement[:nv]**2))
        gap=primal-dual
        if slack.min()>=-2e-8 and gap<=1e-7:break
        x0=x
    if slack.min()<-2e-8 or gap>1e-7:
        raise ArithmeticError(f'Joint QP failed: {rr.message}; feasibility={slack.min():.3g}, gap={gap:.3g}')
    pred=design@x[:p]
    loss=float(np.sum(ww*np.maximum(levels*(yy-pred),(levels-1)*(yy-pred))))
    if abs(primal-loss-penalty*np.sum((x[:nv]-target)**2))>1e-8:raise ArithmeticError('Epigraph mismatch')
    diag=dict(rows=n,aux_rows=len(aux),publishers=train.org.nunique(),penalty=penalty,
        objective=primal,objective_lower=dual,optimality_gap=gap,weighted_pinball=loss,
        min_constraint_slack=float(slack.min()),min_score=float((vd@x[:p]).min()),
        max_score=float((vd@x[:p]).max()),min_quantile_spacing=float((od@x[:p]).min()),
        iterations=int(rr.nit),numerical_refinements=refinement,coefficient_box_active=bool(np.max(abs(x[:p]))>9.999),
        solver_status=str(rr.message),scale_from_aux_only=True)
    return JointFit(model,x[:p],center,scale,names,target,diag)

def conformal_rank(scores,alpha=.1):
    scores=np.asarray(scores,float);m=len(scores);rank=int(np.ceil((m+1)*(1-alpha)))
    if not np.isfinite(scores).all():raise ValueError('Invalid calibration score')
    if rank>m:return np.inf,rank
    return max(0.,float(np.sort(scores)[rank-1])),rank
