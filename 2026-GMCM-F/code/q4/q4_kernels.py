"""Small numerical kernels: weighted SVD and weighted sparse quantile LP."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog
from scipy import sparse
from scipy.special import expit,logit


def pinball(y,pred,q):
    e=np.asarray(y)-np.asarray(pred)
    return np.maximum(q*e,(q-1)*e)


def weighted_quantile(values,q,counts=None):
    """Exactly numpy's linear quantile on integer-frequency replicated data."""
    values=np.asarray(values,float)
    counts=np.ones(len(values),dtype=int) if counts is None else np.asarray(counts)
    if np.any(counts<0) or np.any(counts!=np.floor(counts)):
        raise ValueError('Quantile frequencies must be nonnegative integers')
    keep=(counts>0)&np.isfinite(values)
    v,w=values[keep],counts[keep].astype(int)
    if not len(v) or not 0<=q<=1: raise ValueError('Empty or invalid weighted quantile')
    order=np.argsort(v,kind='stable');v,w=v[order],w[order];cum=np.cumsum(w)
    at=q*(int(cum[-1])-1);lo=int(np.floor(at));hi=int(np.ceil(at))
    return float((1-(at-lo))*v[np.searchsorted(cum,lo,side='right')]+(at-lo)*v[np.searchsorted(cum,hi,side='right')])


def weighted_lstsq(x,y,counts=None,require_rank=True):
    x,y=np.asarray(x,float),np.asarray(y,float)
    w=np.ones(len(y)) if counts is None else np.asarray(counts,float)
    if len(x)!=len(y) or w.shape!=y.shape or np.any(w<0) or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(w).all():
        raise ValueError('Nonfinite or inconsistent least-squares inputs')
    keep=w>0
    if not keep.any(): raise ValueError('No positively weighted observations')
    sw=np.sqrt(w[keep]);a=x[keep]*sw[:,None];b=y[keep]*sw
    coef,_,rank,s=np.linalg.lstsq(a,b,rcond=None)
    if require_rank and rank<x.shape[1]: raise ValueError('Rank deficient design')
    return coef,{'rank':int(rank),'condition':float(s[0]/s[-1]) if s[-1]>0 else float('inf'),
                 'weighted_sse':float(np.sum(w*(y-x@coef)**2)),'effective_rows':int(keep.sum()),'replicated_rows':float(w.sum())}


def cluster_counts(groups,drawn_groups):
    names,inverse=np.unique(np.asarray(groups,str),return_inverse=True)
    lookup={name:i for i,name in enumerate(names)}
    counts=np.zeros(len(names),dtype=int)
    for key in drawn_groups:
        if key in lookup:counts[lookup[key]]+=1
    return counts[inverse]


@dataclass
class QuantileFit:
    coef: np.ndarray
    quantile: float
    objective: float
    backend: str
    rank: int
    n_unique: int
    status: str

    def predict(self,x):return np.asarray(x,float)@self.coef


def quantile_fit(x,y,q,counts=None,backend='auto',require_rank=True):
    x,y=np.asarray(x,float),np.asarray(y,float)
    w=np.ones(len(y)) if counts is None else np.asarray(counts,float)
    if x.ndim!=2 or len(x)!=len(y) or len(w)!=len(y) or not 0<q<1:
        raise ValueError('Invalid quantile inputs')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(w).all() or np.any(w<0):
        raise ValueError('Nonfinite quantile inputs or negative frequency')
    keep=w>0;x=x[keep];y=y[keep];w=w[keep]
    n,p=x.shape
    if n<p or not np.allclose(x[:,0],1):raise ValueError('Require intercept column and enough rows')
    centers=np.average(x,axis=0,weights=w);centers[0]=0
    scales=np.sqrt(np.average((x-centers)**2,axis=0,weights=w));scales[0]=1
    scales[scales<1e-12]=1
    z=(x-centers)/scales
    rank=np.linalg.matrix_rank(z)
    if require_rank and rank<p:raise ValueError('Rank deficient quantile design')
    chosen=('dense' if n<128 else 'sparse') if backend=='auto' else backend
    if chosen=='sparse':
        equality=sparse.hstack([sparse.csr_matrix(z),sparse.eye(n,format='csr'),-sparse.eye(n,format='csr')],format='csr')
    elif chosen=='dense':equality=np.column_stack([z,np.eye(n),-np.eye(n)])
    else:raise ValueError('Unknown quantile backend')
    objective=np.r_[np.zeros(p),q*w,(1-q)*w]
    solved=linprog(objective,A_eq=equality,b_eq=y,bounds=[(None,None)]*p+[(0,None)]*(2*n),method='highs')
    if not solved.success:raise RuntimeError(f'Quantile LP failed: {solved.message}')
    beta=solved.x[:p]/scales;beta[0]-=float(centers@beta)
    value=float(np.sum(w*pinball(y,x@beta,q)))
    if abs(value-solved.fun)>1e-6*max(1,abs(value)):raise ArithmeticError('Quantile objective mismatch')
    return QuantileFit(beta,q,value,chosen,int(rank),n,'optimal')


def score_to_logit(y):return logit(np.clip(np.asarray(y,float)/100,1e-4,1-1e-4))
def logit_to_score(z):return 100*expit(z)


def interval_summary(values,lo=.05,hi=.95):
    v=np.asarray(values,float)
    if not np.isfinite(v).all() or not len(v):raise ValueError('Nonfinite or empty uncertainty sample')
    return {'median':float(np.median(v)),'lo':float(np.quantile(v,lo)),'hi':float(np.quantile(v,hi)),'draws':len(v)}
