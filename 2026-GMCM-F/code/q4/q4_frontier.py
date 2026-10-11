"""Frozen frontier candidates, chronological evaluation and compute regressions."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from data_config import ORIGIN,SELECTION_CUTOFF,CUTOFFS,HORIZON_MONTHS,QUANTILES,FRONTIER_MODELS,OUT,SEED
from q4_data import asof,compute_asof,save_table,write_json
from q4_kernels import quantile_fit,weighted_lstsq,weighted_quantile,score_to_logit,logit_to_score,pinball,cluster_counts


def design(frame,model):
    cols=[np.ones(len(frame))]
    if 'size_time' in model:cols.append(frame.logN.to_numpy(float))
    cols.extend([frame.tau.to_numpy(float),frame.group.eq('posttrained').to_numpy(float)])
    return np.column_stack(cols)


@dataclass
class Frontier:
    model:str
    quantile:float
    coef:np.ndarray|None
    empirical:dict
    fallback:str
    diagnostics:dict

    def predict(self,frame):
        if self.coef is None:return frame.group.map(self.empirical).fillna(self.empirical['all']).to_numpy(float)
        z=design(frame,self.model)@self.coef
        return logit_to_score(z) if self.model.startswith('bounded') else z


def fit_frontier(train,model,q,counts=None,cutoff=ORIGIN):
    w=np.ones(len(train),dtype=int) if counts is None else np.asarray(counts,dtype=int)
    if w.sum()==0:raise ValueError('No training observations in bootstrap')
    positive=train.loc[w>0]
    empirical={'all':weighted_quantile(train.S,q,w)}
    for group in ['pretrained','posttrained']:
        recent=train.group.eq(group)&train.date.ge(pd.Timestamp(cutoff)-pd.Timedelta(days=180))
        full=train.group.eq(group)
        selected=recent if w[recent].sum()>=3 else full
        empirical[group]=weighted_quantile(train.loc[selected,'S'],q,w[selected]) if w[selected].sum()>0 else empirical['all']
    if model=='type_empirical_180d':return Frontier(model,q,None,empirical,'',{'n':len(positive),'kind':'empirical'})
    try:
        if len(positive)<10:raise ValueError('Fewer than 10 distinct positively weighted training rows')
        y=score_to_logit(train.S) if model.startswith('bounded') else train.S.to_numpy(float)
        fit=quantile_fit(design(train,model),y,q,w)
        return Frontier(model,q,fit.coef,empirical,'',{'n':fit.n_unique,'rank':fit.rank,'objective':fit.objective,'backend':fit.backend})
    except (ValueError,RuntimeError,ArithmeticError) as err:
        return Frontier(model,q,None,empirical,str(err),{'n':len(positive),'kind':'empirical_fallback'})


def compute_fit(c4,cutoff=ORIGIN,counts=None,language_only=True):
    f=compute_asof(c4,cutoff,language_only)
    # Counts keyed by stable C4 row id avoid any alignment by frame position.
    w=np.ones(len(f),dtype=int) if counts is None else f.c4_id.map(counts).fillna(0).to_numpy(int)
    annual=[]
    for year,g in f.groupby('year'):
        mask=f.year.eq(year)
        if year<2018 or year>=pd.Timestamp(cutoff).year or w[mask].sum()==0:continue
        annual.append({'year':int(year),'compute_q90':weighted_quantile(g.F,.9,w[mask]),'rows':int(w[mask].sum()),'distinct_rows':int((w[mask]>0).sum())})
    a=pd.DataFrame(annual)
    if len(a)<3:raise ValueError('Fewer than 3 complete years for growth fit')
    growth,gd=weighted_lstsq(np.column_stack([np.ones(len(a)),a.year-2020]),np.log10(a.compute_q90))
    complete=f.N.gt(0)&f.D.gt(0)&np.isfinite(f.N)&np.isfinite(f.D)
    d=f[complete];ww=w[complete]
    x=np.column_stack([np.ones(len(d)),np.log10(d.N),d.year-2020])
    beta,bd=weighted_lstsq(x,np.log10(d.F),ww)
    if beta[1]<=0:raise ValueError('Nonpositive compute-to-size elasticity')
    return {'g':float(growth[1]),'kappa':float(beta[1]),'xi':float(beta[2]),'intercept':float(beta[0]),
            'complete_rows':int((ww>0).sum()),'growth_years':len(a),'growth_rank':gd['rank'],
            'regression_rank':bd['rank'],'regression_condition':bd['condition']},a


def reference_n(train,group,counts=None,cutoff=ORIGIN):
    w=np.ones(len(train),int) if counts is None else np.asarray(counts,int)
    mask=train.group.eq(group)&train.date.ge(pd.Timestamp(cutoff)-pd.Timedelta(days=180))
    if w[mask].sum()<3:mask=train.group.eq(group)
    if w[mask].sum()==0:mask=np.ones(len(train),bool)
    return weighted_quantile(train.loc[mask,'N'],.8,w[mask])


def project_test_n(train,test,c4,cutoff):
    fit,_=compute_fit(c4,cutoff);frame=test.copy()
    h=(frame.date-pd.Timestamp(cutoff)).dt.days.to_numpy()/365.2425
    refs=np.array([reference_n(train,g,cutoff=cutoff) for g in frame.group])
    frame['logN']=np.log10(refs)+(fit['g']-fit['xi'])/fit['kappa']*h
    frame['N']=10**frame.logN
    return frame,fit


def metric_table(predictions,keys):
    rows=[]
    for key,d in predictions.groupby(keys,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        q=float(d['quantile'].iloc[0]) if 'quantile' in d else .85
        rows.append({**dict(zip(keys,key)),'predictions':len(d),'distinct_repositories':d.repo.nunique(),
                     'pinball':float(np.mean(pinball(d.actual,d.prediction,q))),
                     'exceedance_rate':float(np.mean(d.actual>d.prediction)),
                     'MAE_pp':float(np.mean(abs(d.actual-d.prediction))),
                     'RMSE_pp':float(np.sqrt(np.mean((d.actual-d.prediction)**2))),
                     'PI90_coverage':float(np.mean((d.actual>=d.pi_low)&(d.actual<=d.pi_high))),
                     'PI90_mean_width_pp':float(np.mean(d.pi_high-d.pi_low)),
                     'predictions_outside_0_100':int(((d.prediction<0)|(d.prediction>100)).sum()),
                     'interval_crossings_reordered':int(d.interval_crossed.sum()),'fallback_rows':int(d.fallback.ne('').sum())})
    return pd.DataFrame(rows)


def evaluate_fold(train,test,cutoff,end,role,kind,model,q,prepared=None):
    def fitted(level):return prepared[level] if prepared is not None else fit_frontier(train,model,level,cutoff=cutoff)
    fit=fitted(q);lo=fitted(.05);hi=fitted(.95)
    p,l,u=fit.predict(test),lo.predict(test),hi.predict(test)
    rows=[]
    for i,r in enumerate(test.itertuples()):
        rows.append({'cutoff':str(pd.Timestamp(cutoff).date()),'end':str(pd.Timestamp(end).date()),'role':role,
                     'input_mode':kind,'model':model,'quantile':q,'repo':r.repo,'org':r.org,'date':str(r.date.date()),
                     'group':r.group,'N_used':float(r.N),'actual':float(r.S),'prediction':float(p[i]),
                     'pi_low':float(min(l[i],u[i])),'pi_high':float(max(l[i],u[i])),'interval_crossed':bool(l[i]>u[i]),
                     'fallback':fit.fallback,'interval_fallback':lo.fallback+'|'+hi.fallback if lo.fallback or hi.fallback else '',
                     'train_rows':len(train),'train_date_max':str(train.date.max().date()),
                     'train_pubdate_max':str(train.pubdate.max().date())})
    return rows


def backtest(c1,c4,matches,strict):
    rows=[];folds=[]
    for cutoff in CUTOFFS:
        train,_=asof(c1,matches,cutoff,strict=True)
        for months in HORIZON_MONTHS:
            end=pd.Timestamp(cutoff)+pd.DateOffset(months=months)
            if end>pd.Timestamp(ORIGIN):continue
            all_end,_=asof(c1,matches,end,strict=True)
            # Seen includes all identities, even rows that were ineligible at the cutoff.
            seen=set(c1.loc[c1.date.le(pd.Timestamp(cutoff)),'repo'])
            test=all_end[all_end.date.gt(pd.Timestamp(cutoff))&~all_end.repo.isin(seen)].copy()
            role='development' if end<=pd.Timestamp(SELECTION_CUTOFF) else ('holdout' if pd.Timestamp(cutoff)>=pd.Timestamp(SELECTION_CUTOFF) else 'straddling_diagnostic')
            status='ok' if len(train)>=10 and len(test)>0 else 'insufficient_rows'
            folds.append({'cutoff':cutoff,'end':str(end.date()),'months':months,'role':role,'train':len(train),'test':len(test),'status':status,
                          'overlapping_repos':len(set(train.repo)&set(test.repo))})
            if status!='ok':continue
            projected,compute=project_test_n(train,test,c4,cutoff)
            folds[-1].update(compute)
            prepared={m:{q:fit_frontier(train,m,q,cutoff=cutoff) for q in [.05,.85,.9,.95]} for m in FRONTIER_MODELS}
            for mode,frame in [('resource_forecast',projected),('observed_N_diagnostic',test)]:
                for model in FRONTIER_MODELS:
                    for q in QUANTILES:rows.extend(evaluate_fold(train,frame,cutoff,end,role,mode,model,q,prepared[model]))
    df=pd.DataFrame(rows);save_table(df,'rolling_predictions.csv');save_table(pd.DataFrame(folds),'rolling_folds.csv')
    metrics=metric_table(df,['role','input_mode','model','quantile']);save_table(metrics,'rolling_metrics.csv')
    dev=metrics[metrics.role.eq('development')&metrics.input_mode.eq('resource_forecast')&metrics['quantile'].eq(.85)].copy()
    if len(dev)!=len(FRONTIER_MODELS):raise ValueError('Missing candidate development evaluation')
    dev['order']=dev.model.map({x:i for i,x in enumerate(FRONTIER_MODELS)})
    selected=str(dev.sort_values(['pinball','order']).iloc[0].model)
    selection={'selected':selected,'selection_rule':'development resource-forecast pinball at q=.85',
               'training_targets_end_no_later_than':SELECTION_CUTOFF,'holdout_used_for_selection':False,
               'selection_is_retrospective':True,'development_metrics':dev.drop(columns='order').to_dict('records')}
    # Lock the choice before computing any reported holdout improvement interval.
    write_json(OUT/'frontier_selection.json',selection)
    compare=[];rng=np.random.default_rng(SEED+31)
    h=df[df.role.eq('holdout')&df.input_mode.eq('resource_forecast')&df['quantile'].eq(.85)]
    for baseline in ['type_empirical_180d','size_time_linear_quantile']:
        keys=['cutoff','end','repo','org']
        a=h[h.model.eq(selected)].set_index(keys);b=h[h.model.eq(baseline)].set_index(keys).reindex(a.index)
        gain=pinball(a.actual,b.prediction,.85)-pinball(a.actual,a.prediction,.85)
        organizations=a.index.get_level_values('org').to_numpy();unique=np.unique(organizations);boot=[]
        for _ in range(1000):
            w=cluster_counts(organizations,rng.choice(unique,len(unique),replace=True));boot.append(float(np.average(gain,weights=w)))
        compare.append({'selected':selected,'baseline':baseline,'positive_means_selected_better':True,'pinball_gain':float(np.mean(gain)),
                        'gain_90CI_low':float(np.quantile(boot,.05)),'gain_90CI_high':float(np.quantile(boot,.95)),
                        'holdout_predictions':len(a),'distinct_repositories':a.index.get_level_values('repo').nunique(),
                        'publisher_clusters':len(unique),'note':'overlapping forecast windows clustered by publisher; small-sample diagnostic'})
    save_table(pd.DataFrame(compare),'holdout_paired_comparison.csv')
    orgrows=[]
    for org in sorted(strict.org.unique()):
        train=strict[strict.org.ne(org)];test=strict[strict.org.eq(org)]
        for model in FRONTIER_MODELS:
            records=evaluate_fold(train,test,ORIGIN,ORIGIN,'publisher_holdout','observed_N_diagnostic',model,.85)
            for r in records:r['left_out_publisher']=org
            orgrows.extend(records)
    odf=pd.DataFrame(orgrows);save_table(odf,'publisher_holdout_predictions.csv')
    save_table(metric_table(odf,['model','quantile']),'publisher_holdout_metrics.csv')
    return selection,metrics,pd.DataFrame(compare)
