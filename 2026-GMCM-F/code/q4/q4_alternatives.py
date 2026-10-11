"""Exploratory extension: auxiliary trend estimation and ordered strict calibration.

This is a second-stage analysis designed after the first holdout was inspected.
Its reused holdout is NOT a new independent confirmation, even though each fit
still obeys chronological and publisher exclusion rules.
"""
from __future__ import annotations
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import paths
from data_config import OUT,ORIGIN,SELECTION_CUTOFF,CUTOFFS,HORIZON_MONTHS,SEED
from q4_data import asof,compute_asof,write_json
from q4_frontier import project_test_n,reference_n,compute_fit,metric_table
from q4_kernels import score_to_logit,logit_to_score,weighted_quantile,cluster_counts,pinball
from q4_uncertainty import resource_paths,decayed_time


EXT=paths.VALIDATION/'q4_exploratory'
MODELS=['auxiliary_time_ordered','auxiliary_size_time_ordered']
QS=[.05,.85,.90,.95]
PROTOCOL={
 'stage':'exploratory second-stage extension requested after first-stage results',
 'first_stage_selection_and_outputs_unchanged':True,'independent_unseen_holdout':False,
 'reason':'strict sample is small; test training-only trend pooling without adding uncertain identities to calibration',
 'candidates':MODELS,'auxiliary_training':'licensed nonconflicting pre/post models available at cutoff; all strict repository identities excluded',
 'auxiliary_source_limit':'licence alone does not establish C4-confirmed open weights; used only for auxiliary trend, never to enlarge strict validation sample',
 'auxiliary_weight':'each publisher total weight 1; within-publisher rows equal weight',
 'trend_fit':'logit score ridge regression via augmented weighted SVD; intercept unpenalized; standardized slopes penalty 10',
 'ridge_penalty':10.,'calibration':'all strict training rows, type-specific ordered empirical quantiles of logit residuals',
 'calibration_fallback':'if fewer than 3 distinct positively weighted rows of a type, pool strict residuals',
 'crossing_prevention':'same trend coefficients for every quantile; only ordered residual offsets change',
 'validation':'same rolling and publisher holdouts as stage one; reused outcomes labelled exploratory',
 'development_selection':'minimum q=.85 resource-forecast pinball among only the two new candidates, development targets <=2024-12-13',
 'joint_draws':200,'draw_unit':'union of auxiliary publishers, strict publishers and C4 publishers',
 'source_alignment':'shared publisher count used for auxiliary fitting, strict calibration, reference N and C4 fits; Q2 B6 joint rows paired by id',
 'promotion':'no replacement of the original frozen choice based on reused holdout results','seed':SEED+212,
}


def save(frame,name):
    EXT.mkdir(parents=True,exist_ok=True);frame.to_csv(EXT/name,index=False,encoding='utf-8-sig')


def xdesign(frame,model):
    parts=[np.ones(len(frame))]
    if 'size_time' in model:parts.append(frame.logN.to_numpy(float))
    parts.extend([frame.tau.to_numpy(float),frame.group.eq('posttrained').to_numpy(float)])
    return np.column_stack(parts)


@dataclass
class TransferFit:
    model:str
    coef:np.ndarray
    offsets:dict
    diagnostics:dict

    def predict(self,frame,q):
        baseline=xdesign(frame,self.model)@self.coef
        offset=np.array([self.offsets.get(g,self.offsets['all'])[q] for g in frame.group])
        return logit_to_score(baseline+offset)

    def predict_path(self,group,logn,h,rho,q):
        x=[1.]
        if 'size_time' in self.model:x.append(logn)
        x.extend([decayed_time(h,rho),float(group=='posttrained')])
        return float(logit_to_score(np.dot(x,self.coef)+self.offsets.get(group,self.offsets['all'])[q]))


def auxiliary_sample(broad,strict_identities):
    return broad[broad.group.isin(['pretrained','posttrained'])&~broad.repo.isin(strict_identities)].reset_index(drop=True)


def fit_transfer(auxiliary,strict,model,aux_counts=None,strict_counts=None):
    wa=np.ones(len(auxiliary),int) if aux_counts is None else np.asarray(aux_counts,int)
    ws=np.ones(len(strict),int) if strict_counts is None else np.asarray(strict_counts,int)
    if (wa>0).sum()<10 or (ws>0).sum()<3:raise ValueError('Too few weighted auxiliary or calibration observations')
    # Retaining publisher frequencies exactly corresponds to repeatedly drawing
    # publishers with one unit of mass each, not treating 500 derivatives as 500 publishers.
    counts=auxiliary.groupby('org').org.transform('size').to_numpy(float)
    weights=wa/counts
    x=xdesign(auxiliary,model);y=score_to_logit(auxiliary.S)
    center=np.average(x,axis=0,weights=weights);center[0]=0.
    scale=np.sqrt(np.average((x-center)**2,axis=0,weights=weights));scale[0]=1.;scale[scale<1e-12]=1.
    z=(x-center)/scale;p=x.shape[1]
    # Augmented SVD is stable even when one type is absent from a bootstrap draw.
    penalty=np.diag([0.]+[math.sqrt(PROTOCOL['ridge_penalty'])]*(p-1))
    sw=np.sqrt(weights);augx=np.vstack([z*sw[:,None],penalty]);augy=np.r_[y*sw,np.zeros(p)]
    standardized=np.linalg.lstsq(augx,augy,rcond=None)[0]
    coef=standardized/scale;coef[0]-=float(center@coef)
    residual=score_to_logit(strict.S)-xdesign(strict,model)@coef
    offsets={'all':{q:weighted_quantile(residual,q,ws) for q in QS}}
    pooled_types=[]
    for group in ['pretrained','posttrained']:
        mask=strict.group.eq(group).to_numpy()
        if ((ws>0)&mask).sum()<3:offsets[group]=offsets['all'].copy();pooled_types.append(group)
        else:offsets[group]={q:weighted_quantile(residual[mask],q,ws[mask]) for q in QS}
    diagnostics={'auxiliary_rows':int((wa>0).sum()),'auxiliary_publishers':auxiliary.loc[wa>0,'org'].nunique(),
                 'calibration_rows':int((ws>0).sum()),'auxiliary_weight_sum':float(weights.sum()),'pooled_types':pooled_types,
                 'center':center.tolist(),'scale':scale.tolist(),'ridge_penalty':PROTOCOL['ridge_penalty'],
                 'fitted_residual_SSE':float(np.sum(weights*(y-x@coef)**2))}
    return TransferFit(model,coef,offsets,diagnostics)


def records(fit,test,cutoff,end,role,mode,train,aux):
    rows=[];lo=fit.predict(test,.05);hi=fit.predict(test,.95)
    for q in [.85,.90]:
        p=fit.predict(test,q)
        for i,r in enumerate(test.itertuples()):
            rows.append({'cutoff':str(pd.Timestamp(cutoff).date()),'end':str(pd.Timestamp(end).date()),'role':role,
                         'input_mode':mode,'model':fit.model,'quantile':q,'repo':r.repo,'org':r.org,'date':str(r.date.date()),
                         'group':r.group,'N_used':r.N,'actual':r.S,'prediction':p[i],'pi_low':lo[i],'pi_high':hi[i],
                         'interval_crossed':bool(lo[i]>hi[i]),'fallback':'','train_rows':len(train),'auxiliary_rows':len(aux),
                         'train_date_max':str(train.date.max().date()),'auxiliary_date_max':str(aux.date.max().date()),
                         'exploratory_reused_holdout':role=='holdout'})
    return rows


def evaluate(c1,c4,matches,strict):
    rows=[];audits=[]
    # Prevent any known strict identity from doubling as an auxiliary identity,
    # but only use strict identities available at the current cutoff.
    for cutoff in CUTOFFS:
        train,_=asof(c1,matches,cutoff,strict=True);broad,_=asof(c1,matches,cutoff)
        aux=auxiliary_sample(broad,set(train.repo));fits={m:fit_transfer(aux,train,m) for m in MODELS}
        for months in HORIZON_MONTHS:
            end=pd.Timestamp(cutoff)+pd.DateOffset(months=months)
            if end>pd.Timestamp(ORIGIN):continue
            all_end,_=asof(c1,matches,end,strict=True);seen=set(c1.loc[c1.date.le(pd.Timestamp(cutoff)),'repo'])
            test=all_end[all_end.date.gt(pd.Timestamp(cutoff))&~all_end.repo.isin(seen)]
            if len(train)<10 or not len(test):continue
            role='development' if end<=pd.Timestamp(SELECTION_CUTOFF) else ('holdout' if pd.Timestamp(cutoff)>=pd.Timestamp(SELECTION_CUTOFF) else 'straddling_diagnostic')
            projected,_=project_test_n(train,test,c4,cutoff)
            audits.append({'cutoff':cutoff,'end':str(end.date()),'train':len(train),'auxiliary':len(aux),'test':len(test),
                           'auxiliary_strict_overlap':len(set(aux.repo)&set(train.repo)),
                           'auxiliary_test_overlap':len(set(aux.repo)&set(test.repo)),
                           'auxiliary_latest_date':str(aux.date.max().date())})
            for mode,frame in [('resource_forecast',projected),('observed_N_diagnostic',test)]:
                for m in MODELS:rows.extend(records(fits[m],frame,cutoff,end,role,mode,train,aux))
    new=pd.DataFrame(rows);save(new,'rolling_predictions.csv');save(pd.DataFrame(audits),'rolling_input_audit.csv')
    baseline=pd.read_csv(OUT/'rolling_predictions.csv').fillna({'fallback':''})
    combined=pd.concat([baseline,new],ignore_index=True)
    metrics=metric_table(combined,['role','input_mode','model','quantile']);save(metrics,'rolling_metrics_with_baselines.csv')
    dev=metrics[metrics.role.eq('development')&metrics.input_mode.eq('resource_forecast')&metrics['quantile'].eq(.85)&metrics.model.isin(MODELS)]
    selected=min(MODELS,key=lambda m:float(dev[dev.model.eq(m)].pinball.iloc[0]))
    selection={'selected_exploratory':selected,'original_selection_unchanged':True,'independent_holdout':False,
               'rule':PROTOCOL['development_selection'],'development':dev.to_dict('records')}
    write_json(EXT/'selection.json',selection)
    broad,_=asof(c1,matches,ORIGIN);aux=auxiliary_sample(broad,set(strict.repo));orgrows=[];org_audits=[]
    for org in sorted(strict.org.unique()):
        train=strict[strict.org.ne(org)];test=strict[strict.org.eq(org)];a=aux[aux.org.ne(org)]
        org_audits.append({'held_out_publisher':org,'strict_training_same_org':int(train.org.eq(org).sum()),
                           'auxiliary_same_org':int(a.org.eq(org).sum()),'train_rows':len(train),'aux_rows':len(a),'test_rows':len(test)})
        for m in MODELS:
            rr=records(fit_transfer(a,train,m),test,ORIGIN,ORIGIN,'publisher_holdout','observed_N_diagnostic',train,a)
            for z in rr:z['left_out_publisher']=org
            orgrows.extend(rr)
    odf=pd.DataFrame(orgrows);save(odf,'publisher_holdout_predictions.csv');save(pd.DataFrame(org_audits),'publisher_input_audit.csv')
    orig=pd.read_csv(OUT/'publisher_holdout_predictions.csv').fillna({'fallback':''})
    save(metric_table(pd.concat([orig,odf],ignore_index=True),['model','quantile']),'publisher_holdout_metrics.csv')
    # Same target rows, same groups; reused outcomes are not a fresh holdout.
    h=combined[combined.role.eq('holdout')&combined.input_mode.eq('resource_forecast')&combined['quantile'].eq(.85)]
    comparisons=[];rng=np.random.default_rng(PROTOCOL['seed'])
    keys=['cutoff','end','repo','org']
    for m in MODELS:
        a=h[h.model.eq(m)].set_index(keys)
        for baseline in ['type_empirical_180d','size_time_linear_quantile','bounded_size_time_quantile']:
            b=h[h.model.eq(baseline)].set_index(keys).reindex(a.index)
            gain=pinball(a.actual,b.prediction,.85)-pinball(a.actual,a.prediction,.85);orgs=a.index.get_level_values('org').to_numpy();unique=np.unique(orgs)
            draws=[]
            for _ in range(1000):
                w=cluster_counts(orgs,rng.choice(unique,len(unique),replace=True));draws.append(float(np.average(gain,weights=w)))
            comparisons.append({'model':m,'baseline':baseline,'gain':float(gain.mean()),'gain_90CI_low':float(np.quantile(draws,.05)),
                                'gain_90CI_high':float(np.quantile(draws,.95)),'relative_pinball_reduction':float(gain.mean()/pinball(a.actual,b.prediction,.85).mean()),
                                'predictions':len(a),'distinct_repositories':a.index.get_level_values('repo').nunique(),
                                'independent_validation':False,'selected_on_development':m==selected})
    save(pd.DataFrame(comparisons),'paired_exploratory_comparison.csv')
    return selected,metrics,aux,pd.DataFrame(comparisons)


def joint_extension(c4,strict,aux,inputs,selected):
    comp,_=compute_fit(c4);fits={m:fit_transfer(aux,strict,m) for m in MODELS}
    parameters={m:{'coefficients':f.coef.tolist(),'ordered_offsets':f.offsets,'diagnostics':f.diagnostics} for m,f in fits.items()}
    write_json(EXT/'fitted_models.json',{'selected_exploratory':selected,'models':parameters})
    refs={g:reference_n(strict,g) for g in ['pretrained','posttrained']}
    paths=resource_paths(inputs,inputs.params,comp,[0.,.5,1.]);point=[]
    def forecast(fit,reference,path,draw,q,rho,group):
        h=path['horizon_years'];logn=math.log10(reference)+path['delta_logN']
        return {'draw_id':draw,'model':fit.model,'quantile':q,'group':group,'conversion':path['conversion'],
                'growth_factor':path['growth_factor'],'horizon_years':h,'trend_decay':rho,
                'prediction':fit.predict_path(group,logn,h,rho,q),'PI90_low':fit.predict_path(group,logn,h,rho,.05),
                'PI90_high':fit.predict_path(group,logn,h,rho,.95),'log10_N_B':logn,'N_reference_B':reference,
                'note':'exploratory conditional scenario; not independently validated'}
    for path in paths:
        for fit in fits.values():
            for q in [.85,.9]:
                for group in refs:
                    for rho in [0.,.5,1.]:point.append(forecast(fit,refs[group],path,-1,q,rho,group))
    save(pd.DataFrame(point),'forecast_scenarios.csv')
    c4available=compute_asof(c4,ORIGIN);universe=np.unique(np.r_[aux.org,strict.org,c4available.org])
    rng=np.random.default_rng(PROTOCOL['seed']+1);draws=[];coefs=[];failed=[];weights=[]
    for draw in range(PROTOCOL['joint_draws']):
        sample=rng.choice(universe,len(universe),replace=True)
        wa=cluster_counts(aux.org,sample);ws=cluster_counts(strict.org,sample);wc=cluster_counts(c4available.org,sample)
        weights.extend({'draw_id':draw,'publisher':org,'count':int(n)} for org,n in zip(*np.unique(sample,return_counts=True)))
        try:
            comp,_=compute_fit(c4,counts=dict(zip(c4available.c4_id,wc)))
            f=fit_transfer(aux,strict,selected,wa,ws)
            ref={g:reference_n(strict,g,ws) for g in ['pretrained','posttrained']}
            paths=resource_paths(inputs,inputs.draws[draw%len(inputs.draws)],comp,[1.]);rr=[]
            for path in paths:
                for q in [.85,.9]:
                    for group in ref:rr.append(forecast(f,ref[group],path,draw,q,0.,group))
            draws.extend(rr);coefs.append({'draw_id':draw,**comp,'coef':json.dumps(f.coef.tolist()),'offsets':json.dumps(f.offsets),
                                         'auxiliary_rows':f.diagnostics['auxiliary_rows'],'strict_rows':f.diagnostics['calibration_rows'],
                                         'q3_parameter_draw_id':draw%len(inputs.draws),'pooled_types':'|'.join(f.diagnostics['pooled_types'])})
        except (ValueError,ArithmeticError,RuntimeError) as err:failed.append({'draw_id':draw,'message':str(err)})
    df=pd.DataFrame(draws);save(df,'joint_forecast_draws.csv');save(pd.DataFrame(coefs),'joint_coefficients.csv')
    save(pd.DataFrame(weights),'joint_publisher_counts.csv');save(pd.DataFrame(failed,columns=['draw_id','message']),'failed_draws.csv')
    keys=['model','quantile','group','conversion','growth_factor','horizon_years','trend_decay']
    ci=df.groupby(keys).prediction.agg(curve_CI90_low=lambda x:x.quantile(.05),curve_CI90_high=lambda x:x.quantile(.95),draws='size').reset_index()
    pts=pd.DataFrame(point);pts=pts[pts.growth_factor.eq(1)&pts.trend_decay.eq(0)&pts.model.eq(selected)]
    save(ci.merge(pts,on=keys,validate='one_to_one'),'forecast_intervals.csv')
    return {'joint_draws_requested':200,'joint_draws_successful':len(coefs),'failed_draws':len(failed),'universe_publishers':len(universe),
            'forecast_rows':len(ci),'scenario_rows':len(point),'auxiliary_rows':len(aux),'auxiliary_publishers':aux.org.nunique(),
            'crossing_rows':int(df.PI90_low.gt(df.PI90_high).sum()),'scores_outside_0_100':int((df.prediction.lt(0)|df.prediction.gt(100)).sum())}

