"""Shared publisher resampling and training-FLOP-consistent Q3 scenarios."""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import paths
from data_config import OUT,ORIGIN,SEED,QUANTILES,PROTOCOL
from q4_data import compute_asof,asof,save_table,write_json
from q4_frontier import compute_fit,fit_frontier,reference_n
from q4_kernels import cluster_counts,logit_to_score


def decayed_time(h,rho):return h if rho==0 else -math.expm1(-rho*h)/rho


def predict_path(fit,group,logn,h,rho):
    if fit.coef is None:return float(fit.empirical.get(group,fit.empirical['all']))
    x=[1.]
    if 'size_time' in fit.model:x.append(logn)
    x.extend([decayed_time(h,rho),float(group=='posttrained')])
    z=float(np.dot(x,fit.coef))
    return float(logit_to_score(z)) if fit.model.startswith('bounded') else z


def q3_path(inputs,params,g,h,growth_factor,cost):
    sys.path.insert(0,str(paths.CODE/'q3'))
    from q3_core import Solver
    solver=Solver(params,8192,cost,inputs.q0)
    c0=1e22;base=solver.solve(c0);f0=c0*base['training_share']
    logtarget=math.log10(f0)+growth_factor*g*h
    def residual(logc):
        c=10.**logc;s=solver.solve(c)
        return math.log10(c*s['training_share'])-logtarget
    logc=brentq(residual,min(22.,logtarget)-6,max(22.,logtarget)+6,xtol=2e-12)
    sol=solver.solve(10.**logc)
    mismatch=abs(math.log10(sol['budget_FLOPs']*sol['training_share'])-logtarget)
    if mismatch>1e-9:raise ArithmeticError('Q3 training-FLOP target was not matched')
    return {'conversion':'q3_'+cost,'growth_factor':growth_factor,'horizon_years':h,'g':g,
            'delta_logN':math.log10(sol['N_B']/base['N_B']),'N_multiplier':sol['N_B']/base['N_B'],
            'D_multiplier':sol['D_B']/base['D_B'],'training_FLOPs_multiplier':10**(growth_factor*g*h),
            'baseline_N_B':base['N_B'],'baseline_D_B':base['D_B'],'baseline_Q':base['Q_star'],
            'baseline_total_FLOPs':c0,'baseline_training_FLOPs':f0,
            'total_budget_multiplier':sol['budget_FLOPs']/c0,'target_training_FLOPs':10**logtarget,
            'future_total_FLOPs':sol['budget_FLOPs'],'future_N_B':sol['N_B'],'future_D_B':sol['D_B'],'future_Q':sol['Q_star'],
            'training_share':sol['training_share'],'quality_share':sol['quality_share'],'attention_share':sol['attention_share'],
            'training_target_log10_residual':mismatch,'budget_relative_residual':abs(sol['share_sum']-1),
            'Q0':inputs.q0,'context_tokens':8192,'psi':0.,'lambda_p':0.,
            'inside_B6_N_range':bool(inputs.support['N_B'][0]<=sol['N_B']<=inputs.support['N_B'][1]),
            'inside_B6_D_range':bool(inputs.support['D_B'][0]<=sol['D_B']<=inputs.support['D_B'][1]),
            'inside_B6_Q_range':bool(inputs.support['Q'][0]<=sol['Q_star']<=inputs.support['Q'][1])}


def resource_paths(inputs,params,compute,growths):
    rows=[]
    for gf in growths:
        for h in [1,2]:
            rows.extend([{'conversion':'fitted_relation','growth_factor':gf,'horizon_years':h,
                          'delta_logN':(gf*compute['g']-compute['xi'])*h/compute['kappa']},
                         {'conversion':'fixed_training_data','growth_factor':gf,'horizon_years':h,'delta_logN':gf*compute['g']*h}])
            for cost in PROTOCOL['q3_costs']:rows.append(q3_path(inputs,params,compute['g'],h,gf,cost))
    return rows


def forecast_records(train,models,fits,refs,paths,decays,draw_id):
    rows=[]
    for path in paths:
        for model in models:
            for q in QUANTILES:
                fit=fits[(model,q)]
                for group in ['pretrained','posttrained']:
                    logn=math.log10(refs[group])+path['delta_logN'];h=path['horizon_years']
                    for rho in decays:
                        pred=predict_path(fit,group,logn,h,rho)
                        rec={'draw_id':draw_id,'model':model,'quantile':q,'group':group,'conversion':path['conversion'],
                             'growth_factor':path['growth_factor'],'horizon_years':h,'trend_decay':rho,'N_reference_B':refs[group],
                             'delta_logN':path['delta_logN'],'log10_future_N_B':logn,'prediction':pred,
                             'score_outside_0_100':bool(pred<0 or pred>100),'fallback':fit.fallback,
                             'N_extrapolates_training':bool(logn<train.logN.min() or logn>train.logN.max()),
                             'forecast_date':str((pd.Timestamp(ORIGIN)+pd.DateOffset(years=h)).date())}
                        if (model,.05) in fits:
                            lo=predict_path(fits[(model,.05)],group,logn,h,rho);hi=predict_path(fits[(model,.95)],group,logn,h,rho)
                            rec.update(individual_PI90_low=min(lo,hi),individual_PI90_high=max(lo,hi),individual_interval_crossed=bool(lo>hi))
                        rows.append(rec)
    return rows


def joint_forecasts(c4,strict,inputs,selected):
    models=list(dict.fromkeys([selected,'size_time_linear_quantile','bounded_size_time_quantile']))
    comp,annual=compute_fit(c4);save_table(annual,'c4_language_annual_frontier.csv')
    fits={(m,q):fit_frontier(strict,m,q) for m in models for q in [.05,.85,.90,.95]}
    refs={g:reference_n(strict,g) for g in ['pretrained','posttrained']}
    paths=resource_paths(inputs,inputs.params,comp,PROTOCOL['compute_growth_factors'])
    save_table(pd.DataFrame(paths),'resource_paths_point.csv')
    point=pd.DataFrame(forecast_records(strict,models,fits,refs,paths,PROTOCOL['trend_decay_scenarios'],-1))
    save_table(point,'forecast_structural_scenarios.csv')
    parameters=[]
    for (m,q),f in fits.items():
        parameters.append({'model':m,'quantile':q,'coefficients':None if f.coef is None else f.coef.tolist(),
                           'empirical':f.empirical,'fallback':f.fallback,'diagnostics':f.diagnostics})
    write_json(OUT/'frontier_full_fit.json',{'reference_N_B':refs,'compute':comp,'models':parameters})
    available=compute_asof(c4,ORIGIN);universe=np.unique(np.r_[strict.org.to_numpy(),available.org.to_numpy()])
    rng=np.random.default_rng(SEED+77);drawrows=[];coeffrows=[];failures=[];resource_rows=[];weight_rows=[]
    for draw in range(PROTOCOL['joint_bootstrap_draws']):
        sample=rng.choice(universe,len(universe),replace=True)
        wc1=cluster_counts(strict.org,sample);wc4=cluster_counts(available.org,sample)
        wc4map=dict(zip(available.c4_id,wc4))
        weight_rows.extend({'draw_id':draw,'publisher':org,'count':int(n)} for org,n in zip(*np.unique(sample,return_counts=True)))
        try:
            compute,_=compute_fit(c4,counts=wc4map)
            ff={(m,q):fit_frontier(strict,m,q,wc1) for m in models for q in QUANTILES}
            ref={g:reference_n(strict,g,wc1) for g in ['pretrained','posttrained']}
            q3id=draw%len(inputs.draws)
            resources=resource_paths(inputs,inputs.draws[q3id],compute,[1.])
            for r in resources:r.update(draw_id=draw,q3_parameter_draw_id=q3id)
            current=forecast_records(strict,models,ff,ref,resources,[0.],draw)
            # Append only after the entire shared draw succeeds.
            drawrows.extend(current);resource_rows.extend(resources)
            coeffrows.append({'draw_id':draw,**compute,'q3_parameter_draw_id':q3id,
                              'C1_distinct_rows':int((wc1>0).sum()),'C1_replicated_rows':int(wc1.sum()),
                              'Nref_pretrained_B':ref['pretrained'],'Nref_posttrained_B':ref['posttrained'],
                              'frontier_fallback_fits':sum(bool(f.fallback) for f in ff.values()),
                              **{f'{m}_q{q}_coef':json.dumps(None if f.coef is None else f.coef.tolist()) for (m,q),f in ff.items()}})
        except (ValueError,RuntimeError,ArithmeticError,OverflowError) as err:
            failures.append({'draw_id':draw,'error':type(err).__name__,'message':str(err)})
    cf=pd.DataFrame(coeffrows);df=pd.DataFrame(drawrows)
    if len(cf)<100:raise RuntimeError('Fewer than half of the shared bootstrap attempts succeeded')
    save_table(cf,'joint_coefficients.csv');save_table(df,'joint_forecast_draws.csv');save_table(pd.DataFrame(resource_rows),'joint_resource_paths.csv')
    save_table(pd.DataFrame(weight_rows),'joint_publisher_counts.csv')
    save_table(pd.DataFrame(failures,columns=['draw_id','error','message']),'joint_failed_draws.csv')
    keys=['model','quantile','group','conversion','growth_factor','horizon_years','trend_decay']
    interval=df.groupby(keys).prediction.agg(median='median',curve_CI90_low=lambda v:v.quantile(.05),curve_CI90_high=lambda v:v.quantile(.95),draws='size').reset_index()
    comparable=point[point.growth_factor.eq(1)&point.trend_decay.eq(0)]
    interval=interval.merge(comparable,on=keys,how='left',validate='one_to_one')
    save_table(interval,'forecast_intervals.csv')
    corr=cf[['g','kappa','xi']].corr();save_table(corr.rename_axis('parameter').reset_index(),'joint_compute_correlation.csv')
    # Diagnostic of covariance loss, with exactly the same sampled marginals.
    perm=np.random.default_rng(SEED+78);ind_k=perm.permutation(cf.kappa);ind_x=perm.permutation(cf.xi)
    joint=(cf.g-cf.xi)/cf.kappa;independent=(cf.g.to_numpy()-ind_x)/ind_k
    covariance={'joint_delta_logN_one_year_sd':float(np.std(joint,ddof=1)),
                'independent_permuted_delta_logN_one_year_sd':float(np.std(independent,ddof=1)),
                'joint_90_range':np.quantile(joint,[.05,.95]).tolist(),'independent_90_range':np.quantile(independent,[.05,.95]).tolist(),
                'interpretation':'dependence diagnostic, not an accuracy test or proof of interval calibration'}
    summary={'compute_primary':comp,'reference_N_B':refs,'selected_model':selected,'scenario_models':models,
             'requested_joint_draws':PROTOCOL['joint_bootstrap_draws'],'successful_joint_draws':len(cf),'failed_joint_draws':len(failures),
             'shared_publisher_universe':len(universe),'C1_C4_shared_publishers':len(set(strict.org)&set(available.org)),
             'forecast_curve_rows':len(interval),'point_structural_scenarios':len(point),'joint_path_rows':len(resource_rows),
             'raw_score_outside_0_100_point_scenarios':int(point.score_outside_0_100.sum()),
             'bootstrap_frontier_fallback_fits':int(cf.frontier_fallback_fits.sum()),'covariance_diagnostic':covariance,
             'interval_targets':'bootstrap CI of the fitted conditional quantile curve; separate q05-q95 individual band uncalibrated',
             'draw_alignment':'same publisher draw across C1/C4, all quantiles/types/horizons/conversions; independent B6 source paired by draw index',
             'unknown_dependence':'different developer ids / organization aliases are not inferred; C1-C4 linkage is partial',
             'Q3_training_target_max_log10_error':float(pd.DataFrame(resource_rows).training_target_log10_residual.max()),
             'Q3_budget_max_relative_error':float(pd.DataFrame(resource_rows).budget_relative_residual.max())}
    write_json(OUT/'uncertainty_summary.json',summary)
    return summary


def structural_sensitivity(c1,c4,matches,strict,selected):
    rows=[]
    delayed,_=asof(c1,matches,ORIGIN,strict=True,keep_date_conflicts=True)
    for sample,train in [('strict_primary',strict),('date_conflicts_max_date',delayed)]:
        for domain,lang in [('Language',True),('all_domains',False)]:
            compute,_=compute_fit(c4,language_only=lang)
            for model in list(dict.fromkeys([selected,'size_time_linear_quantile','bounded_size_time_quantile'])):
                for q in QUANTILES:
                    f=fit_frontier(train,model,q)
                    for group in ['pretrained','posttrained']:
                        ref=reference_n(train,group)
                        for h in [1,2]:
                            dn=(compute['g']-compute['xi'])/compute['kappa']*h
                            rows.append({'sample':sample,'C4_domain':domain,'n':len(train),'model':model,'quantile':q,'group':group,
                                         'horizon_years':h,**compute,'prediction':predict_path(f,group,math.log10(ref)+dn,h,0),
                                         'delta_logN':dn,'fallback':f.fallback})
    save_table(pd.DataFrame(rows),'structural_sensitivity.csv')
