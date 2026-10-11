"""Q3 training-FLOPs paths and joint publisher/parameter uncertainty."""
import json,math,time,sys,importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import paths
sys.path.insert(0,str(paths.CODE/'q3'))
from protocol import OUT,ORIGIN,GROUPS,PROTOCOL,save,write_json
from ordered_model import fit_joint
from q4_data import read_sources,identity_table,asof,compute_asof
from q4_frontier import compute_fit,reference_n
from q4_alternatives import auxiliary_sample
from q4_kernels import cluster_counts

spec=importlib.util.spec_from_file_location('q3_theory_core',paths.CODE/'q3'/'core.py')
q3=importlib.util.module_from_spec(spec);sys.modules[spec.name]=q3;spec.loader.exec_module(q3)

def upstream():
    """Verify the shipped Q2/Q3 handoffs and return the shared Q3 inputs."""
    handoff=paths.Q3/'q3_handoff_v26.json';q2path=paths.Q2/'q2_handoff_v26.json'
    draws_path=paths.Q2/'joint_parameter_draws.csv';manifest_path=paths.Q3/'run_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    h=json.loads(handoff.read_text(encoding='utf-8'))
    q2h=json.loads(q2path.read_text(encoding='utf-8'))
    for name,sha in manifest['outputs'].items():
        if paths.digest(paths.Q3/name)!=sha:raise ValueError(f'Q3 output mismatch: {name}')
    if paths.digest(handoff)!=manifest['outputs']['q3_handoff_v26.json']:raise ValueError('Q3 handoff changed')
    assert h['schema']=='q3-curved-budget-v26/1' and q2h['selected_model']=='curved'
    assert all(h['params'][k]==q2h['parameters'][k] for k in h['params'])
    draws=pd.read_csv(draws_path);draws=draws[draws.model.eq('curved')].sort_values('draw')
    assert len(draws)==200 and draws.draw.tolist()==list(range(200))
    params=[q3.Params.from_dict(r) for r in draws.to_dict('records')]
    contract=dict(Q3_verified_files=len(manifest['outputs']),Q3_verification='PASS',Q3_schema=h['schema'],
        Q3_manifest_sha256=paths.digest(manifest_path),Q2_handoff_sha256=paths.digest(q2path),
        Q2_draws_sha256=paths.digest(draws_path),Q0=h['Q0'],parameters=h['params'],
        source_support=q2h['support'],score_bridge='B1 real Pythia only; B6 Loss is not mapped to benchmark scores')
    write_json(OUT/'upstream_contract.json',contract)
    return q3.Params.from_dict(h['params']),params,h['Q0'],q2h

class Paths:
    def __init__(self,p,q0):
        self.p=p;self.q0=q0;self.problems={k:q3.Problem([p],8192,k,q0) for k in ['exponential','power','logarithmic']}
        self.base={k:v.solve(1e22,tol=1e-7) for k,v in self.problems.items()}

    def quality_path(self,g,h,gf,kind):
        prob=self.problems[kind];base=self.base[kind];p=self.p
        f0=6e18*base['N_B']*base['D_B'];target=f0*10**(gf*g*h);t=target/1e18
        k=prob.k;cost=prob.cost.h(1.)
        # On a Q=1 branch, the fixed-training-FLOP scale equation is monotone.
        # The candidate is always checked by the global Q3 solver afterwards.
        const=math.log(p.alpha*p.A/(p.beta*p.B*k))+p.beta*math.log(t/6)
        def equation(z):return (p.alpha+p.beta+1)*z-math.log(k*math.exp(z)+cost)-const
        z=brentq(equation,-40,40,xtol=1e-13);n=math.exp(z);d=t/(6*n)
        budget=1e18*d*(k*n+cost);sol=prob.solve(budget,tol=1e-7)
        method='monotone_Q1_equation_checked_globally'
        mismatch=abs(math.log10(6e18*sol['N_B']*sol['D_B']/target))
        if sol['Q_star']!=1. or mismatch>1e-9:
            def residual(lc):
                ss=prob.solve(10**lc,tol=1e-7)
                return math.log10(6e18*ss['N_B']*ss['D_B']/target)
            lc=brentq(residual,20,26,xtol=2e-12);sol=prob.solve(10**lc,tol=1e-7)
            mismatch=abs(residual(lc));method='budget_root_checked'
        if mismatch>1e-8:raise ArithmeticError('Training FLOPs target mismatch')
        return dict(conversion='q3_'+kind,growth_factor=gf,horizon_years=h,g=g,
            delta_logN=math.log10(sol['N_B']/base['N_B']),N_multiplier=sol['N_B']/base['N_B'],
            D_multiplier=sol['D_B']/base['D_B'],baseline_N_B=base['N_B'],baseline_D_B=base['D_B'],baseline_Q=base['Q_star'],
            baseline_total_FLOPs=1e22,baseline_training_FLOPs=f0,future_N_B=sol['N_B'],future_D_B=sol['D_B'],future_Q=sol['Q_star'],
            future_total_FLOPs=sol['budget_FLOPs'],total_budget_multiplier=sol['budget_FLOPs']/1e22,
            target_training_FLOPs=target,training_FLOPs_multiplier=target/f0,training_target_log10_residual=mismatch,
            training_share=sol['training_share'],attention_share=sol['attention_share'],quality_share=sol['quality_share'],
            budget_relative_residual=sol['budget_relative_residual'],Q3_optimality_gap=sol['optimality_gap'],
            N_within_B6_support=sol['N_within_B6_support'],D_within_B6_support=sol['D_within_B6_support'],
            context_tokens=8192,Q0=self.q0,path_method=method)

    def rows(self,compute,growths):
        rows=[]
        for gf in growths:
            for h in [1,2]:
                rows.extend([dict(conversion='fitted_relation',growth_factor=gf,horizon_years=h,
                                 delta_logN=(gf*compute['g']-compute['xi'])*h/compute['kappa']),
                             dict(conversion='fixed_training_data',growth_factor=gf,horizon_years=h,
                                  delta_logN=gf*compute['g']*h)])
                rows.extend(self.quality_path(compute['g'],h,gf,kind) for kind in self.problems)
        return rows

def forecast(fit,refs,paths,draw,decays):
    rows=[]
    for path in paths:
        h=path['horizon_years']
        for group in GROUPS:
            logn=math.log10(refs[group])+path['delta_logN']
            for rho in decays:
                timearg=h if rho==0 else -math.expm1(-rho*h)/rho
                frame=pd.DataFrame(dict(logN=[logn],tau=[timearg],group=[group]))
                pred=fit.matrix(frame)[0]
                rows.append(dict(draw_id=draw,model=fit.model,group=group,conversion=path['conversion'],
                    growth_factor=path['growth_factor'],horizon_years=h,trend_decay=rho,
                    forecast_date=str((pd.Timestamp(ORIGIN)+pd.DateOffset(years=h)).date()),
                    N_reference_B=refs[group],future_logN_B=logn,future_N_B=10**logn,
                    q05=float(pred[0]),q85=float(pred[1]),q95=float(pred[2]),
                    individual_band_calibrated=False,resource_path_conditional=True))
    return rows

def main():
    start=time.perf_counter();p,params,q0,q2h=upstream()
    c1,c4,c6,_,_=read_sources();idx=identity_table(c1,c4);strict,_=asof(c1,idx,ORIGIN,strict=True)
    broad,_=asof(c1,idx,ORIGIN);aux=auxiliary_sample(broad,set(strict.repo))
    selected=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))['selected']
    compute,annual=compute_fit(c4);save(annual,'compute_annual_frontier.csv')
    refs={g:reference_n(strict,g) for g in GROUPS};f=fit_joint(aux,strict,selected)
    # ``paths`` must stay bound to the imported module here: it is also used below
    # to locate the frozen previous run. The resource table keeps its own name.
    resource_paths=Paths(p,q0).rows(compute,[0.,.5,1.]);save(pd.DataFrame(resource_paths),'resource_paths_point.csv')
    point=pd.DataFrame(forecast(f,refs,resource_paths,-1,[0.,.5,1.]));save(point,'forecast_scenarios.csv')
    previous=pd.read_csv(paths.VALIDATION/'q4_optimized'/'resource_paths_point.csv')
    new=pd.DataFrame(resource_paths);new=new[new.conversion.str.startswith('q3_')]
    comp=new.merge(previous,on=['conversion','horizon_years','growth_factor'],suffixes=('_new','_old'),validate='one_to_one')
    save(comp,'q3_resource_version_comparison.csv')
    available=compute_asof(c4,ORIGIN);universe=np.unique(np.r_[aux.org,strict.org,available.org])
    rng=np.random.default_rng(PROTOCOL['seed']+1)
    draws=[];resource_rows=[];coefs=[];failures=[];countrows=[];fitlogs=[]
    for draw in range(PROTOCOL['bootstrap_draws']):
        sample=rng.choice(universe,len(universe),replace=True)
        wa=cluster_counts(aux.org,sample);ws=cluster_counts(strict.org,sample);wc=cluster_counts(available.org,sample)
        countrows.extend(dict(draw_id=draw,publisher=g,count=int(n)) for g,n in zip(*np.unique(sample,return_counts=True)))
        try:
            if any(((ws>0)&strict.group.eq(g)).sum()==0 for g in GROUPS):
                raise ValueError('Missing strict model type: quantile intercept unidentified')
            fit=fit_joint(aux,strict,selected,aux_counts=wa,strict_counts=ws)
            cf,_=compute_fit(c4,counts=dict(zip(available.c4_id,wc)))
            ref={g:reference_n(strict,g,ws) for g in GROUPS}
            pr=Paths(params[draw],q0).rows(cf,[1.]);rr=forecast(fit,ref,pr,draw,[0.])
            draws.extend(rr);resource_rows.extend(dict(draw_id=draw,q2_draw_id=draw,**r) for r in pr)
            coefs.append(dict(draw_id=draw,q2_draw_id=draw,**cf,fit_gap=fit.diagnostics['optimality_gap'],
                Nref_pretrained=ref['pretrained'],Nref_posttrained=ref['posttrained'],
                strict_positive_rows=int((ws>0).sum()),strict_missing_type=bool(any(((ws>0)&strict.group.eq(g)).sum()==0 for g in GROUPS))))
            fitlogs.append(dict(draw_id=draw,**fit.to_dict()))
        except (ValueError,ArithmeticError,RuntimeError,OverflowError) as err:
            failures.append(dict(draw_id=draw,message=str(err)))
        if (draw+1)%50==0:print('Joint uncertainty:',draw+1,'/ 200',flush=True)
    if len(coefs)<100:raise RuntimeError('Too few valid joint uncertainty draws')
    save(pd.DataFrame(draws),'joint_forecast_draws.csv');save(pd.DataFrame(resource_rows),'joint_resource_paths.csv')
    save(pd.DataFrame(coefs),'joint_coefficients.csv');save(pd.DataFrame(countrows),'joint_publisher_counts.csv')
    save(pd.DataFrame(failures,columns=['draw_id','message']),'joint_failed_draws.csv')
    write_json(OUT/'joint_fit_diagnostics.json',fitlogs)
    keys=['model','group','conversion','growth_factor','horizon_years','trend_decay']
    bands=pd.DataFrame(draws).groupby(keys).q85.agg(curve_q05=lambda x:x.quantile(.05),
        curve_q95=lambda x:x.quantile(.95),curve_median='median',draws='size').reset_index()
    bands=bands.merge(point[point.growth_factor.eq(1)&point.trend_decay.eq(0)],on=keys,validate='one_to_one')
    save(bands,'forecast_curve_intervals.csv')
    # Keep B1 and B6 Loss scales separate while refreshing the real-Pythia bridge.
    import q4_data
    q4_data.OUT=OUT
    from q4_bridge import bridge_analysis
    bridge,_=bridge_analysis(c6,q2h['B1_parameters']);write_json(OUT/'bridge_summary.json',bridge)
    paths_df=pd.DataFrame(resource_rows)
    summary=dict(selected_conditional_model=selected,origin=ORIGIN,compute=compute,reference_N=refs,
        joint_draws_successful=len(coefs),joint_draws_failed=len(failures),publisher_universe=len(universe),
        missing_type_exclusions=sum('Missing strict model type' in x['message'] for x in failures),
        aligned_C1_C4_publishers=len(set(np.r_[aux.org,strict.org])&set(available.org)),
        max_fit_gap=float(pd.DataFrame(coefs).fit_gap.max()),
        max_training_target_log10_residual=float(paths_df.training_target_log10_residual.max()),
        max_budget_residual=float(paths_df.budget_relative_residual.max()),
        max_Q3_optimality_gap=float(paths_df.Q3_optimality_gap.max()),
        bootstrap_missing_type=int(pd.DataFrame(coefs).strict_missing_type.sum()),
        elapsed_seconds=time.perf_counter()-start,
        caveat='conditional percentile ranges for fitted quantile curve; separate q05/q95 bands uncalibrated, no real-time or long-horizon validation')
    write_json(OUT/'forecast_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
