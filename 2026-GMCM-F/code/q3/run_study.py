"""Q3 nominal, bounded minimax and conditional resampling study.

All inputs are the self-contained package outputs of Q1/Q2 plus the frozen B6
attachment. Results are conditional Monte Carlo sensitivity on that single
source, never independent real-training evidence.
"""
from pathlib import Path
import sys,json,hashlib,math,time,itertools,platform
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from scipy.optimize import brentq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import paths

from core import Params,Problem,QualityCost,COST_LABELS,local_boundary_budget,sufficient_saturation_bound,FLOPS_SCALE
HERE=Path(__file__).resolve().parent
OUT=paths.Q3
Q2=paths.Q2

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(path,v):path.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def table(name,rows):
    df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(OUT/(name+'.csv'),index=False)
    return df


@dataclass(frozen=True)
class Reference:
    """Q1 quality baseline and anchor recipe reconstructed from package outputs.

    ``q0``/``q0_word`` are share-weighted means over the six quality domains
    that the Q1 mapping table carries empirically; ``coverage`` is the total
    anchor share they cover. ``contexts`` lists the C7 attention-window sizes.
    """
    q0: float
    q0_word: float
    coverage: float
    contexts: list
    surface: pd.DataFrame


def load_inputs():
    h=json.loads((Q2/'q2_handoff_v26.json').read_text(encoding='utf-8'))
    m=json.loads((Q2/'run_manifest.json').read_text(encoding='utf-8'))
    if h['schema']!='q2-quality-response-v26/1' or h['selected_model']!='curved':
        raise ValueError('Q3 requires the selected curved Q2 handoff')
    if h['units']!={'n':'billion parameters','d':'billion tokens'}:
        raise ValueError('Unexpected units')
    for n in ['q2_handoff_v26.json','joint_parameter_draws.csv','full_fit_parameters.json','model_selection.json']:
        if digest(Q2/n)!=m['output_sha256'][n]:raise ValueError('Q2 fingerprint mismatch: '+n)
    b6=paths.B/'supplementary_NQ_experiment.csv'
    b6_relative='real_attachments/B_scaling_laws/'+b6.name
    if digest(b6)!=m['inputs_sha256'][b6_relative]:raise ValueError('B6 changed')
    # Q1 anchor recipe and mapped quality levels come from the package's own Q1
    # outputs; no archived receipt is read here.
    effects=pd.read_csv(paths.Q1/'domain_reallocation_effects.csv')
    mapping=pd.read_csv(paths.Q1/'quality_to_mixture_mapping.csv',encoding='utf-8-sig')
    p0=effects.set_index('domain').anchor_share
    qmap=mapping.dropna(subset=['Q_from_quality_domain']).set_index('mixture_domain').Q_from_quality_domain
    qmap_weighted=mapping.dropna(subset=['Q_token_weighted_from_quality_domain']
                                 ).set_index('mixture_domain').Q_token_weighted_from_quality_domain
    coverage=float(sum(p0.get(d,0.) for d in qmap.index))
    if coverage<=0:raise ValueError('Q1 anchor covers no mapped quality domain')
    surface=pd.read_csv(paths.Q1/'response_surface_evaluations.csv',encoding='utf-8-sig')
    c7=pd.read_csv(paths.C/'model_architecture_metadata.csv')
    values=pd.to_numeric(c7.max_position_embeddings,errors='coerce').to_numpy(float)
    if not np.isfinite(values).all() or np.any(values<=0) or np.any(values!=np.floor(values)):
        raise ValueError('C7 context must be a positive integer')
    reference=Reference(float(sum(p0.get(d,0.)*v for d,v in qmap.items())/coverage),
                        float(sum(p0.get(d,0.)*v for d,v in qmap_weighted.items())/coverage),
                        coverage,pd.Series(values.astype(int)).value_counts().sort_index().index.tolist(),
                        surface[surface.model.eq('selected')].copy())
    if abs(h['Q1_anchor_Q']-reference.q0)>1e-13:raise ValueError('Q1 anchor mismatch')
    draws=pd.read_csv(Q2/'joint_parameter_draws.csv')
    selected=draws[draws.model.eq('curved')].sort_values('draw')
    if len(selected)!=200 or selected.draw.tolist()!=list(range(200)):raise ValueError('Joint draw IDs incomplete')
    q0=h['Q1_anchor_Q']
    pars=Params.from_dict(h['parameters'])
    if abs(h['parameters']['delta_min']-(pars.delta-2*pars.kappa*math.log(.1)))>1e-10:
        raise ValueError('Curvature/endpoint mismatch')
    return h,pars,[Params.from_dict(d) for d in selected.to_dict('records')],reference,m


def independent_draws(n,seed):
    sys.path.insert(0,str(paths.CODE/'q2'))
    from quality_solver import fit,parameter_dict,ScalingData
    b6=pd.read_csv(paths.B/'supplementary_NQ_experiment.csv')
    data=ScalingData.from_frame(b6,quality=True)
    codes=b6.groupby(['N_params_B','D_tokens_B'],sort=True).ngroup().to_numpy()
    rng=np.random.default_rng(seed)
    rows,counts=[],[]
    for i in range(n):
        count=np.bincount(rng.integers(0,45,45),minlength=45)
        d=data.weighted(count[codes])
        power=fit(d,'power');curved=fit(d,'curved',warm=power.theta)
        for name,f in [('power',power),('curved',curved)]:
            rows.append(dict(draw=i,model=name,seed=seed,**parameter_dict(f.theta),
                             fit_kkt=f.kkt_residual,fit_objective=f.objective))
        counts.append(dict(draw=i,**{str(j):int(c) for j,c in enumerate(count)}))
    df=table('independent_parameter_draws',rows)
    table('independent_cell_counts',counts)
    return {m:[Params.from_dict(r) for r in df[df.model.eq(m)].to_dict('records')]
            for m in ['power','curved']}


def transition(p,ctx,kind,q0,scale=1.):
    """Report verified local boundary or a finite global-switch bracket."""
    prob=Problem([p],ctx,kind,q0,cost_scale=scale)
    candidate=local_boundary_budget(p,ctx,prob.cost)
    bound=sufficient_saturation_bound(p,ctx,prob.cost)
    row=dict(context_tokens=ctx,cost_model=kind,cost_scale=scale,local_candidate_FLOPs=candidate,
             **(bound or {}))
    # At a continuous transition, C(1) is necessary for the upper boundary.
    # Check the entire Q interval at C(1); record the numerical bound explicitly.
    if candidate is not None:
        sol=prob.solve(candidate,tol=3e-8)
        endpoint=prob.scale(1,candidate)['value']
        row['candidate_global_gap_upper']=max(0.,endpoint-sol['objective_lower'])
        # A finite margin on each side prevents a shallow near-boundary root
        # from being mistaken for a robustly established phase change.
        lo,hi=candidate*(1-1e-3),candidate*(1+1e-3)
        left=prob.solve(lo,tol=1e-9);right=prob.solve(hi,tol=1e-9)
        row.update(left_Q=left['Q_star'],right_Q=right['Q_star'])
        if endpoint-sol['objective']<1e-9 and left['Q_star']<1-1e-6 and right['Q_star']==1:
            return {**row,'transition_estimate_FLOPs':candidate,'bracket_low_FLOPs':lo,
                    'bracket_high_FLOPs':hi,'classification':'continuous_candidate_globally_checked',
                    'scope':'global check at boundary candidate within reported numerical gap'}
    # Locate a sampled change then bisect a certified endpoint-advantage test.
    # Only a numerical transition bracket is claimed; sufficiency bound remains separate.
    scan=np.geomspace(1.,max(1e22,bound['sufficient_budget_FLOPs'] if bound else 1e26),49)
    last=None
    for c in scan:
        s=prob.solve(float(c),tol=2e-7)
        upper=s['Q_star']==1.
        if upper and last is not None and not last[1]:
            a,b=last[0],float(c)
            for _ in range(18):
                mid=math.sqrt(a*b);v=prob.solve(mid,tol=1e-8)
                if v['Q_star']==1.:b=mid
                else:a=mid
            return {**row,'transition_estimate_FLOPs':math.sqrt(a*b),
                    'bracket_low_FLOPs':a,'bracket_high_FLOPs':b,
                    'classification':'global_switch_numerical_bracket',
                    'scope':'first observed switch on finite scan; no uniqueness theorem asserted'}
        last=(float(c),upper)
    return {**row,'transition_estimate_FLOPs':None,'classification':'unresolved_or_before_scan',
            'scope':'no finite bracket found'}


def near_profile(p,ctx,kind,q0,c,solution,tol=.001):
    prob=Problem([p],ctx,kind,q0)
    q=np.linspace(q0,1,161)
    vals=[prob.scale(float(x),c)['value'] for x in q]
    cutoff=solution['loss_star']+tol
    crossings=[]
    for a,b,va,vb in zip(q[:-1],q[1:],vals[:-1],vals[1:]):
        if (va-cutoff)*(vb-cutoff)<0:
            crossings.append(brentq(lambda x:prob.scale(x,c)['value']-cutoff,a,b))
    bounds=sorted([q0,1.]+crossings)
    intervals=[(a,b) for a,b in zip(bounds[:-1],bounds[1:])
               if prob.scale((a+b)/2,c)['value']<=cutoff]
    z=math.log(solution['N_B']);qs=solution['Q_star']
    f=lambda x:prob.values(x,qs,c)[0][0]-cutoff
    lo,hi=z-1,z+1
    while f(lo)<0:lo-=1
    while f(hi)<0:hi+=1
    nlo=math.exp(brentq(f,lo,z));nhi=math.exp(brentq(f,z,hi))
    return dict(budget_FLOPs=c,context_tokens=ctx,cost_model=kind,loss_tolerance=tol,
        quality_intervals=json.dumps(intervals),N_at_fixed_Q_low=nlo,N_at_fixed_Q_high=nhi,
        quality_root_scope='numerical profile; 161-point bracketing with refinement'),[
        dict(budget_FLOPs=c,context_tokens=ctx,cost_model=kind,Q=float(x),profile_loss=v,
             excess_loss=v-solution['loss_star']) for x,v in zip(q,vals)]


def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=True)
    protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    h,p,design,reference,manifest=load_inputs();q0=reference.q0
    budgets=protocol['nominal_budgets'];contexts=protocol['contexts'];kinds=list(COST_LABELS)
    rows,gain,near,profile_rows=[],[],[],[]
    for c,ctx,kind in itertools.product(budgets,contexts,kinds):
        prob=Problem([p],ctx,kind,q0);r=prob.solve(c,tol=protocol['nominal_loss_gap_tolerance'])
        rows.append(r)
        fixed=prob.scale(q0,c)
        gain.append(dict(budget_FLOPs=c,context_tokens=ctx,cost_model=kind,
            fixed_Q0_loss=fixed['value'],optimized_loss=r['loss_star'],
            relative_gain=1-r['loss_star']/fixed['value']))
        if ctx==8192:
            nr,pr=near_profile(p,ctx,kind,q0,c,r);near.append(nr);profile_rows+=pr
    main_df=table('nominal_configurations',rows)
    table('quality_optimization_gain',gain);table('near_optimal_profiles',near);table('quality_profiles',profile_rows)
    print('45 main configurations complete.',flush=True)
    scan=[]
    for ctx,kind in itertools.product(contexts,kinds):
        prob=Problem([p],ctx,kind,q0)
        for c in np.geomspace(1e16,1e26,41):
            scan.append(prob.solve(float(c),tol=2e-7))
    table('budget_scan',scan)
    transitions=[]
    for ctx,kind in itertools.product(contexts,kinds):
        transitions.append(transition(p,ctx,kind,q0))
    table('saturation_transitions',transitions)
    print('Budget scans and global checks of saturation transitions complete.',flush=True)

    independent=independent_draws(protocol['independent_resampling_draws'],protocol['independent_resampling_seed'])
    print('200 independent paired parameter resamples fitted; decisions remain based on design scenarios.',flush=True)
    opt_cache={}
    propagated=[];robust=[];validation=[];tradeoff=[];cost_eval=[]
    robust_cases=list(itertools.product(protocol['robust_budgets'],protocol['robust_contexts'],kinds))
    def optima(label,params,c,ctx,kind):
        key=(label,c,ctx,kind)
        if key not in opt_cache:
            batch=[]
            for i,pp in enumerate(params):
                sol=Problem([pp],ctx,kind,q0).solve(c,tol=protocol['scenario_optimum_tolerance'],refine=False)
                batch.append(sol)
                propagated.append(dict(parameter_set=label,draw=i,**sol))
            opt_cache[key]=batch
        return opt_cache[key]
    for case,(c,ctx,kind) in enumerate(robust_cases):
        nominal=Problem([p],ctx,kind,q0).solve(c)
        own=optima('design_curved',design,c,ctx,kind)
        vl=np.array([r['objective_lower'] for r in own]);vu=np.array([r['objective'] for r in own])
        robust_problem=Problem(design,ctx,kind,q0,offsets=vu)
        rr=robust_problem.solve(c,tol=8e-7,refine=True)
        z=math.log(rr['N_B']);qq=rr['Q_star']
        upper=float(np.max(robust_problem.values(z,qq,c)[0]+vu-vl))
        rr.update(objective_upper_including_baseline_error=upper,
                  certified_regret_gap=upper-rr['objective_lower'])
        if rr['certified_regret_gap']>protocol['robust_true_regret_gap_tolerance']:
            raise RuntimeError('True-regret error budget exceeded')
        nominal_reg=float(np.max(robust_problem.explicit_loss(nominal['N_B'],nominal['D_B'],nominal['Q_star'])-vu))
        nominal_penalty=float(Problem([p],ctx,kind,q0).explicit_loss(rr['N_B'],rr['D_B'],qq)[0]-nominal['objective'])
        robust.append(dict(**rr,nominal_N_B=nominal['N_B'],nominal_D_B=nominal['D_B'],
            nominal_Q=nominal['Q_star'],nominal_worst_regret=nominal_reg,
            relative_regret_reduction=1-upper/nominal_reg if nominal_reg else 0.,
            nominal_model_penalty=nominal_penalty))
        candidates={'nominal':nominal,'robust':rr}
        # Interpolations are feasible shared decisions after the common budget
        # equation is applied once; they form a tradeoff curve, not a Pareto proof.
        for weight in [0.,.25,.5,.75,1.]:
            zn=(1-weight)*math.log(nominal['N_B'])+weight*z
            qn=(1-weight)*nominal['Q_star']+weight*qq;n=math.exp(zn)
            d=c/FLOPS_SCALE/(robust_problem.k*n+robust_problem.cost.h(qn))
            r=robust_problem.explicit_loss(n,d,qn)-vu
            pn=float(Problem([p],ctx,kind,q0).explicit_loss(n,d,qn)[0]-nominal['objective'])
            tradeoff.append(dict(budget_FLOPs=c,context_tokens=ctx,cost_model=kind,weight=weight,
                N_B=n,D_B=d,Q=qn,design_worst_regret=float(r.max()),nominal_penalty=pn))
        for label,params in [('design_curved',design),('evaluation_curved',independent['curved']),
                             ('evaluation_power',independent['power'])]:
            own_eval=own if label=='design_curved' else optima(label,params,c,ctx,kind)
            v=np.array([r['objective'] for r in own_eval])
            model=Problem(params,ctx,kind,q0)
            for method,x in candidates.items():
                regret=model.explicit_loss(x['N_B'],x['D_B'],x['Q_star'])-v
                for i,r in enumerate(regret):
                    validation.append(dict(budget_FLOPs=c,context_tokens=ctx,cost_model=kind,
                        parameter_set=label,policy=method,draw=i,regret=float(r)))
        for scale in [.5,2.]:
            newprob=Problem([p],ctx,kind,q0,cost_scale=scale)
            for method,x in candidates.items():
                used=x['D_B']*(newprob.k*x['N_B']+newprob.cost.h(x['Q_star']))/(c/FLOPS_SCALE)
                cost_eval.append(dict(budget_FLOPs=c,context_tokens=ctx,cost_model=kind,
                    cost_scale=scale,policy=method,N_B=x['N_B'],D_B=x['D_B'],Q=x['Q_star'],
                    budget_utilization=used,feasible=used<=1+1e-10))
        if (case+1)%3==0:print(f'Bounded robust configurations and conditional evaluation: {case+1}/18.',flush=True)
    prop_df=table('joint_optimal_configurations',propagated)
    robust_df=table('robust_configurations',robust)
    val_df=table('policy_regret_draws',validation);table('regret_tradeoff',tradeoff)
    table('fixed_policy_cost_feasibility',cost_eval)
    summaries=[]
    for keys,g in val_df.groupby(['budget_FLOPs','context_tokens','cost_model','parameter_set','policy']):
        vals=np.sort(g.regret.to_numpy())
        summaries.append(dict(zip(['budget_FLOPs','context_tokens','cost_model','parameter_set','policy'],keys))|
            dict(draws=len(vals),mean_regret=float(vals.mean()),worst_regret=float(vals[-1]),
                 tail10_mean=float(vals[-math.ceil(.1*len(vals)):].mean()),p95=float(np.quantile(vals,.95))))
    table('policy_regret_summary',summaries)
    intervals=[]
    for keys,g in prop_df[prop_df.parameter_set.eq('design_curved')].groupby(['budget_FLOPs','context_tokens','cost_model']):
        row=dict(zip(['budget_FLOPs','context_tokens','cost_model'],keys))
        row['upper_quality_frequency']=float(g.active_boundary.eq('upper').mean())
        for name in ['N_B','D_B','Q_star','loss_star','quality_share']:
            for label,value in zip(['p025','median','p975'],g[name].quantile([.025,.5,.975])):
                row[name+'_'+label]=float(value)
        intervals.append(row)
    table('joint_configuration_intervals',intervals)

    structural=[]
    coverage=reference.coverage
    variants=[('cost_half',q0,.5,p),('cost_double',q0,2.,p),
              ('word_weighted_Q0',reference.q0_word,1.,p),
              ('unmapped_lower_Q0',coverage*q0,1.,p),
              ('unmapped_upper_Q0',coverage*q0+1-coverage,1.,p),
              ('no_quality_benefit',q0,1.,replace(p,delta=0,kappa=0))]
    for label,qq,scale,pp in variants:
        for c,ctx,kind in robust_cases:
            structural.append(dict(scenario=label,**Problem([pp],ctx,kind,qq,cost_scale=scale).solve(c)))
    table('structural_sensitivity',structural)
    # Local KKT candidate uncertainty is named explicitly and not mislabeled
    # as the distribution of the first global transition.
    boundary_draws=[]
    for i,pp in enumerate(design):
        for ctx,kind in itertools.product(contexts,kinds):
            cc=QualityCost(kind,q0)
            boundary_draws.append(dict(draw=i,context_tokens=ctx,cost_model=kind,
                local_boundary_candidate_FLOPs=local_boundary_budget(pp,ctx,cc),
                **(sufficient_saturation_bound(pp,ctx,cc) or {})))
    table('saturation_bound_draws',boundary_draws)
    # Exact derivative of the log local boundary candidate by the chain rule.
    # Finite changes in log-parameters are checked separately in verify.py.
    sensitivity=[]
    for kind in kinds:
        cc=QualityCost(kind,q0);ctx=8192
        gamma=p.delta/p.beta;T=cc.hp(1)/gamma;r=T-cc.h(1)
        if r<=0:continue
        k=6+2e-4*ctx;logn=math.log(r/k);logs=math.log(T)
        logc=math.log(local_boundary_budget(p,ctx,cc)/FLOPS_SCALE)
        dlogc_dlogA=-1/p.beta;dlogc_dlogB=1/p.beta
        dlogc_dalpha=(-1/p.alpha+logn)/p.beta
        dlogc_dlogdelta=-((p.alpha+1)*T/r+(p.beta-1))/p.beta
        dlogc_dlogbeta=(1+(p.alpha+1)*T/r+p.beta*logs+(p.beta-1)-p.beta*logc)/p.beta
        for name,value in [('A',dlogc_dlogA),('B',dlogc_dlogB),
                           ('alpha',p.alpha*dlogc_dalpha),('beta',dlogc_dlogbeta),
                           ('delta',dlogc_dlogdelta),('cost_scale',(p.alpha+p.beta)/p.beta),
                           ('k',-p.alpha/p.beta)]:
            sensitivity.append(dict(context_tokens=ctx,cost_model=kind,parameter=name,
                                    elasticity_of_local_boundary_budget=value))
    table('local_transition_elasticities',sensitivity)
    # Preserve the selected Q1 surface as the only mixture source.
    mix=[]
    probes=reference.surface[reference.surface.point_kind.isin(['anchor','single_reallocation','selected_1m_candidate'])]
    for probe in probes.itertuples():
        for lam,c,kind in itertools.product([0.,1.,2.],[1e19,1e22],kinds):
            prob=Problem([p],8192,kind,q0,psi=float(probe.loss_change_vs_anchor),mix_lambda=lam)
            mix.append(dict(probe=probe.point_kind,domain=probe.domain_i,lambda_mix=lam,
                psi=float(probe.loss_change_vs_anchor),**prob.solve(c)))
    table('q1_mixture_sensitivity',mix)
    summary=dict(status='COMPUTED',q2_handoff_schema=h['schema'],selected_quality_model='curved',
        params=vars(p),Q0=q0,counts=dict(main=len(rows),robust=len(robust),
        design_draws=len(design),independent_draws_per_model=len(independent['curved']),
        propagated=len(propagated)),main_boundary_counts=main_df.active_boundary.value_counts().to_dict(),
        max_nominal_gap=float(main_df.optimality_gap.max()),
        max_robust_regret_gap=float(robust_df.certified_regret_gap.max()),
        elapsed_seconds=time.perf_counter()-start,python=platform.python_version(),
        evidence='B6 conditional semi-synthetic calibration; independent resampling uses the same source',
        certificates='analytic bounds evaluated with double precision and numerical padding; no interval arithmetic',
        Q4_synchronized=False,manuscript_edited=False)
    js(OUT/'results.json',summary)
    js(OUT/'q3_handoff_v26.json',dict(schema='q3-curved-budget-v26/1',
        Q2_input=str(Q2/'q2_handoff_v26.json'),params=vars(p),Q0=q0,
        budget='C/1e18 = D_B*((6+2e-4*context)*N_B+h(Q))',
        main_configurations='nominal_configurations.csv',robust='robust_configurations.csv',
        independent_evaluation='policy_regret_summary.csv',
        caveat='No direct B6 Loss-to-Pythia benchmark transfer; support flags must accompany downstream use'))
    inputs=[Q2/'q2_handoff_v26.json',Q2/'joint_parameter_draws.csv',Q2/'run_manifest.json',
            HERE/'protocol.json',
            paths.B/'supplementary_NQ_experiment.csv',
            paths.C/'model_architecture_metadata.csv']

    def relative(path):
        """Package-relative label; attachments are labelled from the attachment root."""
        path=Path(path)
        for prefix,root in (('outputs/q2',Q2),('outputs/q3',OUT),('code',paths.CODE),
                            ('attachments',paths.ATTACHMENTS)):
            try:return prefix+'/'+path.relative_to(root).as_posix()
            except ValueError:continue
        return path.as_posix()

    js(OUT/'run_manifest.json',dict(inputs={relative(x):digest(x) for x in inputs},
        source={x.name:digest(x) for x in [HERE/'core.py',HERE/'run_study.py']},
        outputs={x.name:digest(x) for x in OUT.iterdir() if x.is_file() and x.name!='run_manifest.json'}))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()

