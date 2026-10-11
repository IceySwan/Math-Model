"""Chronological and publisher validation of the fixed Q4 candidates."""
import json,time,sys
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import paths
from protocol import OUT,ORIGIN,PROTOCOL,save,write_json
from ordered_model import fit_joint,conformal_rank,publisher_weights
from q4_data import read_sources,identity_table,asof
from q4_frontier import project_test_n
from q4_alternatives import auxiliary_sample
from q4_kernels import pinball,cluster_counts

# Comparator predictions preserved from the two earlier validation stages.
FROZEN_STAGE_RUNS=('q4_optimized','q4_exploratory')

def role_of(cutoff,end):
    split=pd.Timestamp(PROTOCOL['development_end'])
    return 'development' if end<=split else 'reused_late' if pd.Timestamp(cutoff)>=split else 'straddling'

def metrics(frame,keys):
    rows=[]
    for key,d in frame.groupby(keys,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        for weighting in ['row','publisher']:
            w=np.ones(len(d))/len(d) if weighting=='row' else publisher_weights(d)
            error=d.actual-d.prediction
            lo,hi=d.pi_low.to_numpy(),d.pi_high.to_numpy();y=d.actual.to_numpy()
            interval=hi-lo+20*np.maximum(lo-y,0)+20*np.maximum(y-hi,0)
            rows.append({**dict(zip(keys,key)), 'weighting':weighting,'predictions':len(d),
                'repositories':d.repo.nunique(),'publishers':d.org.nunique(),
                'pinball85':float(w@pinball(y,d.prediction,.85)),
                'exceedance_rate':float(w@(error.to_numpy()>0)),
                'coverage90':float(w@((y>=lo)&(y<=hi))), 'width_pp':float(w@(hi-lo)),
                'interval_score90':float(w@interval),'MAE_pp':float(w@abs(error)),
                'outside_score_range':int(((d.prediction<-1e-8)|(d.prediction>100+1e-8)).sum()),
                'crossings':int((d.pi_low>d.prediction+1e-8).sum()+(d.prediction>d.pi_high+1e-8).sum())})
    return pd.DataFrame(rows)

def records(fit,test,cutoff,end,role,mode,model=None,bounds=None,extra=None):
    pred=fit.matrix(test);lo,hi=(pred[:,0],pred[:,2]) if bounds is None else bounds
    rows=[]
    for i,r in enumerate(test.itertuples()):
        rows.append(dict(cutoff=str(pd.Timestamp(cutoff).date()),end=str(pd.Timestamp(end).date()),role=role,
            input_mode=mode,model=model or fit.model,repo=r.repo,org=r.org,date=str(r.date.date()),group=r.group,
            N_used=float(r.N),logN_used=float(r.logN),actual=float(r.S),prediction=float(pred[i,1]),
            pi_low=float(lo[i]),pi_high=float(hi[i]),fit_gap=fit.diagnostics['optimality_gap'],**(extra or {})))
    return rows

def split_calibration(c1,c4,index,outer_train,outer_cutoff,model,fitlogs):
    # Use only identities and dates available before the outer cutoff.
    ordered=outer_train.sort_values(['date','repo'])
    splitdate=ordered.iloc[max(0,len(ordered)//2-1)].date
    proper0,_=asof(c1,index,splitdate,strict=True)
    seen=set(c1.loc[c1.date.le(splitdate),'repo'])
    calibration=outer_train[outer_train.date.gt(splitdate)&~outer_train.repo.isin(seen)].copy()
    held=set(calibration.org)
    proper=proper0[~proper0.org.isin(held)].copy()
    broad,_=asof(c1,index,splitdate)
    aux=auxiliary_sample(broad,set(proper0.repo))
    aux=aux[~aux.org.isin(held)].copy()
    if len(proper)<3 or len(calibration)==0 or len(aux)<10:raise ValueError('Insufficient chronological split')
    fit=fit_joint(aux,proper,model)
    fitlogs.append(dict(stage='calibration_fit',cutoff=str(outer_cutoff),**fit.to_dict()))
    projected,_=project_test_n(proper,calibration,c4,splitdate)
    pred=fit.matrix(projected)
    scores=np.maximum(pred[:,0]-calibration.S.to_numpy(),calibration.S.to_numpy()-pred[:,2])
    scoreframe=pd.DataFrame(dict(repo=calibration.repo,org=calibration.org,date=calibration.date,
        actual=calibration.S,pred_lo=pred[:,0],pred_hi=pred[:,2],score=scores,N_used=projected.N))
    grouped=scoreframe.groupby('org').score.max().to_numpy()
    row_c,krow=conformal_rank(scores);org_c,korg=conformal_rank(grouped)
    info=dict(split_date=str(splitdate.date()),proper_rows=len(proper),aux_rows=len(aux),
        calibration_rows=len(calibration),calibration_publishers=len(grouped),
        row_rank=krow,publisher_rank=korg,row_correction=None if np.isinf(row_c) else row_c,
        publisher_correction=None if np.isinf(org_c) else org_c,
        full_support_row_interval=bool(np.isinf(row_c)),full_support_publisher_interval=bool(np.isinf(org_c)),
        proper_latest_date=str(proper.date.max().date()),aux_latest_date=str(aux.date.max().date()),
        fitting_calibration_repo_overlap=len(set(proper.repo)&set(calibration.repo)),
        fitting_calibration_publisher_overlap=len(set(proper.org)&held),
        aux_calibration_publisher_overlap=len(set(aux.org)&held),
        proper_repositories=proper.repo.tolist(),calibration_repositories=calibration.repo.tolist())
    return fit,proper,splitdate,info,scoreframe,row_c,org_c

def load_baselines():
    frames=[]
    for folder in FROZEN_STAGE_RUNS:
        d=pd.read_csv(paths.VALIDATION/folder/'rolling_predictions.csv')
        d=d[d['quantile'].eq(.85)&d.model.isin(PROTOCOL['comparators'])].copy()
        d['role']=d.role.replace({'holdout':'reused_late','straddling_diagnostic':'straddling'})
        d['source']='frozen_previous';frames.append(d)
    return pd.concat(frames,ignore_index=True)

def main():
    start=time.perf_counter();c1,c4,_,_,hashes=read_sources();index=identity_table(c1,c4)
    rows=[];audits=[];calrows=[];calibration_logs=[];fitlogs=[];failures=[];sensitivity=[]
    for cutoff in PROTOCOL['cutoffs']:
        train,_=asof(c1,index,cutoff,strict=True);broad,_=asof(c1,index,cutoff)
        aux=auxiliary_sample(broad,set(train.repo))
        fits={m:fit_joint(aux,train,m) for m in PROTOCOL['candidates']}
        for m,f in fits.items():fitlogs.append(dict(stage='rolling_full',cutoff=cutoff,**f.to_dict()))
        spl={}
        for m in PROTOCOL['candidates']:
            try:
                spl[m]=split_calibration(c1,c4,index,train,cutoff,m,fitlogs)
                f,_,_,info,cs,_,_=spl[m]
                calibration_logs.append(dict(cutoff=cutoff,model=m,**info))
                calrows.extend(dict(cutoff=cutoff,model=m,**r) for r in cs.to_dict('records'))
            except (ValueError,ArithmeticError) as err:
                failures.append(dict(stage='calibration',cutoff=cutoff,model=m,message=str(err)))
        # Sensitivities are predetermined and never candidates for promotion.
        alternatives={}
        for lam in PROTOCOL['penalty_sensitivity']:
            model=f'joint_size_time_lambda{lam}'
            alternatives[model]=fit_joint(aux,train,'joint_size_time',penalty=lam)
            fitlogs.append(dict(stage='penalty_sensitivity',cutoff=cutoff,label=model,**alternatives[model].to_dict()))
        train5=train.copy();aux5=aux.copy()
        for d in [train5,aux5]:d['S']=d[['IFEval','BBH','GPQA','MUSR','MMLU-PRO']].mean(axis=1)
        five=fit_joint(aux5,train5,'joint_size_time');fitlogs.append(dict(stage='five_task',cutoff=cutoff,**five.to_dict()))
        for months in PROTOCOL['months']:
            end=pd.Timestamp(cutoff)+pd.DateOffset(months=months)
            if end>pd.Timestamp(ORIGIN):continue
            all_end,_=asof(c1,index,end,strict=True);seen=set(c1.loc[c1.date.le(pd.Timestamp(cutoff)),'repo'])
            test=all_end[all_end.date.gt(pd.Timestamp(cutoff))&~all_end.repo.isin(seen)].copy()
            if len(train)<10 or not len(test):continue
            role=role_of(cutoff,end);projected,compute=project_test_n(train,test,c4,cutoff)
            audits.append(dict(cutoff=cutoff,end=str(end.date()),months=months,role=role,train_rows=len(train),aux_rows=len(aux),test_rows=len(test),
                train_date_max=str(train.date.max().date()),aux_date_max=str(aux.date.max().date()),
                train_test_repo_overlap=len(set(train.repo)&set(test.repo)),aux_test_repo_overlap=len(set(aux.repo)&set(test.repo)),
                train_test_publisher_overlap=len(set(train.org)&set(test.org)),**compute))
            for mode,frame in [('resource_forecast',projected),('observed_N_diagnostic',test)]:
                for m,f in fits.items():rows.extend(records(f,frame,cutoff,end,role,mode,extra={'months':months,'source':'v26'}))
            for m,f in alternatives.items():sensitivity.extend(records(f,projected,cutoff,end,role,'resource_forecast',m,extra={'months':months,'target':'six_task'}))
            p5=projected.copy();p5['S']=p5[['IFEval','BBH','GPQA','MUSR','MMLU-PRO']].mean(axis=1)
            sensitivity.extend(records(five,p5,cutoff,end,role,'resource_forecast','joint_size_time_five_task',extra={'months':months,'target':'five_task'}))
            for m,values in spl.items():
                f,proper,splitdate,info,_,row_c,org_c=values
                pp,_=project_test_n(proper,test,c4,splitdate);mat=f.matrix(pp)
                for label,correction in [('split_raw',0.),('split_row90',row_c),('split_publisher90',org_c)]:
                    lo=np.clip(mat[:,0]-correction,0,100);hi=np.clip(mat[:,2]+correction,0,100)
                    rows.extend(records(f,pp,cutoff,end,role,'resource_forecast',m+'_'+label,(lo,hi),
                        {'months':months,'source':'v26','split_date':info['split_date'],'calibration_rows':info['calibration_rows'],
                         'calibration_publishers':info['calibration_publishers'],'support_only_interval':bool(np.isinf(correction))}))
        print('Chronological fits:',cutoff,flush=True)
    df=pd.DataFrame(rows);save(df,'rolling_predictions.csv');save(pd.DataFrame(audits),'rolling_input_audit.csv')
    save(pd.DataFrame(calrows),'calibration_residuals.csv');write_json(OUT/'calibration_protocol_audit.json',calibration_logs)
    save(pd.DataFrame(failures,columns=['stage','cutoff','model','message']),'fit_failures.csv')
    combined=pd.concat([load_baselines(),df],ignore_index=True)
    # Month offsets are recomputed from each row's declared calendar window.
    combined['months']=[(pd.Timestamp(e).year-pd.Timestamp(c).year)*12+pd.Timestamp(e).month-pd.Timestamp(c).month for c,e in zip(combined.cutoff,combined.end)]
    save(combined,'rolling_with_baselines.csv')
    mt=metrics(combined,['role','input_mode','model']);save(mt,'rolling_metrics.csv')
    save(metrics(combined[combined.months.eq(1)],['role','input_mode','model']),'one_month_metrics.csv')
    save(metrics(combined,['role','input_mode','model','group']),'type_metrics.csv')
    sens=pd.DataFrame(sensitivity);save(sens,'sensitivity_predictions.csv');save(metrics(sens,['role','model','target']),'sensitivity_metrics.csv')
    dev=mt[mt.role.eq('development')&mt.input_mode.eq('resource_forecast')&mt.weighting.eq('row')&mt.model.isin(PROTOCOL['candidates'])]
    selected=min(PROTOCOL['candidates'],key=lambda m:float(dev[dev.model.eq(m)].pinball85.iloc[0]))
    write_json(OUT/'selection.json',dict(selected=selected,rule=PROTOCOL['selection'],independent_holdout=False,
        original_selection_unchanged=True,development=dev.to_dict('records')))
    # Paired late diagnostics, with all overlapping windows kept in a publisher block.
    late=combined[combined.role.eq('reused_late')&combined.input_mode.eq('resource_forecast')]
    comparisons=[];rng=np.random.default_rng(PROTOCOL['seed'])
    keys=['cutoff','end','repo','org'];a=late[late.model.eq(selected)].set_index(keys)
    for baseline in PROTOCOL['comparators']:
        b=late[late.model.eq(baseline)].set_index(keys).reindex(a.index)
        assert b.actual.notna().all() and np.max(abs(b.actual-a.actual))<1e-10
        gain=pinball(a.actual,b.prediction,.85)-pinball(a.actual,a.prediction,.85)
        orgs=a.index.get_level_values('org').to_numpy();unique=np.unique(orgs);boot=[]
        for _ in range(1000):
            counts=cluster_counts(orgs,rng.choice(unique,len(unique),replace=True));boot.append(float(np.average(gain,weights=counts)))
        comparisons.append(dict(selected=selected,baseline=baseline,pinball_gain=float(np.mean(gain)),
            relative_gain=float(np.mean(gain)/np.mean(pinball(a.actual,b.prediction,.85))),
            cluster_gain_q05=float(np.quantile(boot,.05)),cluster_gain_q95=float(np.quantile(boot,.95)),
            predictions=len(a),repositories=a.index.get_level_values('repo').nunique(),publishers=len(unique),independent_confirmation=False))
    save(pd.DataFrame(comparisons),'paired_late_comparison.csv')
    # Organization holdouts remove that organization from both sources.
    strict,_=asof(c1,index,ORIGIN,strict=True);broad,_=asof(c1,index,ORIGIN);aux=auxiliary_sample(broad,set(strict.repo))
    orgrows=[];orgaudit=[]
    for org in sorted(strict.org.unique()):
        tr=strict[strict.org.ne(org)];te=strict[strict.org.eq(org)];aa=aux[aux.org.ne(org)]
        orgaudit.append(dict(held_publisher=org,train_rows=len(tr),test_rows=len(te),aux_rows=len(aa),strict_overlap=int(tr.org.eq(org).sum()),aux_overlap=int(aa.org.eq(org).sum())))
        for m in PROTOCOL['candidates']:
            f=fit_joint(aa,tr,m);fitlogs.append(dict(stage='publisher_holdout',held_publisher=org,**f.to_dict()))
            orgrows.extend(records(f,te,ORIGIN,ORIGIN,'publisher_holdout','observed_N_diagnostic',extra={'held_publisher':org}))
    orgdf=pd.DataFrame(orgrows);save(orgdf,'publisher_predictions.csv');save(pd.DataFrame(orgaudit),'publisher_input_audit.csv')
    earlier=[]
    for folder in FROZEN_STAGE_RUNS:
        d=pd.read_csv(paths.VALIDATION/folder/'publisher_holdout_predictions.csv')
        earlier.append(d[d['quantile'].eq(.85)&d.model.isin(PROTOCOL['comparators'])])
    save(metrics(pd.concat([*earlier,orgdf],ignore_index=True),['model']),'publisher_metrics.csv')
    full=fit_joint(aux,strict,selected);fitlogs.append(dict(stage='full_origin',**full.to_dict()))
    write_json(OUT/'full_fit.json',full.to_dict());write_json(OUT/'fit_diagnostics.json',fitlogs)
    write_json(OUT/'study_summary.json',dict(selected=selected,rolling_prediction_rows=len(df),fit_count=len(fitlogs),
        max_fit_gap=max(x['diagnostics']['optimality_gap'] for x in fitlogs),
        max_constraint_violation=max(0.,max(-x['diagnostics']['min_constraint_slack'] for x in fitlogs)),
        calibration_failed=len(failures),elapsed_seconds=time.perf_counter()-start,source_sha256=hashes,
        validation_scope='retrospective conditional; current score/metadata availability not reconstructible'))
    print('Selected on development:',selected,flush=True)
    print(mt[mt.role.eq('reused_late')&mt.input_mode.eq('resource_forecast')&mt.weighting.eq('row')].to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()

