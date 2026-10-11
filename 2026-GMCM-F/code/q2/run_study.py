"""Frozen-protocol Q2 comparison, nested validation and joint uncertainty.

Run from any directory. The isolated Q2 outputs are written only under
``paths.Q2``; no other stage's outputs are overwritten.
"""
import sys
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

from quality_solver import (ScalingData, fit, predict, prediction_jacobian,
                            parameter_dict, quality_slope, log_quality)
from scaling_solver import fit_scaling, predict as old_predict

HERE = Path(__file__).resolve().parent
DATA = paths.B
OUT = paths.Q2
# Q1 交接量与本脚本输入的对照表（随包分发，见 data/README.md）
REF = paths.VALIDATION / "q2_reference"
MODELS = ("power", "curved")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def metrics(y, pred):
    r = np.asarray(pred)-np.asarray(y)
    return dict(n=len(r), rmse=float(np.sqrt(np.mean(r*r))), mae=float(np.mean(np.abs(r))),
                bias=float(np.mean(r)))


def blocks(labels, nfold=None):
    groups = sorted(set(labels))
    if nfold is None:
        return [(str(g), labels==g) for g in groups]
    nfold = min(nfold,len(groups))
    # Sorted, round-robin allocation is deterministic, outcome-independent,
    # and preserves all observations within each (N,D) cell.
    return [(str(f),np.isin(labels,groups[f::nfold])) for f in range(nfold)]


def summarize_predictions(frame):
    result = []
    for keys, part in frame.groupby(["protocol","model"], sort=True):
        fold_rmse = [metrics(g.y,g.prediction)["rmse"] for _,g in part.groupby("fold")]
        result.append(dict(protocol=keys[0],model=keys[1], **metrics(part.y,part.prediction),
                           macro_group_rmse=float(np.mean(fold_rmse)),groups=len(fold_rmse)))
    return pd.DataFrame(result)


def comparison(frame):
    wide = frame.pivot(index=["fold","row_id"],columns="model",values="prediction")
    y = frame.drop_duplicates(["fold","row_id"]).set_index(["fold","row_id"])["y"]
    wide = wide.join(y)
    r0 = metrics(wide.y,wide.power)["rmse"]
    r1 = metrics(wide.y,wide.curved)["rmse"]
    groups = list(wide.groupby(level="fold"))
    wins = sum(metrics(g.y,g.curved)["rmse"] < metrics(g.y,g.power)["rmse"] for _,g in groups)
    return dict(power_rmse=r0,curved_rmse=r1,relative_gain=1-r1/r0,
                curved_wins=int(wins),groups=len(groups),
                passes=bool(r1<=.99*r0 and wins>len(groups)/2))


class Experiment:
    def __init__(self,b6):
        self.frame=b6
        self.data=ScalingData.from_frame(b6,quality=True)
        self.cells=b6.groupby(["N_params_B","D_tokens_B"],sort=True).ngroup().to_numpy()
        self.q=b6.Q_score.to_numpy()
        self.cache={}
        self.records=[]

    def fit_pair(self,mask):
        key=np.packbits(mask).tobytes().hex()
        if key not in self.cache:
            d=self.data.take(mask)
            power=fit(d,"power")
            curved=fit(d,"curved",warm=power.theta)
            pair={}
            for model,f in zip(MODELS,(power,curved)):
                ident=len(self.records)
                self.records.append(dict(fit_id=ident,model=model,n_rows=int(mask.sum()),
                    training_row_ids=" ".join(map(str,np.flatnonzero(mask))),**f.record()))
                pair[model]=(f,ident)
            self.cache[key]=pair
        return self.cache[key]

    def cv(self,mask,labels,protocol,nfold=None):
        rows=[]
        sublabels=labels[mask]
        available=np.flatnonzero(mask)
        for name,local_test in blocks(sublabels,nfold):
            test=np.zeros(len(mask),bool)
            test[available[local_test]]=True
            train=mask&~test
            pair=self.fit_pair(train)
            for model,(f,ident) in pair.items():
                pred=predict(f.theta,self.data.take(test))
                for i,p in zip(np.flatnonzero(test),pred):
                    rows.append(dict(protocol=protocol,fold=name,model=model,row_id=int(i),
                        y=float(self.data.y[i]),prediction=float(p),fit_id=ident,
                        N_params_B=float(self.frame.iloc[i].N_params_B),
                        D_tokens_B=float(self.frame.iloc[i].D_tokens_B),Q_score=float(self.q[i]),
                        cell_id=int(self.cells[i])))
        return pd.DataFrame(rows)

    def nested(self,labels,protocol):
        selected_rows,selection=[],[]
        inner_predictions=[]
        for fold,test in blocks(labels):
            train=~test
            iq=self.cv(train,self.q,"inner_Q")
            ic=self.cv(train,self.cells,"inner_cell",nfold=5)
            cq,cc=comparison(iq),comparison(ic)
            chosen="curved" if cq["passes"] and cc["passes"] else "power"
            selection.append(dict(protocol=protocol,fold=fold,selected=chosen,
                **{f"Q_{k}":v for k,v in cq.items()},**{f"cell_{k}":v for k,v in cc.items()}))
            inner=pd.concat([iq,ic],ignore_index=True)
            inner["outer_protocol"],inner["outer_fold"]=protocol,fold
            inner_predictions.append(inner)
            f,ident=self.fit_pair(train)[chosen]
            pred=predict(f.theta,self.data.take(test))
            for i,p in zip(np.flatnonzero(test),pred):
                selected_rows.append(dict(protocol=protocol,fold=fold,model="nested_selector",
                    selected_model=chosen,row_id=int(i),y=float(self.data.y[i]),prediction=float(p),
                    fit_id=ident,N_params_B=float(self.frame.iloc[i].N_params_B),
                    D_tokens_B=float(self.frame.iloc[i].D_tokens_B),Q_score=float(self.q[i]),
                    cell_id=int(self.cells[i])))
        return pd.DataFrame(selected_rows),pd.DataFrame(selection),pd.concat(inner_predictions,ignore_index=True)


def source_audit(b6,b7,b8):
    b7dup=b7.merge(b6,on="experiment_id",suffixes=("_B7","_B6"))
    for col in ["N_params_B","D_tokens_B","Q_score","val_loss"]:
        assert np.allclose(b7dup[col+"_B7"],b7dup[col+"_B6"],atol=0,rtol=0)
    assert len(b7dup)==len(b6)
    audit={"B6_rows":len(b6),"B6_cells":b6.groupby(["N_params_B","D_tokens_B"]).ngroups,
           "B6_Q_levels":sorted(map(float,b6.Q_score.unique())),"B7_duplicate_B6_rows":len(b7dup),
           "B7_new_rows":int((~b7.experiment_id.isin(b6.experiment_id)).sum()),
           "B8_metadata":"source_manifest gives file size only; no common quality mapping or generation formula is documented there",
           "B8_action":"not pooled or used for model selection"}
    for label,frame in [("B6",b6),("B8_calibrated",b8[b8.data_type.eq("calibrated")]),
                        ("B8_extrapolated",b8[b8.data_type.ne("calibrated")])]:
        slopes=[]
        for _,cell in frame.groupby(["N_params_B","D_tokens_B"]):
            if cell.Q_score.nunique()>1:
                slopes.append(float(np.polyfit(np.log(cell.Q_score),cell.val_loss,1)[0]))
        audit[label+"_quality_direction"]={"rows":len(frame),"cells":len(slopes),
            "positive_fraction":float(np.mean(np.array(slopes)>0)),
            "negative_fraction":float(np.mean(np.array(slopes)<0)),"median_slope":float(np.median(slopes))}
    return audit


def derived(theta,q0,q1,n=1.,d=150.):
    ln=theta[1]*n**(-theta[3])
    t0=theta[2]*d**(-theta[4])*np.exp(log_quality(theta,np.log(q0)))
    t1=theta[2]*d**(-theta[4])*np.exp(log_quality(theta,np.log(q1)))
    drop=float(t0-t1)
    # At unchanged Q0, N_eq must yield the same loss reduction as Q0 -> Q1.
    # No finite N can do so when drop >= the reducible N term.
    possible=bool(drop<ln)
    return dict(Q0=q0,Q1=q1,loss_drop=drop,delta_local=float(quality_slope(theta,q0)),
        gamma_local=float(quality_slope(theta,q0)/theta[4]),
        token_factor_at_Q0=float(np.exp((log_quality(theta,np.log(q0))-log_quality(theta,np.log(q1)))/theta[4])),
        iso_loss_D1_over_D0=float(np.exp((log_quality(theta,np.log(q1))-log_quality(theta,np.log(q0)))/theta[4])),
        equivalent_N_B=float(n*(1-drop/ln)**(-1/theta[3])) if possible else None,
        finite_equivalent_N=possible)


def main():
    begin=perf_counter()
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((HERE/"protocol.json").read_text(encoding="utf-8"))
    sources={"B1":DATA/"pythia_training_log_existing.csv",
           "B6":DATA/"supplementary_NQ_experiment.csv",
           "B7":DATA/"supplementary_NQ_experiment_expanded.csv",
           "B8":DATA/"supplementary_NQ_experiment_large.csv"}
    b1,b6,b7,b8=[pd.read_csv(sources[k]) for k in ("B1","B6","B7","B8")]
    assert not b6.experiment_id.duplicated().any()
    assert b6.groupby(["N_params_B","D_tokens_B"]).Q_score.nunique().eq(8).all()
    assert not b6.isna().any().any()
    old=json.loads((REF/"results.json").read_text(encoding="utf-8"))
    audit=source_audit(b6,b7,b8)
    write_json(OUT/"data_audit.json",audit)
    print("Data audited. Reproducing B1 grouped validation.",flush=True)
    d1=ScalingData.from_frame(b1)
    f1=fit_scaling(d1,method="profile")
    b1pred=np.zeros(len(b1))
    for _,test in blocks(b1.N_params_B.to_numpy()):
        f=fit_scaling(d1.take(~test),method="profile")
        b1pred[test]=old_predict(f.theta,d1.take(test))
    pd.DataFrame(dict(row_id=np.arange(len(b1)),N_params_B=b1.N_params_B,
        D_tokens_B=b1.D_tokens_B,y=b1.val_loss,prediction=b1pred)).to_csv(OUT/"B1_grouped_predictions.csv",index=False)
    exp=Experiment(b6)
    allrows=np.ones(len(b6),bool)
    full=exp.fit_pair(allrows)
    fullparams={m:parameter_dict(full[m][0].theta) for m in MODELS}
    write_json(OUT/"full_fit_parameters.json",fullparams)
    print("Running paired primary and N/D sensitivity folds.",flush=True)
    direct=[]
    for name,label in [("leave_Q",exp.q),("leave_cell",exp.cells),
                       ("leave_N",b6.N_params_B.to_numpy()),("leave_D",b6.D_tokens_B.to_numpy())]:
        direct.append(exp.cv(allrows,label,name))
    direct=pd.concat(direct,ignore_index=True)
    direct.to_csv(OUT/"grouped_predictions.csv",index=False)
    print("Running nested model selection within each outer training set.",flush=True)
    nested,selection,inners=[],[],[]
    for name,label in [("leave_Q",exp.q),("leave_cell",exp.cells)]:
        n,s,i=exp.nested(label,name)
        nested.append(n); selection.append(s); inners.append(i)
        print(f"Nested {name}: {len(s)} folds completed.",flush=True)
    nested=pd.concat(nested,ignore_index=True)
    selection=pd.concat(selection,ignore_index=True)
    nested.to_csv(OUT/"nested_predictions.csv",index=False)
    selection.to_csv(OUT/"nested_selection.csv",index=False)
    pd.concat(inners,ignore_index=True).to_csv(OUT/"nested_inner_predictions.csv",index=False)
    combined=pd.concat([direct,nested],ignore_index=True)
    metric_table=summarize_predictions(combined)
    extra=[]
    b7new=b7[~b7.experiment_id.isin(b6.experiment_id)].copy()
    d7=ScalingData.from_frame(b7new,quality=True)
    for m in MODELS:
        theta=full[m][0].theta
        for name,d in [("B6_train",exp.data),("B7_new_Q",d7)]:
            extra.append(dict(protocol=name,model=m,**metrics(d.y,predict(theta,d)),macro_group_rmse=None,groups=None))
        b7new["prediction_"+m]=predict(theta,d7)
    b7new.to_csv(OUT/"B7_new_quality_predictions.csv",index=False)
    metric_table=pd.concat([metric_table,pd.DataFrame(extra)],ignore_index=True)
    metric_table.to_csv(OUT/"validation_metrics.csv",index=False)
    comps={name:comparison(direct[direct.protocol.eq(name)]) for name in ["leave_Q","leave_cell"]}
    nested_gains={}
    for name in comps:
        sub=metric_table[metric_table.protocol.eq(name)].set_index("model")
        nested_gains[name]=float(1-sub.loc["nested_selector","rmse"]/sub.loc["power","rmse"])
    adopt=all(c["passes"] for c in comps.values()) and all(g>0 for g in nested_gains.values())
    chosen="curved" if adopt else "power"
    decision=dict(selected_model=chosen,primary_comparisons=comps,nested_relative_gains=nested_gains,
        nested_selection_counts=selection.groupby(["protocol","selected"]).size().rename("folds").reset_index().to_dict("records"),
        protocol_sha256=sha(HERE/"protocol.json"),B7_used_for_selection=False,
        interpretation="retrospective source-conditional comparison; not independent real-training evidence")
    write_json(OUT/"model_selection.json",decision)
    print(f"Frozen rule selected {chosen}. Computing joint cell bootstrap and diagnostics.",flush=True)
    # Paired cell resampling retains the entire observed quality trajectory.
    rng=np.random.default_rng(protocol["bootstrap_seed"])
    ncell=len(np.unique(exp.cells))
    draws,countsrows,scenarios,bootfits=[],[],[],[]
    q0=float(old["representative_iso_loss_scenario"]["Q0"])
    qt=float(old["representative_iso_loss_scenario"]["token_weighted_quality_baseline"]["Q0"])
    for draw in range(protocol["bootstrap_draws"]):
        count=np.bincount(rng.integers(0,ncell,ncell),minlength=ncell)
        countsrows.append(dict(draw=draw,**{str(j):int(x) for j,x in enumerate(count)}))
        db=exp.data.weighted(count[exp.cells])
        p=fit(db,"power",warm=full["power"][0].theta)
        c=fit(db,"curved",warm=p.theta)
        for m,f in zip(MODELS,(p,c)):
            draws.append(dict(draw=draw,model=m,**parameter_dict(f.theta)))
            bootfits.append(dict(draw=draw,model=m,**f.record()))
            for label,q in [("doc_equal",q0),("token_weighted",qt)]:
                scenarios.append(dict(draw=draw,model=m,anchor=label,**derived(f.theta,q,q+.1)))
    drawframe=pd.DataFrame(draws)
    drawframe.to_csv(OUT/"joint_parameter_draws.csv",index=False)
    pd.DataFrame(countsrows).to_csv(OUT/"bootstrap_cell_counts.csv",index=False)
    pd.DataFrame(bootfits).to_csv(OUT/"bootstrap_solver_diagnostics.csv",index=False)
    pd.DataFrame(scenarios).to_csv(OUT/"joint_scenario_draws.csv",index=False)
    scenario_points=[]
    for m in MODELS:
        for label,q in [("doc_equal",q0),("token_weighted",qt)]:
            scenario_points.append(dict(model=m,anchor=label,**derived(full[m][0].theta,q,q+.1)))
    pd.DataFrame(scenario_points).to_csv(OUT/"scenario_point_estimates.csv",index=False)
    scenario_summary=[]
    for (m,a),part in pd.DataFrame(scenarios).groupby(["model","anchor"]):
        for col in ["loss_drop","delta_local","gamma_local","token_factor_at_Q0",
                    "iso_loss_D1_over_D0","equivalent_N_B"]:
            vals=part[col].dropna()
            lo,med,hi=vals.quantile([.025,.5,.975])
            scenario_summary.append(dict(model=m,anchor=a,quantity=col,q025=lo,median=med,q975=hi,
                                         finite_draws=len(vals),total_draws=len(part)))
    pd.DataFrame(scenario_summary).to_csv(OUT/"scenario_intervals.csv",index=False)
    # Jacobian geometry and the least identifiable normalized-column direction.
    svds=[]
    directions=[]
    residuals=[]
    profiles=[]
    curves=[]
    parameter_summary=[]
    for m in MODELS:
        f=full[m][0]; th=f.theta
        ph,jac=prediction_jacobian(th,exp.data)
        scale=np.linalg.norm(jac,axis=0)
        names=["E","A","B","alpha","beta","delta"]+(["delta_min"] if m=="curved" else [])
        for kind,j in [("raw",jac),("column_normalized",jac/scale)]:
            _,s,vh=np.linalg.svd(j,full_matrices=False)
            for ix,value in enumerate(s):
                svds.append(dict(model=m,scaling=kind,index=ix,singular_value=value,condition_number=s[0]/s[-1]))
            if kind=="column_normalized":
                directions += [dict(model=m,parameter=name,loading=value) for name,value in zip(names,vh[-1])]
        sub=drawframe[drawframe.model.eq(m)].drop(columns=["model","draw"])
        sub.corr().to_csv(OUT/f"parameter_correlation_{m}.csv")
        for col in sub:
            lo,med,hi=sub[col].quantile([.025,.5,.975])
            parameter_summary.append(dict(model=m,parameter=col,estimate=fullparams[m][col],
                q025=lo,median=med,q975=hi,boundary_zero_fraction=float(np.mean(np.abs(sub[col])<1e-7))))
        resid=b6.copy(); resid["prediction"]=ph; resid["residual"]=ph-exp.data.y
        resid["cell_id"]=exp.cells; resid["model"]=m
        residuals.append(resid)
        for idx,name in [(1,"beta"),(2,"delta")]:
            grid=np.unique(np.r_[th[idx+3],np.linspace(max(.005 if idx==1 else 0,th[idx+3]*.45),th[idx+3]*1.7,27)])
            for value in grid:
                fp=fit(exp.data,m,warm=th,fixed={idx:float(value)})
                profiles.append(dict(model=m,parameter=name,value=value,objective=fp.objective,
                    excess_SSE=fp.objective-f.objective,kkt=fp.kkt_residual))
        qgrid=np.geomspace(.1,1,101)
        for q in qgrid:
            boot_sub=drawframe[drawframe.model.eq(m)]
            bt=boot_sub[names].to_numpy()
            response=np.array([np.exp(log_quality(t,np.log(q))) for t in bt])
            slope=np.array([quality_slope(t,q)/t[4] for t in bt])
            lo,hi=np.quantile(response,[.025,.975]);sl,sh=np.quantile(slope,[.025,.975])
            curves.append(dict(model=m,Q=q,data_term_ratio=np.exp(log_quality(th,np.log(q))),
                ratio_q025=lo,ratio_q975=hi,gamma_local=quality_slope(th,q)/th[4],gamma_q025=sl,gamma_q975=sh))
    pd.DataFrame(svds).to_csv(OUT/"jacobian_singular_values.csv",index=False)
    pd.DataFrame(directions).to_csv(OUT/"weak_parameter_directions.csv",index=False)
    pd.DataFrame(parameter_summary).to_csv(OUT/"parameter_intervals.csv",index=False)
    pd.DataFrame(profiles).to_csv(OUT/"parameter_profiles.csv",index=False)
    pd.DataFrame(curves).to_csv(OUT/"quality_response_curves.csv",index=False)
    residuals=pd.concat(residuals,ignore_index=True)
    residuals.to_csv(OUT/"training_residuals.csv",index=False)
    residual_audit={}
    for m,part in residuals.groupby("model"):
        sse=part.assign(sq=part.residual**2).groupby("cell_id").sq.sum().sort_values(ascending=False)
        residual_audit[m]=dict(top_cell_sse_fraction=float(sse.iloc[0]/sse.sum()),
            top_10_percent_cells_sse_fraction=float(sse.iloc[:int(np.ceil(.1*len(sse)))].sum()/sse.sum()))
    ra=residual_audit["power"]
    residual_audit["robust_candidate_triggered"]=bool(ra["top_cell_sse_fraction"]>.2 or ra["top_10_percent_cells_sse_fraction"]>.5)
    write_json(OUT/"residual_audit.json",residual_audit)
    pd.DataFrame(exp.records).to_csv(OUT/"solver_diagnostics.csv",index=False)
    # Descriptive paired group resampling of already computed cross-validation errors.
    # Overlapping CV training sets invalidate an iid test interpretation.
    resampling=[]
    rng2=np.random.default_rng(protocol["seed"])
    for name in ["leave_Q","leave_cell"]:
        part=direct[direct.protocol.eq(name)]
        group=[]
        for fold,g in part.groupby("fold"):
            pw=g.pivot(index="row_id",columns="model",values="prediction")
            yy=g.drop_duplicates("row_id").set_index("row_id").y.reindex(pw.index)
            group.append([np.sum((pw.power-yy)**2),np.sum((pw.curved-yy)**2),len(yy)])
        arr=np.array(group)
        for repeat in range(2000):
            samp=arr[rng2.integers(0,len(arr),len(arr))].sum(axis=0)
            gain=1-np.sqrt(samp[1]/samp[0])
            resampling.append(dict(protocol=name,repeat=repeat,relative_rmse_gain=gain))
    pd.DataFrame(resampling).to_csv(OUT/"descriptive_group_resampling.csv",index=False)
    # Preserve Q1's exact surface difference and the beta*lambda multiplier.
    previous=pd.read_csv(REF/"p_term_sensitivity.csv")
    ps=[]
    for m in MODELS:
        theta=full[m][0].theta
        for row in previous.itertuples(index=False):
            psi=float(row.q1_surface_psi_for_5pp_reallocation)
            dl=theta[2]*150**(-theta[4])*np.exp(log_quality(theta,np.log(row.Q0)))
            change=dl*np.expm1(theta[4]*row.lambda_p*psi)
            ps.append(dict(model=m,quality_baseline=row.quality_baseline,domain=row.domain,
                Q0=row.Q0,lambda_p=row.lambda_p,psi=psi,predicted_delta_loss=change))
    pd.DataFrame(ps).to_csv(OUT/"q1_transfer_sensitivity.csv",index=False)
    handoff=dict(schema="q2-quality-response-v26/1",selected_model=chosen,
        form="E+A*n^(-alpha)+B*d^(-beta)*exp(-delta*log(Q)+kappa*log(Q)^2+beta*lambda_mix*psi)",
        units={"n":"billion parameters","d":"billion tokens"},parameters=fullparams[chosen],
        B1_parameters=dict(zip(["E","A","B","alpha","beta"],map(float,f1.theta))),
        joint_parameter_draws="joint_parameter_draws.csv",joint_draw_filter={"model":chosen},
        support={"n":[float(b6.N_params_B.min()),float(b6.N_params_B.max())],
                 "d":[float(b6.D_tokens_B.min()),float(b6.D_tokens_B.max())],"Q":[.1,1.]},
        Q1_anchor_Q=q0,quality_scale_alignment="conditional assumption, not empirically identified",
        mix_transfer="psi=J_hat(p)-J_hat(p0)",lambda_mix_identified=False,
        downstream_status="not synchronized; Q3/Q4 must adopt g and g' explicitly if selected model is curved",
        evidence="B6 semi-synthetic; B7 historical source-related extension; B8 incompatible quality direction")
    write_json(OUT/"q2_handoff_v26.json",handoff)
    summary=dict(selection=decision,B1_validation=metrics(b1.val_loss,b1pred),
        B1_parameters=handoff["B1_parameters"],full_parameters=fullparams,
        baseline_reproduction_max_prediction_error=float(np.max(np.abs(
            predict(full["power"][0].theta,exp.data)-old_predict(np.array(list(old["B6_quality_parameters"].values())),exp.data)))),
        bootstrap_draws_per_model=protocol["bootstrap_draws"],unique_training_fits=len(exp.records),
        residual_audit=residual_audit,elapsed_seconds=perf_counter()-begin)
    write_json(OUT/"results.json",summary)
    def _label(p):
        """Package-relative label; attachments are labelled from the attachment root.

        The prefix must match the one Q3 rebuilds when it re-hashes B6
        (``real_attachments/B_scaling_laws/<name>``), otherwise the handoff
        fingerprint check cannot be evaluated.
        """
        for prefix, root in (("real_attachments", paths.ATTACHMENTS), ("", paths.PACKAGE)):
            try:
                rel = p.relative_to(root).as_posix()
                return f"{prefix}/{rel}" if prefix else rel
            except ValueError:
                continue
        return p.name
    inputs={_label(p):sha(p) for p in sources.values()}
    for p in [HERE/"protocol.json",REF/"results.json",REF/"p_term_sensitivity.csv"]:
        inputs[p.relative_to(paths.PACKAGE).as_posix()]=sha(p)
    manifest=dict(completed_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,
        platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
        threads=1,inputs_sha256=inputs,
        source_sha256={p.name:sha(p) for p in [HERE/"run_study.py",HERE/"quality_solver.py",HERE/"scaling_solver.py"]},
        output_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!="run_manifest.json"})
    write_json(OUT/"run_manifest.json",manifest)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()


