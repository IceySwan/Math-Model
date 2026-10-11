"""Auditable inputs and as-of views. No fitting, imputation or fuzzy identity joins."""
from __future__ import annotations
import hashlib
import json
import re
import sys
from dataclasses import dataclass,field
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import paths
from data_config import CROOT,OUT,ORIGIN,LICENSES,SCORES,PROTOCOL,CUTOFFS


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def save_table(frame,name):
    OUT.mkdir(parents=True,exist_ok=True)
    frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')


def type_group(value):
    s=str(value).lower()
    if 'pretrained' in s:return 'pretrained'
    if 'chat models' in s or 'fine-tuned' in s:return 'posttrained'
    if 'base merges' in s:return 'merge'
    return 'other'


def repo_key(value):return str(value).strip().strip('/').casefold()


def version_key(value):
    parts=repo_key(value).split('/')
    if len(parts)!=2:return None
    # Numeric separators remain meaningful: 2.5 -> ('2','5'), never '25'.
    return (parts[0],tuple(re.findall(r'[a-z]+|\d+',parts[1])))


def read_sources():
    names=['leaderboard_cleaned.csv','epoch_all_ai_models.csv','loss_benchmark_bridge_expanded.csv','leaderboard_extended_timeseries.csv']
    c1,c4,c6,c3=[pd.read_csv(CROOT/n,low_memory=False) for n in names]
    source_sha={str((CROOT/n).relative_to(paths.ATTACHMENTS)):digest(CROOT/n) for n in names}
    c1['row_id']=np.arange(len(c1));c1['repo']=c1.Model.map(repo_key)
    c1['org']='hf:'+c1.repo.str.split('/').str[0]
    c1['date']=pd.to_datetime(c1['Submission Date'],errors='coerce')
    for col in SCORES:c1[col]=pd.to_numeric(c1[col],errors='coerce')
    c1['complete_scores']=c1[SCORES].notna().all(axis=1)&c1[SCORES].ge(0).all(axis=1)&c1[SCORES].le(100).all(axis=1)
    c1['S']=c1[SCORES].mean(axis=1).where(c1.complete_scores)
    c1['N']=pd.to_numeric(c1['#Params (B)'],errors='coerce')
    c1['logN']=np.log10(c1.N.where(c1.N>0))
    c1['group']=c1.Type.map(type_group)
    c1['license']=c1['Hub License'].fillna('').str.strip().str.lower()
    c1['licensed']=c1.license.isin(LICENSES)
    c1['tau']=(c1.date-pd.Timestamp(ORIGIN)).dt.days/365.2425
    c4['c4_id']=np.arange(len(c4));c4['pubdate']=pd.to_datetime(c4['Publication date'],errors='coerce')
    c4['year']=c4.pubdate.dt.year
    c4['weights_yes']=c4['Open model weights?'].fillna('').str.lower().eq('yes')
    for name,raw,scale in [('F','Training compute (FLOP)',1),('D','Training dataset size (total)',1e9),('N','Parameters',1e9)]:
        c4[name]=pd.to_numeric(c4[raw],errors='coerce')/scale
    def org(r):
        dev=str(r['Hugging Face developer id']).strip().casefold()
        if dev!='nan' and re.fullmatch(r'[\w.-]+',dev):return 'hf:'+dev
        return 'org:'+re.sub(r'\s+',' ',str(r.Organization).strip().casefold())
    c4['org']=c4.apply(org,axis=1)
    c4['metadata_date']=pd.to_datetime(c4['Last modified'],errors='coerce',utc=True).dt.tz_localize(None)
    return c1,c4,c6,c3,source_sha


def _match_frame(c1,c4):
    exact,tokens,basenames={},{},{}
    def put(table,key,ix):
        if key:table.setdefault(key,set()).add(ix)
    for ix,r in c4.iterrows():
        dev=str(r['Hugging Face developer id']).strip().casefold();aliases=set()
        if len(str(r.Model).split('/'))==2:aliases.add(repo_key(r.Model))
        if dev!='nan' and re.fullmatch(r'[\w.-]+',dev):aliases.add(repo_key(dev+'/'+str(r.Model)))
        aliases.update(repo_key(s) for s in re.findall(r'https?://huggingface\.co/([\w.-]+/[\w.-]+)',str(r.Link)))
        for name in aliases:
            put(exact,name,ix);put(tokens,version_key(name),ix)
        put(basenames,tuple(re.findall(r'[a-z]+|\d+',str(r.Model).casefold())),ix)
    records=[]
    for r in c1.itertuples():
        hit=exact.get(r.repo,set());method='exact_owner_path'
        if not hit:hit=tokens.get(version_key(r.repo),set());method='publisher_version_tokens'
        base=basenames.get(tuple(re.findall(r'[a-z]+|\d+',r.repo.split('/')[-1])),set())
        rec={'row_id':r.row_id,'repo':r.repo,'match_candidates':len(hit),'match_method':method if hit else 'none',
             'basename_candidates_diagnostic_only':len(base),'c4_id':np.nan}
        if len(hit)==1:
            ix=next(iter(hit));b=c4.loc[ix]
            rel=abs(r.N-b.N)/b.N if np.isfinite(b.N) and b.N>0 else np.nan
            rec.update(c4_id=int(ix),c4_model=b.Model,pubdate=b.pubdate,weights_yes=bool(b.weights_yes),
                       c4_N=b.N,F=b.F,D=b.D,parameter_relative_difference=rel,
                       parameter_match=bool(np.isfinite(rel) and rel<=.10),date_conflict=bool(r.date<b.pubdate))
        records.append(rec)
    return pd.DataFrame(records)


@dataclass
class IdentityIndex:
    c1:pd.DataFrame
    c4:pd.DataFrame
    full:pd.DataFrame
    cache:dict=field(default_factory=dict)

    def at(self,cutoff):
        key=str(pd.Timestamp(cutoff).date())
        if key not in self.cache:
            # A later alias cannot make an earlier unambiguous record ambiguous.
            available=self.c4[self.c4.pubdate.le(pd.Timestamp(cutoff))]
            self.cache[key]=_match_frame(self.c1,available)
        return self.cache[key]


def identity_table(c1,c4):return IdentityIndex(c1,c4,_match_frame(c1,c4))


def asof(c1,matches,cutoff,strict=False,keep_date_conflicts=False):
    """Filter available rows first, audit conflicting versions, then deduplicate."""
    cutoff=pd.Timestamp(cutoff)
    available_matches=matches.at(cutoff)
    frame=c1[c1.date.le(cutoff)&c1.complete_scores&c1.N.gt(0)].copy()
    # A repository/date with inconsistent evaluation or identity is not a unique observation.
    conflict_cols=['group','N','license',*SCORES]
    conflicts=frame.groupby(['repo','date'])[conflict_cols].nunique(dropna=False).gt(1).any(axis=1)
    bad=set(conflicts[conflicts].index)
    excluded=frame[[tuple(z) in bad for z in frame[['repo','date']].to_numpy()]].copy()
    frame=frame[[tuple(z) not in bad for z in frame[['repo','date']].to_numpy()]]
    frame=frame.sort_values(['date','row_id']).drop_duplicates(['repo','date']).drop_duplicates('repo',keep='last')
    frame=frame[frame.licensed]
    frame=frame.merge(available_matches.drop(columns='repo'),on='row_id',how='left',validate='one_to_one')
    if strict:
        frame=frame[frame.match_candidates.eq(1)&frame.weights_yes.eq(True)&frame.parameter_match.eq(True)&frame.pubdate.le(cutoff)]
        if keep_date_conflicts:
            frame['date']=frame[['date','pubdate']].max(axis=1)
            frame['tau']=(frame.date-pd.Timestamp(ORIGIN)).dt.days/365.2425
        else:frame=frame[frame.date_conflict.eq(False)]
        frame=frame[frame.group.isin(['pretrained','posttrained'])]
    else:frame=frame[frame.group.isin(['pretrained','posttrained','merge'])]
    return frame.reset_index(drop=True),excluded


def compute_asof(c4,cutoff,language_only=True):
    f=c4[c4.weights_yes&c4.pubdate.le(pd.Timestamp(cutoff))&c4.F.gt(0)].copy()
    if language_only:f=f[f.Domain.eq('Language')]
    return f.reset_index(drop=True)


def c8_index():
    """Parse every available version once; the cache is keyed by file SHA-256."""
    files=sorted((CROOT/'detailed_results').rglob('*.json'))
    manifest=[{'path':str(p.relative_to(CROOT)),'sha256':digest(p)} for p in files]
    sha=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    cache=OUT/'c8_index_manifest.json';table=OUT/'c8_all_versions.csv'
    if cache.exists() and table.exists():
        saved=json.loads(cache.read_text(encoding='utf-8'))
        if saved['source_sha256']==sha and saved['table_sha256']==digest(table):
            f=pd.read_csv(table);f['evaldate']=pd.to_datetime(f.evaldate,errors='coerce');return f,saved
    rows=[]
    groups={'BBH':('leaderboard_bbh','acc_norm,none'),'MATH Lvl 5':('leaderboard_math_hard','exact_match,none'),
            'GPQA':('leaderboard_gpqa','acc_norm,none'),'MUSR':('leaderboard_musr','acc_norm,none'),'MMLU-PRO':('leaderboard_mmlu_pro','acc,none')}
    def metric(obj,key):
        try:v=float(obj[key]);return v if np.isfinite(v) else np.nan
        except (KeyError,TypeError,ValueError):return np.nan
    for p,item in zip(files,manifest):
        found=re.search(r'(\d{4}-\d\d-\d\d)T(\d\d)-(\d\d)-(\d\d(?:\.\d+)?)',p.name)
        date=pd.Timestamp(f'{found[1]}T{found[2]}:{found[3]}:{found[4]}') if found else pd.NaT
        rec={**item,'directory':p.parent.name,'evaldate':date,'valid':False,'parse_error':''}
        try:obj=json.loads(p.read_text(encoding='utf-8'))
        except (json.JSONDecodeError,UnicodeDecodeError) as err:
            rec['parse_error']=type(err).__name__;rows.append(rec);continue
        if not isinstance(obj,dict) or not isinstance(obj.get('results'),dict):
            rec['parse_error']='InvalidResultSchema';rows.append(rec);continue
        res=obj['results'];ie=res.get('leaderboard_ifeval') or {}
        a,b=metric(ie,'prompt_level_strict_acc,none'),metric(ie,'inst_level_strict_acc,none')
        vals={'IFEval':50*(a+b) if np.isfinite(a+b) else np.nan}
        vals.update({name:100*metric(res.get(group) or {},key) for name,(group,key) in groups.items()})
        subtasks=(obj.get('group_subtasks') or {}).get('leaderboard_bbh',[]);sc=[];ww=[]
        for sub in subtasks:
            v=metric(res.get(sub) or {},'acc_norm,none')
            if not np.isfinite(v):v=metric(res.get(sub) or {},'acc,none')
            n=metric((obj.get('n-samples') or {}).get(sub) or {},'effective')
            if np.isfinite(v):sc.append(100*v);ww.append(n if np.isfinite(n) and n>0 else 0)
        rec.update(valid=True,repo=repo_key(obj.get('model_name') or obj.get('model_name_sanitized') or p.parent.name),
                   evaluation_versions=json.dumps(obj.get('versions') or {},sort_keys=True),n_bbh_subtasks=len(sc),
                   BBH_macro=float(np.mean(sc)) if sc else np.nan,
                   BBH_n_weighted=float(np.average(sc,weights=ww)) if sum(ww)>0 else np.nan,**vals)
        rows.append(rec)
    f=pd.DataFrame(rows);save_table(f,'c8_all_versions.csv')
    info={'schema_version':1,'source_sha256':sha,'files':len(paths),'invalid_files':int((~f.valid).sum()),
          'table_sha256':digest(table),'files_manifest':manifest}
    write_json(cache,info)
    return f,info


def c8_asof(index,cutoff):
    # A date-only cutoff includes the complete date but no following day.
    f=index[index.valid.eq(True)&index.evaldate.lt(pd.Timestamp(cutoff)+pd.Timedelta(days=1))]
    return f.sort_values(['evaldate','path']).drop_duplicates('repo',keep='last').copy()


def audit_inputs():
    """Audit the bundled C1/C3/C4/C6/C8 inputs and write the as-of tables.

    The Q2/Q3 handoff contract is written by ``resource_forecast.upstream``.
    """
    OUT.mkdir(parents=True,exist_ok=True);write_json(OUT/'protocol.json',PROTOCOL)
    c1,c4,c6,c3,hashes=read_sources();matches=identity_table(c1,c4)
    save_table(matches.full,'identity_audit.csv')
    save_table(matches.at(ORIGIN),'identity_asof_origin.csv')
    strict,conflicts=asof(c1,matches,ORIGIN,strict=True);broad,_=asof(c1,matches,ORIGIN)
    sensitivity,_=asof(c1,matches,ORIGIN,strict=True,keep_date_conflicts=True)
    save_table(strict,'strict_primary.csv');save_table(broad,'licensed_primary.csv')
    save_table(conflicts,'conflicting_repository_versions.csv');save_table(sensitivity,'strict_max_date_sensitivity.csv')
    index,cache=c8_index();c8=c8_asof(index,ORIGIN);save_table(c8,'c8_asof_origin.csv')
    snapshots=[]
    for cutoff in [*CUTOFFS,ORIGIN]:
        selected=c8_asof(index,cutoff)
        snapshots.append({'cutoff':cutoff,'selected_models':len(selected),'last_eval_date':str(selected.evaldate.max()),
                          'max_future_files_selected':int((selected.evaldate>=pd.Timestamp(cutoff)+pd.Timedelta(days=1)).sum())})
    save_table(pd.DataFrame(snapshots),'c8_cutoff_audit.csv')
    pairs=broad.merge(c8,on='repo',suffixes=('_C1','_C8'),validate='one_to_one');metrics=[]
    for s in SCORES:
        p=pairs[[s+'_C1',s+'_C8']].dropna();err=p[s+'_C1']-p[s+'_C8']
        metrics.append({'task':s,'matched':len(p),'mean_C1_minus_C8_pp':float(err.mean()),'rmse_pp':float(np.sqrt(np.mean(err**2)))})
    save_table(pd.DataFrame(metrics),'c8_c1_metric_comparison.csv')
    compute=compute_asof(c4,ORIGIN);save_table(compute,'c4_language_training_compute.csv')
    scorecols=['IFEval','BBH','MATH_Lvl5','GPQA','MUSR','MMLU_PRO']
    historical=c3.Source.fillna('').str.contains('Historical',case=False)
    structural=historical&c3[scorecols].eq(0).any(axis=1)
    summary={'raw_rows':{'C1':len(c1),'C3':len(c3),'C4':len(c4),'C6':len(c6)},
             'strict_primary_rows':len(strict),'strict_groups':{str(k):int(v) for k,v in strict.group.value_counts().items()},
             'strict_organizations':strict.org.nunique(),'licensed_primary_rows':len(broad),
             'conflicting_rows_excluded':len(conflicts),'conflicting_repos':sorted(conflicts.repo.unique().tolist()),
             'date_conflict_additional_rows_in_sensitivity':len(sensitivity)-len(strict),
             'C1_complete_scores':int(c1.complete_scores.sum()),
             'C1_average_max_abs_error':float(abs(c1.S-pd.to_numeric(c1['Average ⬆️'],errors='coerce')).max()),
             'C3_historical_rows_with_zero_placeholders_excluded':int(structural.sum()),'C3_used_for_fitting':False,
             'C4_language_compute_rows':len(compute),'C4_language_complete_FDN':int((compute.D.gt(0)&compute.N.gt(0)).sum()),
             'C4_metadata_modified_after_origin':int(c4.metadata_date.gt(pd.Timestamp(ORIGIN)).sum()),
             'validation_scope':'retrospective date cutoffs; historical metadata availability unverified',
             'C8_files':cache['files'],'C8_invalid_files':cache['invalid_files'],'C8_models_asof_origin':len(c8),
             'C8_used_for_fitting':False,'source_sha256':hashes,'C8_source_sha256':cache['source_sha256']}
    write_json(OUT/'data_audit.json',summary)
    return c1,c4,c6,matches,strict,broad,summary


if __name__=='__main__':
    *_,summary=audit_inputs();print(json.dumps(summary,ensure_ascii=False,indent=2))
