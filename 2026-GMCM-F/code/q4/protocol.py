"""Fixed Q4 protocol: all analysis choices were declared before the evaluation."""
import json,hashlib,sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import paths

OUT=paths.Q4
ORIGIN='2025-03-13'
QS=(.05,.85,.95)
GROUPS=('pretrained','posttrained')
BOX={'logN':[-2.,4.],'tau':[-2.,2.]}
PROTOCOL={
 'stage':'v26 theory-driven Q4; retrospective exploratory comparison',
 'origin':ORIGIN,'independent_unseen_holdout':False,
 'development_end':'2024-12-13','cutoffs':['2024-09-13','2024-10-13','2024-11-13','2024-12-13','2025-01-13','2025-02-13'],
 'months':[1,3],'quantiles':list(QS),'primary_quantile':.85,
 'candidates':['joint_time','joint_size_time'],
 'selection':'minimum development q85 resource-forecast pinball; time first breaks ties; no promotion from reused late results alone',
 'score_scale':'C1 complete six-task unweighted average /100; raw-scale pinball',
 'auxiliary':'same C1 target, licensed pre/post only, exclude strict training repositories and held-out publishers; not new strict evidence',
 'weights':'each observed publisher has equal total mass, normalized to sum one within each source',
 'auxiliary_ridge':.01,'strict_slope_penalty':1.,'penalty_sensitivity':[.1,10.],
 'slope_penalty':'squared distance to auxiliary standardized slopes; same slopes for all quantiles, six type/quantile intercepts',
 'prediction_domain':BOX,'coefficient_box':[-10.,10.],
 'constraints':'quantile order and 0..1 bounds at every rectangle vertex for each type; no posthoc sorting',
 'solver':'convex quadratic epigraph, analytic gradients, SLSQP; explicit box Lagrangian lower bound; tolerance 1e-7 normalized objective',
 'calibration':'chronological half split by repository count, split-date ties kept together; hold all calibration publishers out of both fitting sources; model not refit after calibration',
 'calibration_inputs':'calendar date/type, N projected from proper-training C4 and strict reference N; no future actual N in primary forecast',
 'intervals':'unadjusted split fit, row rank correction and publisher-max rank correction at alpha=.1; nonnegative corrections; if rank exceeds count use [0,100]',
 'interval_validity':'rank theorem requires exchangeability, absent for these temporal data; only empirical retrospective coverage reported',
 'comparators':['type_empirical_180d','size_time_linear_quantile','auxiliary_size_time_ordered'],
 'diagnostics':['observed_N','nonoverlapping_one_month','publisher_holdout','publisher_equal_metrics','penalty_sensitivity','five_task_without_MATH'],
 'C3':'audit historical and leaderboard layers separately; audit C1 duplication; unverified historical zeros not numerical training data',
 'C8':'all local evaluation files, full owner/path, dates, revisions, task metric/version/shot/config hash; normalization before aggregation, no replacement of C1',
 'bootstrap_draws':200,'seed':20260928,
 'bootstrap':'shared publisher counts for C1 auxiliary, strict, reference N and C4; independent B6 whole parameter vectors paired by original draw id',
 'bootstrap_missing_type':'exclude and log a draw if a strict model type has no positive-weight observations; its type intercept is unidentified',
 'Q3':'v26 curved quality model; training FLOPs target matched, own C=1e22 baseline, context8192,p0,three costs',
 'forecasts':'conditional 1/2-year paths from 2025-03-13 with growth multipliers 0,.5,1 and residual-trend decay 0,.5,1; B6 Loss never mapped to Pythia score',
 'paper_update':False,
}

def fs_path(p):
    p=Path(p).resolve()
    return Path('\\\\?\\'+str(p)) if len(str(p))>=248 and not str(p).startswith('\\\\?\\') else p
def digest(p):return hashlib.sha256(fs_path(p).read_bytes()).hexdigest()
def write_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def save(df,name):
    OUT.mkdir(parents=True,exist_ok=True);df.to_csv(OUT/name,index=False,encoding='utf-8-sig')
