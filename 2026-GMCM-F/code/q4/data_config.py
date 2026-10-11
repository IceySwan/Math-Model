"""Frozen analysis choices, defined before looking at held-out scores."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import paths

CROOT=paths.C
OUT=paths.Q4
ORIGIN='2025-03-13'
SELECTION_CUTOFF='2024-12-13'
CUTOFFS=['2024-09-13','2024-10-13','2024-11-13','2024-12-13','2025-01-13','2025-02-13']
HORIZON_MONTHS=[1,3]
QUANTILES=[.85,.90]
SEED=20260924
SCORES=['IFEval','BBH','MATH Lvl 5','GPQA','MUSR','MMLU-PRO']
BRIDGE_SCORES=['LB_IFEval','LB_BBH','LB_MATH','LB_GPQA','LB_MUSR','LB_MMLU_PRO']
LICENSES={'apache-2.0','mit','bsd-2-clause','bsd-3-clause','bsd-3-clause-clear','cc-by-4.0',
          'cc-by-sa-4.0','cc0-1.0','isc','mpl-2.0'}
FRONTIER_MODELS=['type_empirical_180d','time_linear_quantile','size_time_linear_quantile',
                 'bounded_time_quantile','bounded_size_time_quantile']
PROTOCOL={
 'origin':ORIGIN,'model_selection_uses_targets_no_later_than':SELECTION_CUTOFF,
 'rolling_cutoffs':CUTOFFS,'horizon_months':HORIZON_MONTHS,'quantiles':QUANTILES,
 'primary_quantile':.85,'frontier_candidates':FRONTIER_MODELS,
 'bridge_candidates':['training_mean','logN_linear','loss_linear','loss_ridge10'],
 'bridge_high_validation':'leave one model out','bridge_medium_validation':'leave publisher out',
 'minimum_frontier_train_rows':10,'empirical_recent_days':180,'empirical_type_minimum':3,
 'primary_identity':'unique full owner/path or publisher-and-version token match; basename alone excluded',
 'identity_availability':'build alias candidates from C4 records published by each cutoff before resolving ambiguity',
 'parameter_match_relative_tolerance':.10,
 'date_conflicts':'exclude from primary; keep a separate max(submission,publication) sensitivity',
 'duplicates':'same repository/date with conflicting type, N, license or task scores excluded; identical copies collapsed; latest valid date per repository',
 'score_target':'complete six-task unweighted mean on C1; no C8 score replacement',
 'rolling_target':'new repositories submitted during the future calendar window',
 'rolling_primary_inputs':'calendar date and declared type; N projected from training-only C4 trend and training type q80 N; future observed N used only in a separate conditional diagnostic',
 'overlapping_windows':'pooled forecast errors are dependent; report distinct repositories and paired publisher bootstrap, not an IID significance test',
 'validation_kind':'retrospective publication/submission cutoffs; metadata availability at the time is unobserved',
 'selection_rule':'minimum pooled development pinball loss at 0.85; fixed candidate order breaks exact ties',
 'failed_fit_policy':'record failure and use the training-only empirical quantile fallback',
 'bounded_transform':'logit(clip(S/100, 1e-4, 1-1e-4)); prediction=100*expit(z)',
 'compute_primary':'C4 open weights Yes, Domain exactly Language, publication no later than cutoff',
 'growth_frontier':'annual 90th percentile of training FLOPs, complete calendar years 2018..cutoff.year-1',
 'joint_bootstrap_draws':200,'bootstrap_unit':'pooled union of C1/C4 publisher keys; shared counts per draw',
 'future_horizon_years':[1,2],'compute_growth_factors':[0.,.5,1.],
 'trend_decay_scenarios':[0.,.5,1.],
 'q3_reference_total_budget':1e22,'q3_reference_context':8192,
 'q3_costs':['exponential','power','logarithmic'],
 'q3_transfer':'match future training FLOPs to Q3 training component; infer total budget; never map B6 Loss to Pythia scores',
 'tex_changes':False,'seed':SEED,
}
