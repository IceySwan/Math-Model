"""Q1 mixture orchestration: fit A4/A5, freeze, then evaluate historical tables."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures
from threadpoolctl import threadpool_limits

from mixture_solver import SimplexFeatures, train_solver, prediction_metrics
from mixture_effects import build_probe_design, domain_effects, bootstrap_effects, response_surface_export


def rank_metrics(actual, predicted):
    first, oracle = int(np.argmin(predicted)), int(np.argmin(actual))
    order = np.argsort(predicted, kind="stable")
    return dict(
        spearman=float(spearmanr(actual, predicted).statistic) if np.ptp(predicted) > 1e-14 else None,
        kendall=float(kendalltau(actual, predicted).statistic) if np.ptp(predicted) > 1e-14 else None,
        selected_row_1based=first + 1, oracle_row_1based=oracle + 1,
        selected_actual_objective=float(actual[first]), oracle_actual_objective=float(actual[oracle]),
        top1_regret_pct=float(100 * (actual[first] / actual[oracle] - 1)),
        top10_actual_mean=float(actual[order[:10]].mean()),
        top10_mean_regret_pct=float(100 * (actual[order[:10]].mean() / np.sort(actual)[:10].mean() - 1)),
        top10_mean_vs_all_pct=float(100 * (actual[order[:10]].mean() / actual.mean() - 1)))


def analyze_mixtures(load_pair, quality_mapping, quality_summary, out_dir,
                     *, jobs=4, repeats=3, refresh=False, n_boot=80):
    out = Path(out_dir)
    started = perf_counter()
    x, raw, mcols, ycols, train_index, train_meta = load_pair(
        "train_mixture_1m.csv", "train_pile_loss_1m.csv")
    domains = [c.removeprefix("train_the_pile_") for c in mcols]
    # train_solver never sees any external mixtures or labels.
    fitted = train_solver(x, raw, out, jobs=jobs, repeats=repeats, reuse=not refresh)
    selected, quad_name = fitted.selected_name, fitted.quadratic_name
    baseline = fitted.selected.target_scale_
    normalized_y = raw / baseline
    selected_model, quadratic = fitted.selected.estimator_, fitted.quadratic.estimator_
    fitted.cv.to_csv(out / "mixture_cv.csv", index=False)
    fitted.cv.to_csv(out / "mixture_cv_repeated.csv", index=False)
    pd.DataFrame({"training_row": np.arange(len(x)), "index": train_index}).to_csv(
        out / "solver_training_indices.csv", index=False)
    pd.DataFrame({"loss_column": ycols, "A5_training_mean": baseline}).to_csv(
        out / "training_target_scales.csv", index=False)
    with threadpool_limits(limits=1):
        legacy = clone(fitted.models["legacy_extra_trees"]).fit(x, normalized_y)
        probes = build_probe_design(x, domains)
        simple, pairs = domain_effects(quadratic, x, domains, probes=probes, out_dir=out)
        boot = bootstrap_effects(fitted.models[quad_name], x, normalized_y, domains,
                                 n_boot, probes=probes, out_dir=out)
        selected_boot = bootstrap_effects(fitted.models[selected], x, normalized_y, domains,
                                          n_boot, probes=probes)
        selected_boot.to_csv(out / "effect_bootstrap_selected.csv", index=False, encoding="utf-8-sig")
    train_meta.update(unique_recipes=int(len(np.unique(x, axis=0))),
                      simplex_affine_rank=int(np.linalg.matrix_rank(x - x.mean(axis=0))),
                      quadratic_features_original=PolynomialFeatures(2, include_bias=False).fit(x).n_output_features_,
                      quadratic_features_contrast=PolynomialFeatures(2, include_bias=False).fit(
                          SimplexFeatures("contrast").fit_transform(x)).n_output_features_)
    neighbors = NearestNeighbors(n_neighbors=1).fit(x)
    evaluators = {selected: selected_model, quad_name: quadratic, "legacy_extra_trees": legacy}
    validators = [
        ("1M", "test_mixture_1m.csv", "test_pile_loss_1m.csv", "measured"),
        ("60M", "test_mixture_60m.csv", "test_pile_loss_60m.csv", "measured"),
        ("1B", "test_mixture_1B.csv", "test_pile_loss_1B.csv", "measured"),
        ("10B", "est_mixture_10b.csv", "est_pile_loss_10b.csv", "estimated"),
        ("70B", "est_mixture_70b.csv", "est_pile_loss_70b.csv", "estimated"),
    ]
    reports, detailed, predictions, arrays, selected_details = [], [], [], {}, None
    source_hashes = dict(train_meta["source_sha256"])
    for scale, mfile, yfile, provenance in validators:
        p, y_raw, _, _, index, meta = load_pair(mfile, yfile, mcols, ycols)
        source_hashes.update(meta["source_sha256"])
        distances = neighbors.kneighbors(p)[0][:, 0]
        arrays[scale] = (p, y_raw, index)
        truth_fixed = (y_raw / baseline).mean(axis=1)
        truth_self = (y_raw / y_raw.mean(axis=0)).mean(axis=1)
        for name, model in evaluators.items():
            with threadpool_limits(limits=1):
                prediction = model.predict(p)
            objective = prediction.mean(axis=1)
            diagnostic = rank_metrics(truth_self, objective)
            fixed = rank_metrics(truth_fixed, objective)
            first, oracle = int(np.argmin(objective)), int(np.argmin(truth_fixed))
            metrics = prediction_metrics(y_raw, prediction * baseline, baseline)
            report = dict(diagnostic, scale=scale, provenance=provenance, model=name,
                          evaluation_role="historical_reference_not_new_blind_test",
                          n=len(p), mixture_source=mfile, loss_source=yfile,
                          raw_sum_min=meta["raw_sum_min"], raw_sum_max=meta["raw_sum_max"],
                          nearest_train_distance_median=float(np.median(distances)),
                          selected_index=index[first].item() if isinstance(index[first], np.generic) else index[first],
                          oracle_fixed_a5_index=index[oracle].item() if isinstance(index[oracle], np.generic) else index[oracle],
                          **{f"fixed_a5_{key}": value for key, value in fixed.items()})
            # Absolute 1M surrogate errors are meaningful on the held-out 1M table.
            # Cross-scale tables are evaluated for ordering, without level calibration.
            if scale == "1M":
                report.update({key: metrics[key] for key in ("macro_rmse", "pooled_rmse", "mae", "objective_rmse")})
                report["macro_r2"] = float(r2_score(y_raw / baseline, prediction))
                for j, col in enumerate(ycols):
                    detailed.append(dict(loss_domain=col.removeprefix("metric/the_pile_").removesuffix("_val_loss"),
                                         model=name, rmse=float(np.sqrt(np.mean((y_raw[:, j] / baseline[j] - prediction[:, j])**2))),
                                         r2=float(r2_score(y_raw[:, j] / baseline[j], prediction[:, j]))))
            reports.append(report)
            for row in range(len(p)):
                predictions.append(dict(scale=scale, provenance=provenance, model=name,
                                        mixture_source=mfile, loss_source=yfile, index=index[row],
                                        row_1based=row + 1, observed_J_fixed_a5=truth_fixed[row],
                                        observed_J_table_normalized=truth_self[row], predicted_J_fixed_a5=objective[row],
                                        nearest_train_distance=distances[row],
                                        **{f"predicted_raw_{col}": prediction[row, j] * baseline[j] for j, col in enumerate(ycols)}))
            if scale == "1M" and name == selected:
                chosen = first
                selected_details = dict(source_file=mfile, loss_file=yfile, index=report["selected_index"],
                                        row_1based=first + 1, predicted_J_fixed_a5=float(objective[first]),
                                        historical_observed_J_fixed_a5=float(truth_fixed[first]),
                                        candidate_pool_size=len(p), nearest_train_distance=float(distances[first]),
                                        selection_uses_labels=False)
                fig, ax = plt.subplots(figsize=(5.8, 5.2))
                ax.scatter(truth_fixed, objective, s=12, alpha=.62, color="#376c9c")
                endpoints = [min(truth_fixed.min(), objective.min()), max(truth_fixed.max(), objective.max())]
                ax.plot(endpoints, endpoints, "--", color=".5", linewidth=1)
                ax.set(xlabel="Observed J (fixed A5 normalization)", ylabel="Predicted J",
                       title="1M historical reference")
                fig.tight_layout()
                fig.savefig(out / "mixture_1m_validation.pdf")
                plt.close(fig)
        print(f"Evaluated historical {scale}: {len(p)} recipes ({provenance})", flush=True)
    pd.DataFrame(reports).to_csv(out / "mixture_validation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(detailed).to_csv(out / "mixture_domain_errors.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(predictions).to_csv(out / "historical_predictions.csv", index=False, encoding="utf-8-sig")
    # Verify the estimated tables describe the same indexed recipes, not just row order.
    p10, _, ix10 = arrays["10B"]
    p70, _, ix70 = arrays["70B"]
    order70 = pd.Index(ix70).get_indexer(ix10)
    if np.any(order70 < 0) or len(ix10) != len(ix70) or not np.allclose(p10, p70[order70], rtol=0, atol=1e-12):
        raise ValueError("10B/70B estimated recipes differ after index matching")
    transfer = []
    for scale in ("10B", "70B"):
        pe, ye, _ = arrays[scale]
        matched = []
        for row in pe:
            hit = np.flatnonzero(np.all(np.isclose(x, row, rtol=0, atol=1e-9), axis=1))
            if len(hit) != 1:
                raise ValueError(f"{scale} estimated recipe is not a unique training subset row")
            matched.append(int(hit[0]))
        ref = raw[matched]
        transfer.append(dict(scale=scale, domain="mean_13",
                             spearman_1m_train_vs_est=float(spearmanr((ref / ref.mean(axis=0)).mean(axis=1),
                                                                    (ye / ye.mean(axis=0)).mean(axis=1)).statistic)))
        for j, col in enumerate(ycols):
            transfer.append(dict(scale=scale, domain=col.removeprefix("metric/the_pile_").removesuffix("_val_loss"),
                                 spearman_1m_train_vs_est=float(spearmanr(ref[:, j], ye[:, j]).statistic)))
    pd.DataFrame(transfer).to_csv(out / "estimated_transfer_diagnostics.csv", index=False)
    p1m = arrays["1M"][0]
    pd.DataFrame({"domain": domains, "share": p1m[chosen]}).to_csv(
        out / "selected_1m_test_candidate.csv", index=False, encoding="utf-8-sig")
    (out / "selected_candidate_provenance.json").write_text(json.dumps(selected_details, indent=2), encoding="utf-8")
    with threadpool_limits(limits=1):
        surface = response_surface_export(selected_model, quadratic, x, domains, p1m[chosen], probes=probes)
    surface.to_csv(out / "response_surface_evaluations.csv", index=False, encoding="utf-8-sig")
    # Quantify whether the less accurate quadratic interpretation agrees with
    # the selected nonlinear predictor at exactly the same simplex probes.
    selected_single, selected_pairs = domain_effects(selected_model, x, domains, probes=probes)
    selected_single.to_csv(out / "domain_reallocation_effects_selected.csv", index=False, encoding="utf-8-sig")
    selected_pairs.to_csv(out / "domain_pair_interactions_selected.csv", index=False, encoding="utf-8-sig")
    agreement_single = simple.merge(selected_single, on="domain", suffixes=("_quadratic", "_selected"))
    agreement_single = agreement_single.rename(columns={"domain": "domain_i",
        "loss_change_5pct_reallocation_quadratic": "quadratic_effect",
        "loss_change_5pct_reallocation_selected": "selected_effect"})
    agreement_single["domain_j"] = ""
    agreement_pair = pairs.merge(selected_pairs, on=["domain_i", "domain_j"], suffixes=("_quadratic", "_selected"))
    agreement_pair = agreement_pair.rename(columns={"interaction_loss_quadratic": "quadratic_effect",
                                                    "interaction_loss_selected": "selected_effect"})
    cols = ["domain_i", "domain_j", "quadratic_effect", "selected_effect"]
    agreement = pd.concat([agreement_single[cols], agreement_pair[cols]], ignore_index=True).merge(
        boot, on=["domain_i", "domain_j"], validate="one_to_one")
    agreement = agreement.merge(selected_boot.rename(columns={col: f"selected_{col}" for col in
        ("effect_median", "effect_ci025", "effect_ci975", "fraction_negative")}),
        on=["domain_i", "domain_j"], validate="one_to_one")
    agreement["sign_agrees"] = np.sign(agreement.quadratic_effect) == np.sign(agreement.selected_effect)
    agreement["quadratic_interval_excludes_zero"] = (agreement.effect_ci025 > 0) | (agreement.effect_ci975 < 0)
    agreement["selected_interval_excludes_zero"] = (agreement.selected_effect_ci025 > 0) | (agreement.selected_effect_ci975 < 0)
    agreement["stable_across_both_models"] = (agreement.sign_agrees
        & agreement.quadratic_interval_excludes_zero & agreement.selected_interval_excludes_zero
        & (np.sign(agreement.quadratic_effect) == np.sign(agreement.effect_ci025))
        & (np.sign(agreement.selected_effect) == np.sign(agreement.selected_effect_ci025)))
    agreement.to_csv(out / "effect_model_agreement.csv", index=False, encoding="utf-8-sig")
    (out / "mixture_input_manifest.json").write_text(json.dumps(source_hashes, indent=2), encoding="utf-8")
    handoff = dict(schema_version=2, output_target="J = equal mean of raw loss / full A5 means",
                   selected_model=selected, quadratic_model=quad_name, anchor_training_row=probes.anchor_index,
                   anchor_index=int(train_index[probes.anchor_index]), bootstrap_resamples=n_boot,
                   bootstrap_target_scale="fixed full A5 objective; scaler and Ridge refitted in each resample",
                   bootstrap_scope="conditional on quadratic model and hyperparameters; not post-selection coverage",
                   effect_definition="single: (1-delta)*p0+delta*e_i; pair: (1-2*delta)*p0+delta*(e_i+e_j)",
                   delta=.05, single_own_share_increment="delta*(1-p0_i), not absolute 5 percentage points",
                   interpretation="local surrogate association; no causal conclusion",
                   validation_status="historical reference sets; not a fresh blind test",
                   effect_sign_agreement_single=int(agreement.loc[agreement.domain_j.eq(""), "sign_agrees"].sum()),
                   effect_sign_agreement_pair=int(agreement.loc[agreement.domain_j.ne(""), "sign_agrees"].sum()),
                   stable_effects_across_models=int(agreement.stable_across_both_models.sum()),
                   effect_transfer_rule="inspect selected and quadratic conditional bootstrap; only stable_across_both_models directions have shared sign support",
                   selected_candidate=selected_details)
    (out / "q2_handoff_metadata.json").write_text(json.dumps(handoff, indent=2), encoding="utf-8")
    return dict(train=train_meta, training_target=handoff["output_target"], selected_model=selected,
                quadratic_interpretation_model=quad_name, cv=fitted.cv.to_dict(orient="records"),
                selection=fitted.metadata, validation=reports, selected_candidate=selected_details,
                selected_candidate_row_1based=chosen + 1, top_reallocation_effects=simple.head(5).to_dict(orient="records"),
                top_complementary_pairs=pairs.head(5).to_dict(orient="records"),
                estimated_transfer=transfer, response_surface_export="response_surface_evaluations.csv",
                mapping=quality_mapping(quality_summary, domains), elapsed_seconds=perf_counter() - started)
