"""Generate publication tables/macros and the Overleaf package from saved results.

No model is fitted here. The only test-set operation is a documented post-hoc
paired bootstrap over the already-saved test labels and probabilities.
"""
from __future__ import annotations

import itertools
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
RESULTS = ROOT / "results"
TABLES = REPORT / "generated" / "tables"


def tex(value: object) -> str:
    """Escape ordinary text for a LaTeX table cell."""
    return (str(value).replace("\\", r"\textbackslash{}").replace("&", r"\&")
            .replace("%", r"\%").replace("_", r"\_").replace("#", r"\#"))


def table(headers: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(headers) - 1))
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(headers) + " " + chr(92) * 2, r"\midrule"]
    lines.extend(" & ".join(row) + " " + chr(92) * 2 for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def write_table(name: str, headers: list[str], rows: list[list[str]], align: str | None = None) -> None:
    (TABLES / name).write_text(table(headers, rows, align), encoding="utf-8")


def fmt(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}"


def pairwise_ap(audit: dict, final: dict, seed: int = 42, n_bootstrap: int = 1000) -> dict:
    """Paired, stratified bootstrap intervals using saved holdout predictions only."""
    saved = np.load(RESULTS / "test_probabilities.npz")
    y = saved["y_test"].astype(int)
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    models = final["models"]
    pairs = []
    for left, right in itertools.combinations(models, 2):
        left_key = f'{left["model"]}_{left["strategy"]}'
        right_key = f'{right["model"]}_{right["strategy"]}'
        p_left, p_right = saved[left_key], saved[right_key]
        point = float(average_precision_score(y, p_left) - average_precision_score(y, p_right))
        # Reset to the existing S4 seed for each pair: paired resamples, and the
        # first (CV top-two) interval reproduces the S4 bootstrap draw sequence.
        rng = np.random.default_rng(seed)
        differences = np.empty(n_bootstrap, dtype=float)
        for b in range(n_bootstrap):
            idx = np.concatenate((rng.choice(positives, len(positives), replace=True),
                                  rng.choice(negatives, len(negatives), replace=True)))
            differences[b] = (average_precision_score(y[idx], p_left[idx])
                              - average_precision_score(y[idx], p_right[idx]))
        pairs.append({
            "left_model": left_key, "right_model": right_key,
            "left_selection_basis": "train-only cross-validation; locked before holdout",
            "right_selection_basis": "train-only cross-validation; locked before holdout",
            "difference_left_minus_right_ap": point,
            "ci95_stratified_bootstrap": [float(np.quantile(differences, .025)),
                                           float(np.quantile(differences, .975))],
            "bootstrap_replicates": n_bootstrap,
            "seed": seed,
            "post_hoc_descriptive_only": True,
            "used_for_model_or_threshold_selection": False,
        })
    return {"description": "Post-hoc paired stratified-bootstrap AP differences from saved holdout predictions; descriptive intervals only.",
            "test_predictions_source": "results/test_probabilities.npz",
            "test_rows": int(len(y)), "test_frauds": int(len(positives)),
            "bootstrap_replicates": n_bootstrap, "seed": seed, "pairs": pairs}


def hypothesis_rows(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "---" in line or "Model/strategy" in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3:
            continue
        match = re.match(r"\s*(PARTLY CONFIRMED|CONFIRMED|INCONCLUSIVE|REFUTED)\b", cells[2], re.IGNORECASE)
        verdict = match.group(1).upper() if match else "REVIEW"
        rows.append([tex(cells[0]), tex(cells[1]), tex(cells[2]), tex(verdict)])
    return rows


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    generated = REPORT / "generated"
    figures = REPORT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / "figures").glob("*.pdf"):
        shutil.copy2(source, figures / source.name)

    audit = json.loads((RESULTS / "data_audit.json").read_text(encoding="utf-8"))
    config = json.loads((RESULTS / "config_snapshot.json").read_text(encoding="utf-8"))
    final = json.loads((RESULTS / "final_metrics.json").read_text(encoding="utf-8"))
    cv = pd.read_csv(RESULTS / "cv_summary.csv")
    pair_path = RESULTS / "final_pairwise.json"
    if pair_path.exists():
        pair_results = json.loads(pair_path.read_text(encoding="utf-8"))
    else:
        pair_results = pairwise_ap(audit, final, int(config["seed"]), 1000)
        pair_path.write_text(json.dumps(pair_results, indent=2) + "\n", encoding="utf-8")

    # Split table and data-derived prose macros.
    split_rows = []
    macros: dict[str, str | int] = {
        "DataRows": audit["rows_after_deduplication"],
        "FraudCount": audit["frauds_after_deduplication"],
    }
    for display, key, macro in (("Train", "train", "Train"),
                                ("Validation", "validation", "Validation"),
                                ("Test", "test", "Test")):
        part = audit["splits"][key]
        rate = part["frauds"] / part["rows"]
        start_h, end_h = part["time_min_seconds"] / 3600, part["time_max_seconds"] / 3600
        split_rows.append([display, f'{part["rows"]:,}', f'{part["frauds"]:,}',
                           f"{100*rate:.3f}\%", f"{start_h:.3f}--{end_h:.3f}"])
        macros[f"{macro}Rows"] = part["rows"]
        macros[f"{macro}Frauds"] = part["frauds"]
        macros[f"{macro}FraudRate"] = f"{100*rate:.3f}"
    write_table("split_metrics.tex", ["Split", "Rows", "Frauds", "Fraud rate", "Time (h)"], split_rows)

    # Cross-validation comparison: all saved model/strategy combinations.
    cv_rows = []
    for row in cv.itertuples(index=False):
        cv_rows.append([tex(row.model), tex(row.strategy),
                        rf"${row.mean:.3f} \pm {row.std:.3f}$", fmt(row.mean_fit_seconds)])
    write_table("cv_metrics.tex", ["Model", "Strategy", "AP (mean $\pm$ SD)", "Fit (s)"], cv_rows)
    cv_sorted = cv.sort_values("mean", ascending=False)
    macros["CVBestMean"] = fmt(float(cv_sorted.iloc[0]["mean"]), 3)
    macros["CVBestSD"] = fmt(float(cv_sorted.iloc[0]["std"]), 3)
    macros["CVSecondMean"] = fmt(float(cv_sorted.iloc[1]["mean"]), 3)
    macros["CVBestModel"] = tex(f'{cv_sorted.iloc[0].model}/{cv_sorted.iloc[0].strategy}')
    macros["CVFastestTopCandidateSeconds"] = fmt(float(cv.sort_values("mean_fit_seconds").iloc[0].mean_fit_seconds), 3)

    # Per-model hyperparameters come only from the saved configuration snapshot.
    model_settings = config["models"]
    strategy_labels = {"weights": "class weights", "smote": "SMOTE", "undersample": "undersampling", "none": "none"}
    hp_rows = []
    for row in cv.itertuples(index=False):
        settings = model_settings.get(row.model, {})
        hp_rows.append([tex(row.model), tex(strategy_labels.get(row.strategy, row.strategy)),
                        tex(", ".join(f"{key}={value}" for key, value in settings.items()) or "default")])
    write_table("hyperparameters.tex", ["Model", "Strategy", "Saved model settings"], hp_rows, "llp{0.61\\columnwidth}")

    # Two compact holdout tables, with confusion-count columns from the saved JSON.
    ranking_rows, threshold_rows = [], []
    for model in final["models"]:
        ci = model["ci95_stratified_bootstrap"]
        label = tex(f'{model["model"]}/{model["strategy"]}')
        ranking_rows.append([label,
                             f'{model["pr_auc"]:.3f} [{ci["pr_auc"][0]:.3f}, {ci["pr_auc"][1]:.3f}]',
                             f'{model["roc_auc"]:.3f} [{ci["roc_auc"][0]:.3f}, {ci["roc_auc"][1]:.3f}]'])
        tn, fp = model["confusion_matrix"][0]
        fn, tp = model["confusion_matrix"][1]
        threshold_rows.append([label, f'{model["threshold_f1_validation"]:.3f}',
                               f'{model["precision"]:.3f}', f'{model["recall"]:.3f}',
                               f'{model["f1"]:.3f}', f'{model["mcc"]:.3f}',
                               str(tp), str(fp), str(fn)])
    write_table("holdout_ranking.tex", ["Model/strategy", "AP [95\% CI]", "ROC-AUC [95\% CI]"], ranking_rows)
    write_table("holdout_thresholds.tex", ["Model/strategy", "Threshold", "Prec.", "Recall", "F1", "MCC", "TP", "FP", "FN"], threshold_rows)

    # Hypothesis scorecard, preserving the evidence text from the logged result file.
    write_table("hypotheses.tex", ["Model/strategy", "Hypothesis", "Observed evidence", "Verdict"],
                hypothesis_rows(RESULTS / "hypotheses.md"), "p{0.14\\columnwidth}p{0.22\\columnwidth}p{0.46\\columnwidth}p{0.12\\columnwidth}")

    # Feature table: top ten by SHAP, unioned with top ten by permutation importance.
    shap = pd.read_csv(RESULTS / "shap_importance.csv").sort_values("mean_absolute_shap_or_proxy", ascending=False)
    perm = pd.read_csv(RESULTS / "permutation_importance.csv").sort_values("importance_mean", ascending=False)
    eda = pd.read_csv(RESULTS / "eda_feature_ranking.csv")
    eda["abs_cliff"] = eda["cliffs_delta"].abs()
    eda["cliff_rank"] = eda["abs_cliff"].rank(method="min", ascending=False).astype(int)
    eda["mi_rank"] = eda["mutual_information"].rank(method="min", ascending=False).astype(int)
    top_features = list(dict.fromkeys(shap.head(10).feature.tolist() + perm.head(10).feature.tolist()))
    shap_map = shap.set_index("feature")["mean_absolute_shap_or_proxy"]
    perm_map = perm.set_index("feature")["importance_mean"]
    eda_map = eda.set_index("feature")
    feature_rows = []
    for feature in top_features:
        rank = eda_map.loc[feature]
        feature_rows.append([tex(feature), fmt(float(shap_map.get(feature, np.nan))),
                             fmt(float(perm_map.get(feature, np.nan))),
                             str(int(rank.cliff_rank)), str(int(rank.mi_rank))])
    write_table("feature_importance.tex", ["Feature", "Mean $|$SHAP$|$", "Permutation", "$|$Cliff$|$ rank", "MI rank"], feature_rows)
    shap_top = set(shap.head(10).feature)
    perm_top = set(perm.head(10).feature)
    macros["SHAPPermTopTenOverlap"] = len(shap_top & perm_top)
    macros["SHAPEDATopTenCliffOverlap"] = len(shap_top & set(eda.nsmallest(10, "cliff_rank").feature))

    # Error profile table and derived prose macros.
    profile = pd.read_csv(RESULTS / "validation_error_profile.csv").set_index("group")
    profile_rows = []
    for group, label in (("false_negative", "False negatives"), ("true_positive", "True positives")):
        item = profile.loc[group]
        profile_rows.append([label, str(int(item["count"])), fmt(float(item["amount_median"])), fmt(float(item["hour_median"]))])
        prefix = "FN" if group == "false_negative" else "TP"
        for field, suffix in (("count", "Count"), ("amount_median", "MedianAmount"), ("hour_median", "MedianHour")):
            macros[prefix + suffix] = fmt(float(item[field]), 3) if field != "count" else int(item[field])
    write_table("error_profile.tex", ["Validation group", "Count", "Median amount", "Median hour"], profile_rows)

    # Baseline, selection, threshold, cost, tuning, and pairwise prose values.
    primary_name = f'{cv_sorted.iloc[0].model}/{cv_sorted.iloc[0].strategy}'
    primary = next(m for m in final["models"] if f'{m["model"]}/{m["strategy"]}' == primary_name)
    macros.update({
        "PrimaryModel": tex(primary_name), "BestModel": tex(primary_name),
        "BestCVModel": tex(primary_name), "PrimaryCVAP": fmt(float(cv_sorted.iloc[0]["mean"])),
        "PrimaryHoldoutAP": fmt(primary["pr_auc"]), "PrimaryHoldoutAPLow": fmt(primary["ci95_stratified_bootstrap"]["pr_auc"][0]),
        "PrimaryHoldoutAPHigh": fmt(primary["ci95_stratified_bootstrap"]["pr_auc"][1]),
        "DummyAP": fmt(final["dummy_baseline"]["pr_auc"]),
        "AllLegitAccuracy": fmt(final["all_legit_accuracy"]),
        "TestRows": final["test_rows"], "TestFrauds": final["test_frauds"],
        "ValidationFraudRateRatio": fmt((audit["splits"]["validation"]["frauds"] / audit["splits"]["validation"]["rows"]) /
                                         (audit["splits"]["train"]["frauds"] / audit["splits"]["train"]["rows"]), 3),
        "TestFraudRate": fmt(100 * audit["splits"]["test"]["frauds"] / audit["splits"]["test"]["rows"], 3),
        "TestEvaluationCount": final["test_evaluation_count"],
    })
    for model in final["models"]:
        key = f'{model["model"].title()}{model["strategy"].title()}'
        macros[key + "Threshold"] = fmt(model["threshold_f1_validation"])
        macros[key + "StrictThreshold"] = fmt(model["threshold_precision_90_validation"])
    for pair in pair_results["pairs"]:
        left = pair["left_model"].replace("_", "").title()
        right = pair["right_model"].replace("_", "").title()
        tag = left + "Vs" + right
        macros[tag + "Difference"] = fmt(pair["difference_left_minus_right_ap"])
        macros[tag + "Low"] = fmt(pair["ci95_stratified_bootstrap"][0])
        macros[tag + "High"] = fmt(pair["ci95_stratified_bootstrap"][1])
    cost = pd.read_csv(RESULTS / "cost_sensitivity.csv")
    macros["ReviewCostMin"] = fmt(float(cost.review_cost_per_alert.min()), 0)
    macros["ReviewCostMax"] = fmt(float(cost.review_cost_per_alert.max()), 0)
    tuning = json.loads((RESULTS / "tuning_results.json").read_text(encoding="utf-8"))
    for item in tuning:
        tag = item["model"].title()
        macros[tag + "TunedCVAP"] = fmt(item["best_cv_pr_auc"])
        macros[tag + "TunedValidationAP"] = fmt(item["validation_pr_auc"])

    # Keep compatibility with existing report macros, but define the primary
    # model explicitly from CV rather than using the test-AP ranking.
    macros.update({
        "BestPRAUC": fmt(primary["pr_auc"]),
        "BestROCAUC": fmt(primary["roc_auc"]), "BestPrecision": fmt(primary["precision"]),
        "BestRecall": fmt(primary["recall"]), "BestFOne": fmt(primary["f1"]),
        "BestPRAUCLow": fmt(primary["ci95_stratified_bootstrap"]["pr_auc"][0]),
        "BestPRAUCHigh": fmt(primary["ci95_stratified_bootstrap"]["pr_auc"][1]),
        "RFAblationDrop": fmt(float(pd.read_csv(RESULTS / "ablation.csv").query("model == 'rf' and stage == 'A0_legacy_like'").f1.mean()
                                     - pd.read_csv(RESULTS / "ablation.csv").query("model == 'rf' and stage == 'A3_chronological'").f1.mean())),
        "XGBAblationDrop": fmt(float(pd.read_csv(RESULTS / "ablation.csv").query("model == 'xgb' and stage == 'A0_legacy_like'").f1.mean()
                                      - pd.read_csv(RESULTS / "ablation.csv").query("model == 'xgb' and stage == 'A3_chronological'").f1.mean())),
    })
    macro_lines = ["% Generated only from files in results/; do not edit by hand."]
    macro_lines.extend(rf"\newcommand{{\{name}}}{{{tex(value)}}}" for name, value in macros.items())
    (generated / "numbers.tex").write_text("\n".join(macro_lines) + "\n", encoding="utf-8")

    target = REPORT / "overleaf_upload.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in REPORT.rglob("*"):
            if path.is_file() and path != target:
                archive.write(path, path.relative_to(REPORT))
    print(f"Generated {len(cv_rows)} CV rows, {len(split_rows)} split rows, {len(pair_results['pairs'])} post-hoc pairs and report tables.")
    print(f"Saved {RESULTS / 'final_pairwise.json'} and rebuilt {target}.")


if __name__ == "__main__":
    main()
