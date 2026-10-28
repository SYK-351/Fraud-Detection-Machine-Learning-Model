"""Restyle saved result plots for readable IEEE column and page widths.

This script reads cached results only. It never fits a predictive model.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
COLORS = {"xgb": "#4477AA", "catboost": "#CC6677", "rf": "#228833"}
plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
    "figure.titlesize": 15,
})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cv = pd.read_csv(RESULTS / "cv_summary.csv")
    fig, ax = plt.subplots(figsize=(7, 4.8))
    strategy_colors = {"weights": "#4477AA", "smote": "#CC6677", "undersample": "#228833", "none": "#AA4499"}
    for strategy, group in cv.groupby("strategy"):
        ax.errorbar(group["mean"], group["model"], xerr=group["std"], fmt="o", capsize=4,
                    label=strategy, color=strategy_colors.get(strategy, "#666666"))
    ax.set_xlabel("Cross-validation average precision")
    ax.set_title("Train-only five-fold average precision by model and strategy")
    ax.legend(title="Imbalance strategy", ncol=2)
    save(fig, "f07_cv_pr_auc")

    final = json.loads((RESULTS / "final_metrics.json").read_text(encoding="utf-8"))
    saved = np.load(RESULTS / "test_probabilities.npz")
    y = saved["y_test"].astype(int)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.axhline(float(y.mean()), color="#666666", linestyle="--", label="Test prevalence")
    for model in final["models"]:
        key = f'{model["model"]}_{model["strategy"]}'
        p = saved[key]
        precision, recall, _ = precision_recall_curve(y, p)
        color = COLORS[model["model"]]
        ax.plot(recall, precision, color=color, label=f'{model["model"]} / {model["strategy"]}')
        threshold = model["threshold_f1_validation"]
        cm = np.asarray(model["confusion_matrix"])
        tp, fp, fn = int(cm[1, 1]), int(cm[0, 1]), int(cm[1, 0])
        op_precision = tp / (tp + fp) if tp + fp else 0.0
        op_recall = tp / (tp + fn) if tp + fn else 0.0
        ax.scatter([op_recall], [op_precision], color=color, edgecolor="black", zorder=4)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision",
           title="Single holdout precision-recall curves")
    ax.legend(loc="upper right")
    save(fig, "f08_test_pr_curves")

    fig, axes = plt.subplots(1, len(final["models"]), figsize=(6.4, 3.8))
    for ax, model in zip(np.atleast_1d(axes), final["models"]):
        cm = np.asarray(model["confusion_matrix"], dtype=int)
        ax.imshow(cm, cmap="Blues", vmin=0)
        ax.set_title(f'{model["model"]} / {model["strategy"]}', fontsize=14)
        ax.set_xticks((0, 1), labels=("0", "1"))
        ax.set_yticks((0, 1), labels=("0", "1"))
        if ax is axes[0]:
            ax.set_ylabel("Actual class")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=14,
                    color="white" if value > cm.max() * .5 else "black")
    save(fig, "f09_confusion_matrices")

    ablation = pd.read_csv(RESULTS / "ablation.csv")
    labels = {"A0_legacy_like": "A0 legacy-like", "A1_fold_local_scale": "A1 fold-local",
              "A2_validation_threshold": "A2 validation threshold", "A3_chronological": "A3 chronological"}
    fig, ax = plt.subplots(figsize=(7, 4.8))
    stats = ablation.groupby(["model", "stage"]).f1.agg(["mean", "std"]).reset_index()
    stage_order = list(labels)
    for model, group in stats.groupby("model"):
        group = group.set_index("stage").reindex(stage_order).dropna().reset_index()
        ax.errorbar(group["mean"], [labels[x] for x in group.stage], xerr=group["std"],
                    fmt="o-", capsize=4, label=model, color=COLORS.get(model, "#666666"))
    ax.set_xlabel("F1 on the protocol-specific development partition")
    ax.set_title("Protocol ablation; primary holdout excluded")
    ax.legend(title="Model")
    save(fig, "f12_protocol_ablation")

    shap = pd.read_csv(RESULTS / "shap_importance.csv").sort_values("mean_absolute_shap_or_proxy", ascending=False).head(10)
    profile = pd.read_csv(RESULTS / "validation_error_profile.csv").set_index("group")
    fig, axes = plt.subplots(1, 2, figsize=(7, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    shap = shap.sort_values("mean_absolute_shap_or_proxy")
    axes[0].barh(shap.feature, shap.mean_absolute_shap_or_proxy, color="#4477AA")
    axes[0].set_xlabel("Mean absolute SHAP value")
    axes[0].set_title("Validation feature attribution")
    axes[1].bar(["False\nnegative", "True\npositive"],
                [profile.loc["false_negative", "count"], profile.loc["true_positive", "count"]],
                color=["#CC6677", "#4477AA"])
    axes[1].set_ylabel("Validation fraud cases")
    axes[1].set_title("Errors at selected threshold")
    save(fig, "f14_explainability_errors")

    calibration = pd.read_csv(RESULTS / "calibration_curve.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    calibration_colors = {"weights": "#4477AA", "smote": "#CC6677"}
    for model, group in calibration.groupby("model_strategy"):
        name = model.split("_")[1]
        ax.plot(group.mean_predicted, group.observed_fraction, marker="o",
                label=model.replace("_", " / "), color=calibration_colors.get(name, "#666666"))
    ax.plot([0, 1], [0, 1], "--", color="#666666")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraud fraction",
           title="Out-of-fold probability calibration")
    ax.legend()
    save(fig, "f11_reliability")

    cost = pd.read_csv(RESULTS / "cost_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(cost.review_cost_per_alert, cost.illustrative_net_saving, color="#4477AA", linewidth=2)
    ax.axhline(0, color="#666666", linestyle="--")
    ax.set(xlabel="Illustrative review cost per alert", ylabel="Flagged amount minus review cost",
           title="Illustrative cost sensitivity")
    save(fig, "f13_cost_sensitivity")

    print("Restyled F7-F9 and F11-F14 from saved results and saved test predictions; no predictive models fitted.")


if __name__ == "__main__":
    main()
