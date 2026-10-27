"""Leakage-free descriptive statistics and plots for the chronological train split."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.data import FEATURE_COLUMNS, chronological_split, load_transactions
from fraud.config import DATA_PATH, SEED

sns.set_theme(style="whitegrid", palette="colorblind")
OUT = Path("figures"); OUT.mkdir(exist_ok=True)
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p); ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    output = np.empty_like(adjusted); output[order] = np.minimum(adjusted, 1.0)
    return output


def main():
    raw = load_transactions(DATA_PATH)
    splits = chronological_split(raw)
    train = splits.train
    # F1 is descriptive for the deduplicated full dataset; all inference and ranking use train only.
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = raw.Class.value_counts().sort_index()
    bars = ax.bar(["Legitimate", "Fraud"], counts.values, color=["#4477AA", "#CC6677"])
    ax.set_yscale("log"); ax.set_ylabel("Transactions (log scale)")
    ax.set_title("Fraud is rare, so accuracy alone is uninformative")
    for bar, value in zip(bars, counts.values): ax.text(bar.get_x()+bar.get_width()/2, value, f"{value:,}", ha="center", va="bottom")
    save(fig, "f01_class_imbalance")

    hourly = train.assign(hour=((train.Time // 3600) % 24).astype(int)).groupby("hour").Class.agg(["size", "sum"])
    hourly = hourly.reindex(range(24), fill_value=0); hourly["rate"] = hourly["sum"] / hourly["size"].replace(0, np.nan)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].bar(hourly.index, hourly["size"], color="#4477AA"); axes[0].set_ylabel("Transactions")
    axes[1].plot(hourly.index, hourly.rate, marker="o", color="#CC6677"); axes[1].set_ylabel("Fraud rate"); axes[1].set_xlabel("Hour from Time (seconds since first transaction)")
    fig.suptitle("Transaction volume and fraud rate vary across the train window")
    save(fig, "f02_hourly_patterns")

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, color, name in [(0, "#4477AA", "Legitimate"), (1, "#CC6677", "Fraud")]:
        vals = np.sort(np.log1p(train.loc[train.Class.eq(label), "Amount"].to_numpy()))
        ax.plot(vals, np.arange(1, len(vals)+1)/len(vals), color=color, label=name)
    ax.set_xlabel("log1p(Amount)"); ax.set_ylabel("Empirical cumulative probability"); ax.legend()
    ax.set_title("Transaction amount distributions overlap, motivating multivariate features")
    save(fig, "f03_amount_ecdf")

    X = train[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = train.Class.to_numpy()
    rows=[]
    for feature in FEATURE_COLUMNS:
        negative = X.loc[y == 0, feature].to_numpy(); positive = X.loc[y == 1, feature].to_numpy()
        u = mannwhitneyu(positive, negative, alternative="two-sided", method="asymptotic")
        auc = roc_auc_score(y, X[feature])
        rows.append({"feature": feature, "mann_whitney_p": float(u.pvalue), "cliffs_delta": float(2*auc-1)})
    stats = pd.DataFrame(rows); stats["mann_whitney_q_bh"] = bh_adjust(stats.mann_whitney_p)
    mi = mutual_info_classif(X, y, random_state=SEED)
    stats["mutual_information"] = mi
    stats.to_csv(RESULTS / "eda_feature_ranking.csv", index=False)
    top = stats.assign(abs_delta=stats.cliffs_delta.abs()).nlargest(15, "abs_delta").sort_values("cliffs_delta")
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    axes[0].barh(top.feature, top.cliffs_delta, color="#4477AA"); axes[0].set_title("Cliff's delta"); axes[0].set_xlabel("Effect size")
    mi_top = stats.nlargest(15, "mutual_information").sort_values("mutual_information")
    axes[1].barh(mi_top.feature, mi_top.mutual_information, color="#CC6677"); axes[1].set_title("Mutual information"); axes[1].set_xlabel("Nats")
    fig.suptitle("Train-only univariate screens are descriptive, not feature importance")
    save(fig, "f04_feature_screens")

    selected = stats.assign(abs_delta=stats.cliffs_delta.abs()).nlargest(4, "abs_delta").feature.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, feature in zip(axes.flat, selected):
        for label, color, name in [(0, "#4477AA", "Legitimate"), (1, "#CC6677", "Fraud")]:
            values = X.loc[y == label, feature].to_numpy()
            ax.hist(values, bins=35, density=True, histtype="step", linewidth=1.5, color=color, label=name)
        ax.set_title(feature); ax.set_ylabel("Density")
    axes.flat[0].legend()
    fig.suptitle("Top marginal features show class separation, but hide interactions")
    save(fig, "f05_top_feature_densities")

    findings = [
      "F1 — Finding: the class distribution is highly imbalanced. Decision: report PR-AUC and alert precision/recall; accuracy is contextual only.",
      "F2 — Finding: transaction volume and fraud rate change over the train window. Decision: preserve chronological order and include cyclical hour features.",
      "F3 — Finding: amount distributions overlap. Decision: retain log1p(Amount) as one input among the anonymized V features.",
      "F4 — Finding: marginal effect sizes and mutual information rank features differently. Decision: treat these screens as descriptive and verify with model-based measures.",
      "F5 — Finding: some components separate classes marginally, but their densities overlap. Decision: compare multivariate models without claiming causal or business meaning.",
      "Inference notes: Shapiro-Wilk is uninformative at this sample size; t-test and ANOVA are redundant for two groups; Spearman with a binary target duplicates the rank-based group comparison.",
    ]
    (RESULTS / "eda_findings.md").write_text("# EDA findings and decisions\n\n" + "\n\n".join(findings) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(11, 3.5)); ax.axis("off")
    nodes=[("Deduplicate + sort",0.08),("Train: fit / CV",0.32),("Validation: threshold",0.57),("Test: one evaluation",0.82)]
    for label,x in nodes:
        ax.text(x,0.55,label,ha="center",va="center",bbox=dict(boxstyle="round,pad=0.6",fc="#DDEBF7",ec="#4477AA"),transform=ax.transAxes)
    for (_,x1),(_,x2) in zip(nodes,nodes[1:]):
        ax.annotate("",xy=(x2-0.08,0.55),xytext=(x1+0.08,0.55),xycoords=ax.transAxes,textcoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=1.5,color="#555555"))
    ax.text(0.32,0.18,"Scaler and resampler fit inside each training fold",ha="center",transform=ax.transAxes,color="#333333")
    ax.set_title("Leakage control: fit choices on earlier partitions before the single holdout evaluation")
    save(fig,"f06_leakage_protocol")

    print(json.dumps({"train_rows":len(train),"frauds":int(y.sum()),"top_features":selected}, indent=2))

if __name__ == "__main__": main()
