"""Sequential, resumable stratified CV; writes one result bundle per model/strategy."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.config import DATA_PATH, SEED
from fraud.data import FEATURE_COLUMNS, chronological_split, load_transactions
from fraud.models import MODELS, STRATEGIES, make_pipeline

OUT = Path("results/cv")


def run_pair(frame, model, strategy, fast=False):
    X = frame[FEATURE_COLUMNS].to_numpy(); y = frame.Class.to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows=[]; oof=np.full(len(y), np.nan); fold_times=[]
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), 1):
        pipeline=make_pipeline(model, strategy, fast=fast)
        if model == "xgb" and strategy == "weights":
            positives = int(y[train_idx].sum()); negatives = len(train_idx) - positives
            pipeline.set_params(model__scale_pos_weight=negatives / positives)
        start=time.perf_counter(); pipeline.fit(X[train_idx], y[train_idx]); elapsed=time.perf_counter()-start
        proba=pipeline.predict_proba(X[valid_idx])[:, 1]; oof[valid_idx]=proba
        pred=(proba >= 0.5).astype(int)
        rows.append({"model":model,"strategy":strategy,"fold":fold,
            "pr_auc":average_precision_score(y[valid_idx],proba),
            "roc_auc":roc_auc_score(y[valid_idx],proba),
            "mcc_at_0_5":matthews_corrcoef(y[valid_idx],pred),"fit_seconds":elapsed})
        fold_times.append(elapsed)
    return pd.DataFrame(rows), oof, float(np.mean(fold_times))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["lr","rf","knn","xgb","catboost"])
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=["weights","smote","undersample","none"])
    parser.add_argument("--fast", action="store_true", help="development sample and small estimators")
    args=parser.parse_args(); OUT = Path("results/cv_fast" if args.fast else "results/cv"); OUT.mkdir(exist_ok=True, parents=True)
    frame=load_transactions(DATA_PATH); train=chronological_split(frame).train
    if args.fast:
        # Keep all positives and a deterministic capped majority sample for a bounded smoke run.
        positives=train[train.Class.eq(1)]; negatives=train[train.Class.eq(0)].sample(n=min(2500, (train.Class.eq(0)).sum()), random_state=SEED)
        train=pd.concat([positives, negatives]).sample(frac=1, random_state=SEED)
    all_rows=[]
    for model in args.models:
        for strategy in args.strategies:
            if model == "knn" and strategy not in {"none", "undersample"}: continue
            if model != "knn" and strategy == "none": continue
            target=OUT / f"{model}_{strategy}.csv"
            if target.exists():
                all_rows.append(pd.read_csv(target)); continue
            print(f"Running {model}/{strategy} on {len(train):,} rows", flush=True)
            try:
                rows, oof, avg_seconds=run_pair(train, model, strategy, fast=args.fast)
            except ImportError as exc:
                print(f"SKIP {model}/{strategy}: missing optional dependency: {exc}", flush=True); continue
            rows.to_csv(target,index=False)
            np.savez_compressed(OUT / f"{model}_{strategy}_oof.npz", y=train.Class.to_numpy(), probability=oof)
            print(f"Finished {model}/{strategy}; mean fold fit {avg_seconds:.1f}s", flush=True)
            all_rows.append(rows)
    if all_rows:
        cv=pd.concat([pd.read_csv(path) for path in OUT.glob("*.csv") if not path.name.endswith("_oof.csv")],ignore_index=True)
        summary=cv.groupby(["model","strategy"]).agg(mean=("pr_auc","mean"),std=("pr_auc","std"),mean_fit_seconds=("fit_seconds","mean")).reset_index()
        if not args.fast:
            cv.to_csv("results/cv_results.csv",index=False)
            summary.to_csv("results/cv_summary.csv",index=False)
        print(summary.to_string(index=False))
        if not args.fast:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            summary=summary.sort_values("mean")
            fig,ax=plt.subplots(figsize=(9,6))
            for strategy, group in summary.groupby("strategy"):
                ax.errorbar(group["mean"], group["model"]+" / "+strategy, xerr=group["std"], fmt="o", capsize=3, label=strategy)
            ax.set_xlabel("Cross-validation average precision")
            ax.set_title("Fold-mean PR-AUC supports train-only model selection")
            ax.legend(title="Imbalance strategy"); fig.tight_layout()
            Path("figures").mkdir(exist_ok=True)
            fig.savefig("figures/f07_cv_pr_auc.png",dpi=300); fig.savefig("figures/f07_cv_pr_auc.pdf"); plt.close(fig)
        if args.fast:
            completed=float(cv.fit_seconds.sum()); total_pairs=len(args.models)*len(args.strategies)-int("knn" in args.models and "smote" in args.strategies)
            full_estimate=completed * 170235 / max(len(train),1) * total_pairs / max(len(cv.model.unique()) * len(cv.strategy.unique()),1)
            print(f"FAST runtime {completed:.1f}s; rough full-run estimate across requested pairs {full_estimate/60:.1f} minutes; variability can be substantial")

if __name__ == "__main__": main()
