"""Package figures and generate report-only numbers/tables from results files."""
from __future__ import annotations
import json, shutil, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; REPORT=ROOT/"report"; RESULTS=ROOT/"results"

def esc(value): return str(value).replace("&",r"\&").replace("%",r"\%").replace("_",r"\_")

def main():
    figures=REPORT/"figures"; generated=REPORT/"generated"; tables=generated/"tables"
    figures.mkdir(parents=True,exist_ok=True); tables.mkdir(parents=True,exist_ok=True)
    for pdf in (ROOT/"figures").glob("*.pdf"): shutil.copy2(pdf,figures/pdf.name)
    audit_path=RESULTS/"data_audit.json"; audit=json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    final_path=RESULTS/"final_metrics.json"; final=json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else {}
    cv_path=RESULTS/"cv_summary.csv"
    cv=__import__("pandas").read_csv(cv_path) if cv_path.exists() else None
    commands={"DataRows":audit.get("rows_after_deduplication","pending"),"FraudCount":audit.get("frauds_after_deduplication","pending"),"BestModel":"pending","BestPRAUC":"pending","BestCVModel":"pending","BestCVPRAUC":"pending","PairedDiffLow":"pending","PairedDiffHigh":"pending","RFAblationDrop":"pending","XGBAblationDrop":"pending","TestRows":"pending"}
    if cv is not None and len(cv):
        best_cv=cv.sort_values("mean",ascending=False).iloc[0]
        commands["BestCVModel"]=best_cv.model+" / "+best_cv.strategy
        commands["BestCVPRAUC"]=f'{best_cv["mean"]:.4f}'
    if final.get("models"):
        best=final["models"][0]; commands["BestModel"]=best["model"]+" / "+best["strategy"]; commands["BestPRAUC"]=f'{best["pr_auc"]:.4f}'
        commands["BestROCAUC"]=f'{best["roc_auc"]:.4f}'; commands["BestPrecision"]=f'{best["precision"]:.4f}'; commands["BestRecall"]=f'{best["recall"]:.4f}'
        commands["BestFOne"]=f'{best["f1"]:.4f}'; commands["TestRows"]=final["test_rows"]
        ci=best["ci95_stratified_bootstrap"]["pr_auc"]; commands["BestPRAUCLow"]=f"{ci[0]:.4f}"; commands["BestPRAUCHigh"]=f"{ci[1]:.4f}"
        delta=final.get("paired_pr_auc_difference_top1_minus_top2",{}).get("ci95",["pending","pending"])
        commands["PairedDiffLow"]=f"{delta[0]:.4f}" if isinstance(delta[0],(float,int)) else delta[0]
        commands["PairedDiffHigh"]=f"{delta[1]:.4f}" if isinstance(delta[1],(float,int)) else delta[1]
    (generated/"numbers.tex").write_text("% Generated from results/*.json; do not edit by hand.\n"+"\n".join(f"\\newcommand{{\\{key}}}{{{esc(value)}}}" for key,value in commands.items())+"\n",encoding="utf-8")
    if final.get("models"):
        lines=[r"\begin{tabular}{lrrrr}",r"\toprule Model & PR-AUC & ROC-AUC & Precision & Recall \\",r"\midrule"]
        for item in final["models"]: lines.append(f'{esc(item["model"] + " / " + item["strategy"])} & {item["pr_auc"]:.4f} & {item["roc_auc"]:.4f} & {item["precision"]:.4f} & {item["recall"]:.4f} \\\\')
        lines += [r"\bottomrule",r"\end{tabular}"]
    else: lines=[r"\begin{tabular}{l}",r"Full final metrics are pending the prescribed CV and single test evaluation. \\",r"\end{tabular}"]
    (tables/"final_metrics.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    ablation_path=RESULTS/"ablation.csv"
    if ablation_path.exists():
        import pandas as pd
        ab=pd.read_csv(ablation_path); stats=ab.groupby(["model","stage"]).agg(f1_mean=("f1","mean"),f1_sd=("f1","std"),pr_mean=("pr_auc","mean"),pr_sd=("pr_auc","std"),recall_mean=("recall","mean")).reset_index()
        for model in ("rf","xgb"):
            means=stats[stats.model.eq(model)].set_index("stage").f1_mean
            if "A0_legacy_like" in means and "A3_chronological" in means: commands[model.upper()+"AblationDrop"]=f'{means["A0_legacy_like"]-means["A3_chronological"]:.3f}'
        rows=[r"\begin{tabular}{llrrrr}",r"\toprule Model & Protocol & F1 mean & F1 SD & PR-AUC mean & Recall mean \\",r"\midrule"]
        labels={"A0_legacy_like":"A0 legacy-like","A1_fold_local_scale":"A1 fold-local","A2_validation_threshold":"A2 validation threshold","A3_chronological":"A3 chronological"}
        for row in stats.itertuples(index=False): rows.append(f'{esc(row.model)} & {esc(labels.get(row.stage,row.stage))} & {row.f1_mean:.3f} & {row.f1_sd:.3f} & {row.pr_mean:.3f} & {row.recall_mean:.3f} \\\\')
        rows += [r"\bottomrule",r"\end{tabular}"]
        (tables/"ablation_metrics.tex").write_text("\n".join(rows)+"\n",encoding="utf-8")
        (generated/"numbers.tex").write_text("% Generated from results/*.json and *.csv; do not edit by hand.\n"+"\n".join(f"\\newcommand{{\\{key}}}{{{esc(value)}}}" for key,value in commands.items())+"\n",encoding="utf-8")
    tuning_path=RESULTS/"tuning_results.json"
    if tuning_path.exists():
        tuning=json.loads(tuning_path.read_text(encoding="utf-8")); rows=[r"\begin{tabular}{lrrl}",r"\toprule Model & Tuned CV AP & Validation AP & Parameters \\",r"\midrule"]
        for item in tuning:
            params=", ".join(f"{k.split('__')[-1]}={v}" for k,v in item["best_params"].items())
            rows.append(f'{esc(item["model"]+" / "+item["strategy"])} & {item["best_cv_pr_auc"]:.4f} & {item["validation_pr_auc"]:.4f} & {esc(params)} \\\\')
        rows += [r"\bottomrule",r"\end{tabular}"]
        (tables/"tuning_results.tex").write_text("\n".join(rows)+"\n",encoding="utf-8")
    target=REPORT/"overleaf_upload.zip"
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as archive:
        for p in REPORT.rglob("*"):
            if p.is_file() and p!=target: archive.write(p,p.relative_to(REPORT))
    print(f"Generated {generated/'numbers.tex'}, {tables/'final_metrics.tex'}, and {target}")

if __name__=="__main__": main()
