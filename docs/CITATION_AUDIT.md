# Citation audit

The supplied report digest says its reference list was not visible in the extracted pages. The PDF text extraction utility was unavailable during setup, so references beyond the two numbered placeholders below cannot be identified reliably. No unverified citation has been carried into the new report.

| Old reference | Status | Reason |
|---|---|---|
| [1] (confusion-matrix definitions; identity unavailable) | REPLACE | Replace generic definitions with explicit metric formulas and cite the verified PR-AUC/MCC methodological sources only where relevant. The source cannot be bibliographically verified from the digest. |
| [3] (accuracy definition; identity unavailable) | DROP | Accuracy is defined directly in the report and used only as an uninformative contextual baseline; the unidentified source adds no study-specific evidence. |

The report uses the references listed in PROJECT_BRIEF.md Section 9 plus three additional sources requested by the author. The CatBoost citation supports ordered boosting, not categorical-feature advantages for this numeric dataset. The added SMOTE sources provide method context and limitations; they are not cited as evidence of improved PR-AUC or boosting performance.

| Brief reference | Status | Use in report |
|---|---|---|
| R10 (`lundberg2017`) | KEEP | Cited where SHAP is introduced and interpreted in Section VII-E. Bibliographic fields are copied exactly from the verified brief. |
| R11 (`kapoor2022`) | KEEP | Cited in the Introduction and Related Work for leakage and reproducibility context. Bibliographic fields are copied exactly from the verified brief. |

| Author-added reference | Status | Use in report |
|---|---|---|
| Blagus and Lusa (2013), `blagus2013smote` | KEEP | Cited for the caution that SMOTE behavior in high-dimensional data can differ; DOI and publication details checked against the publisher record. |
| Fernández et al. (2018), `fernandez2018smote` | KEEP | Cited as a review of SMOTE progress and challenges; bibliographic details and DOI checked against the journal record. |
| Lemaître et al. (2017), `JMLR:v18:16-365` | KEEP | Cited for the imbalanced-learn toolkit used by the resampling pipelines; checked against the JMLR article record. |

The old report's own reference list is still unrecovered because its PDF text extraction was unavailable during setup. Paste that list if you want the old-reference audit completed. The three new citations extend the original R1--R11 verified-reference set at the author's request.
