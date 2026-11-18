import pandas as pd, numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
R = "results/"; cv = pd.read_csv(R+"cv_summary.csv"); T = pd.read_csv(R+"cv_tests.csv"); sp = pd.read_csv(R+"split_paper_text.csv"); rp = pd.read_csv(R+"split_repo_code.csv")
OURS = "OURS (ensemble + engineered features)"
HF = PatternFill("solid", fgColor="14284B"); HFont = Font(bold=True, color="FFFFFF"); thin = Side(style="thin", color="D0D7E2"); BD = Border(left=thin, right=thin, top=thin, bottom=thin)
BLUE = PatternFill("solid", fgColor="E7F1FB"); YEL = PatternFill("solid", fgColor="FFF2CC")
wb = Workbook()
def sheet(name, title, sub=None):
    ws = wb.create_sheet(name); ws["A1"] = title; ws["A1"].font = Font(bold=True, size=14, color="14284B")
    if sub: ws["A2"] = sub; ws["A2"].font = Font(italic=True, color="5F6368")
    return ws
def table(ws, r0, df, fmt=None, widths=None):
    for j, c in enumerate(df.columns, 1):
        x = ws.cell(r0, j, c); x.fill = HF; x.font = HFont; x.alignment = Alignment(wrap_text=True, vertical="center"); x.border = BD
    for i, row in enumerate(df.itertuples(index=False), r0 + 1):
        for j, v in enumerate(row, 1):
            v = None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v)
            x = ws.cell(i, j, v); x.border = BD
            if isinstance(v, float): x.number_format = (fmt or {}).get(df.columns[j-1], "0.000")
    return r0 + len(df) + 1
def widths(ws, ws_w):
    for i, w in enumerate(ws_w, 1): ws.column_dimensions[get_column_letter(i)].width = w
S = {"mat": "Mathematics (395)", "por": "Portuguese (649)"}

# ---------------- README
ws = wb.active; ws.title = "README"; ws["A1"] = "Task 1 and 2: new dataset (UCI Student Performance) and a journal paper on the same data"; ws["A1"].font = Font(bold=True, size=14, color="14284B")
lines = [("Dataset", "UCI Machine Learning Repository, 'Student Performance' (dataset 320): https://archive.ics.uci.edu/dataset/320. Original source: Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. Proc. FUBUTEC 2008 (the dataset reference; the paper itself is a conference paper and is NOT used as the journal comparison)."),
 ("Data files", "student-mat.csv (395 students, Mathematics) and student-por.csv (649 students, Portuguese); 33 columns each. Files downloaded from two independent GitHub copies; both copies are byte-identical (md5 4dc304be... and 973e7bac...) and have the row/column counts stated by UCI."),
 ("Journal paper compared", "Kesgin, K., Kiraz, S., Kosunalp, S., & Stoycheva, B. (2025). Beyond Performance: Explaining and Ensuring Fairness in Student Academic Performance Prediction with Machine Learning. Applied Sciences, 15(15), 8409. DOI 10.3390/app15158409 (MDPI, peer-reviewed journal). Same dataset; logistic regression, random forest, XGBoost with SMOTE."),
 ("Target", "Binary (primary): pass = final grade G3 >= 10, fail otherwise (as in the journal paper). Secondary: the 5-level scale of Cortez & Silva (fail / sufficient / satisfactory / good / excellent)."),
 ("Two scenarios", "G1+G2 included (previous period grades, as in the journal paper) and G1+G2 excluded (harder, early-warning setting)."),
 ("Proposed method", "Same as the dropout project: soft-voting ensemble (LightGBM, CatBoost, XGBoost, ExtraTrees) with balanced class weights and engineered features (19 when G1/G2 are included, 13 otherwise). Hyper-parameters fixed in advance, not tuned on test folds."),
 ("Evaluation", "Repeated stratified 5-fold cross-validation, 3 repeats (15 folds); every step (feature selection, scaling, SMOTE, fitting) is done on the training fold only. Paired comparisons use the Nadeau-Bengio corrected resampled t-test. The single 80/20 split of the journal paper is also replicated on 10 seeds."),
 ("Compared pipelines", "Islam et al. (2025) pipeline (the baseline of the dropout project) and the three models of Kesgin et al. (2025) as described in their paper."),
 ("Sheets", "Headline | CV_binary | CV_5level | Tests | Paper_replication | Notes")]
for i, (a, b) in enumerate(lines, 3):
    ws.cell(i, 1, a).font = Font(bold=True); c = ws.cell(i, 2, b); c.alignment = Alignment(wrap_text=True, vertical="top"); ws.row_dimensions[i].height = 62
widths(ws, [24, 130])

# ---------------- Headline
ws = sheet("Headline", "Headline: binary pass/fail, repeated stratified 5-fold CV (15 folds)", "Accuracy / macro-F1; 'best published' = best of the three Kesgin et al. models in that setting. Diff cells are formulas.")
rows = []
for sub in ["mat", "por"]:
    for gr in ["G1+G2 included", "G1+G2 excluded"]:
        g = cv[(cv.subject == sub) & (cv.grades == gr) & (cv.target == "binary")].set_index("model")
        k = g.loc[[m for m in g.index if m.startswith("Kesgin")], "accuracy"]; best = k.idxmax()
        rows.append([S[sub], gr, g.loc[OURS, "accuracy"], g.loc[OURS, "accuracy_sd"], k[best], best.replace("Kesgin et al. (2025) ", ""), g.loc["Islam et al. (2025) pipeline", "accuracy"], g.loc[OURS, "macro_f1"],
                     g.loc[[m for m in g.index if m.startswith("Kesgin")], "macro_f1"].max(), g.loc[OURS, "f1_fail"], g.loc[best, "f1_fail"]])
H = pd.DataFrame(rows, columns=["Subject", "Scenario", "Ours: accuracy", "Ours: SD", "Best Kesgin: accuracy", "Best Kesgin model", "Islam pipeline: accuracy", "Ours: macro-F1", "Best Kesgin: macro-F1", "Ours: F1 (fail class)", "F1 fail of best-accuracy Kesgin model"])
end = table(ws, 4, H); ws.cell(4, 12, "Diff accuracy (ours - best Kesgin)").fill = HF; ws.cell(4, 12).font = HFont; ws.cell(4, 13, "Diff macro-F1 (ours - best Kesgin)").fill = HF; ws.cell(4, 13).font = HFont
for r in range(5, 9):
    ws.cell(r, 12, f"=C{r}-E{r}").number_format = "+0.000;-0.000"; ws.cell(r, 13, f"=H{r}-I{r}").number_format = "+0.000;-0.000"
ws.cell(10, 1, "Reading: with the same fair protocol, the proposed ensemble is on par with the best published model in accuracy (differences within 0.01 and not significant, see Tests) and is a little better in macro-F1 and in finding the minority (fail) class when G1/G2 are excluded.").alignment = Alignment(wrap_text=True)
ws.merge_cells("A10:M10"); ws.row_dimensions[10].height = 44; widths(ws, [20, 17, 12, 9, 14, 22, 14, 12, 14, 14, 18, 16, 16])

# ---------------- CV sheets
for nm, tg, ttl in [("CV_binary", "binary", "Binary pass/fail"), ("CV_5level", "five", "5-level grade")]:
    ws = sheet(nm, f"{ttl}: mean over 15 folds", "Higher is better. AUC and F1 (fail) only for the binary target.")
    d = cv[cv.target == tg].copy(); d["subject"] = d.subject.map(S)
    cols = ["subject", "grades", "model", "accuracy", "accuracy_sd", "balanced_acc", "macro_f1", "macro_f1_sd"] + (["f1_fail", "auc"] if tg == "binary" else []) + ["folds"]
    d = d[cols]; d.columns = ["Subject", "Scenario", "Model", "Accuracy", "Accuracy SD", "Balanced acc.", "Macro-F1", "Macro-F1 SD"] + (["F1 (fail)", "AUC"] if tg == "binary" else []) + ["Folds"]
    table(ws, 4, d, {"Folds": "0"}); widths(ws, [20, 17, 44, 10, 11, 12, 10, 11, 10, 8, 7])
    for r in range(5, 5 + len(d)):
        if ws.cell(r, 3).value == OURS:
            for c in range(1, ws.max_column + 1): ws.cell(r, c).fill = BLUE
    ws.freeze_panes = "D5"

# ---------------- Tests
ws = sheet("Tests", "Ours vs each baseline: paired differences over 15 folds, Nadeau-Bengio corrected t-test", "Positive diff = ours better. p < 0.05 highlighted. 'Wins' = folds (of 15) in which ours is higher.")
t = T[T.metric.isin(["accuracy", "macro_f1", "f1_fail"]) & (T.baseline != "OURS without engineered features")].copy(); t["subject"] = t.subject.map(S)
t = t[["subject", "grades", "target", "baseline", "metric", "ours", "baseline_mean", "mean_diff", "corrected_p", "wins"]]
t.columns = ["Subject", "Scenario", "Target", "Baseline", "Metric", "Ours", "Baseline mean", "Diff", "Corrected p", "Wins /15"]
table(ws, 4, t, {"Wins /15": "0", "Corrected p": "0.000", "Diff": "+0.000;-0.000"}); widths(ws, [20, 17, 9, 44, 10, 9, 13, 9, 12, 9])
for r in range(5, 5 + len(t)):
    if (ws.cell(r, 9).value or 1) < 0.05:
        for c in range(1, 11): ws.cell(r, c).fill = YEL
ws2 = sheet("Tests_ablation", "Ablation: ours with vs without engineered features (paired, 15 folds)")
a = T[(T.baseline == "OURS without engineered features")][["subject", "grades", "target", "metric", "ours", "baseline_mean", "mean_diff", "corrected_p", "wins"]].copy(); a["subject"] = a.subject.map(S)
a.columns = ["Subject", "Scenario", "Target", "Metric", "With eng. features", "Without", "Diff", "Corrected p", "Wins /15"]
table(ws2, 3, a, {"Wins /15": "0", "Diff": "+0.000;-0.000"}); widths(ws2, [20, 17, 9, 10, 18, 10, 9, 12, 9])

# ---------------- Paper replication
ws = sheet("Paper_replication", "Replication of Kesgin et al. (2025): reported vs reproduced", "Mathematics file, G1+G2 included, pass/fail, stratified 80/20 split (79 test students), SMOTE on the training part, paper settings. Reproduced on 10 different seeds.")
rep = sp.groupby("model").agg(mean_acc=("accuracy", "mean"), sd_acc=("accuracy", "std"), min_acc=("accuracy", "min"), max_acc=("accuracy", "max"), mean_f1=("f1_fail", "mean"), mean_macro=("macro_f1", "mean")).reset_index()
rep.columns = ["Model (10 splits)", "Mean accuracy", "SD", "Min", "Max", "Mean F1 (fail)", "Mean macro-F1"]
e = table(ws, 4, rep)
ws.cell(e + 1, 1, "What the paper reports").font = Font(bold=True)
rr = [["Logistic regression: accuracy 0.962, AUC 0.987", "Sections 5.2 and Conclusions"], ["XGBoost: accuracy 0.789, F1 0.803", "Abstract (the text also calls logistic regression the best model; the two statements conflict)"],
      ["Sample size", "Methods say 1,044 merged students; Results use 395 (train 316 / test 79)"], ["Differences between models", "Reported as not statistically significant"]]
for i, (a1, b1) in enumerate(rr, e + 2): ws.cell(i, 1, a1); ws.cell(i, 2, b1)
i = e + 2 + len(rr) + 1
ws.cell(i, 1, "Reproduced: share of the 10 splits with logistic-regression accuracy >= 0.962").font = Font(bold=True); ws.cell(i, 2, int((sp[sp.model == "Kesgin LR"].accuracy >= 0.962).sum())); ws.cell(i, 3, "of 10 (maximum observed 0.949)")
ws.cell(i + 2, 1, "Repository-code variant (kdrksgn/fairsynedu): Portuguese file, 10 predictors, plain random split seed 42, grid-searched RF and XGBoost").font = Font(bold=True)
r2 = rp[["model", "n_test", "accuracy", "macro_f1", "f1_fail", "auc"]].copy(); r2.columns = ["Model", "Test n", "Accuracy", "Macro-F1", "F1 (fail)", "AUC"]
table(ws, i + 3, r2, {"Test n": "0"})
ws.cell(i + 10, 1, "Note: the authors' repository notebook does not match the paper text (different file, 10 predictors, no stratification), so the paper's numbers could not be traced to the code.").alignment = Alignment(wrap_text=True); ws.merge_cells(start_row=i + 10, start_column=1, end_row=i + 10, end_column=7); ws.row_dimensions[i + 10].height = 32
sp.round(4).to_csv("results/_tmp.csv"); widths(ws, [58, 70, 14, 10, 10, 14, 14])
ws3 = sheet("Split_seeds", "Per-seed results, paper-text protocol (79 test students)"); sp2 = sp.copy(); sp2.columns = [c.replace("_", " ") for c in sp2.columns]
table(ws3, 3, sp2.round(4), {"seed": "0", "n test": "0"})

# ---------------- Notes
ws = sheet("Notes", "Caveats and how to read this")
notes = ["Different task: Student Performance predicts a pass/fail grade of secondary-school students, not university dropout. It is used to test whether the method transfers to another education dataset with a reputable source.",
 "Grades G1 and G2 dominate the prediction. With them, every model reaches about 0.90 to 0.93 accuracy and the differences between methods are tiny; without them accuracy drops to about 0.67 (Mathematics) and 0.84 (Portuguese).",
 "The proposed method is NOT significantly better than the best published model on accuracy in any setting; it is better than the logistic regression of the journal paper in some settings (see Tests) and better than the Islam et al. pipeline when G1/G2 are excluded.",
 "Engineered features give no significant gain in any setting (see Tests_ablation); the largest difference, Portuguese without G1/G2 (+0.009 accuracy, +0.013 macro-F1), is within noise.",
 "Small test sets: a single 80/20 split has 79 (Mathematics) or 130 (Portuguese) test students, so one split can swing accuracy by several points; the journal paper's 0.962 was not reproduced on any of 10 splits.",
 "Hyper-parameters of the proposed method were fixed before running and not tuned; the repeated CV and Nadeau-Bengio test are used so that conclusions do not depend on one split.",
 "The journal paper has internal inconsistencies (see Paper_replication); we reproduced the method as described in its text and, separately, as coded in its repository."]
for i, n in enumerate(notes, 3):
    c = ws.cell(i, 1, f"{i-2}. {n}"); c.alignment = Alignment(wrap_text=True, vertical="top"); ws.row_dimensions[i].height = 48
widths(ws, [150])
wb.save("/home/claude/outputs/student_performance_new_dataset_results.xlsx")
