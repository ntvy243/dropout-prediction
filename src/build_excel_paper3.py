import pandas as pd, numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from common import nadeau_bengio
from common import RESULTS, paper3_all_path
d = pd.read_csv(paper3_all_path()); sel = pd.read_csv(RESULTS / "paper3_selection_oof.csv")
h = d[d.kind == "holdout"]; c = d[d.kind == "cv"]; O2, O1 = "OURS v2 (sqrt weights)", "OURS v1 (balanced weights)"
HF = PatternFill("solid", fgColor="14284B"); HFont = Font(bold=True, color="FFFFFF"); thin = Side(style="thin", color="D0D7E2"); BD = Border(left=thin, right=thin, top=thin, bottom=thin)
BLUE = PatternFill("solid", fgColor="E7F1FB"); YEL = PatternFill("solid", fgColor="FFF2CC")
wb = Workbook()
def sheet(name, title, sub=None):
    ws = wb.create_sheet(name); ws["A1"] = title; ws["A1"].font = Font(bold=True, size=14, color="14284B")
    if sub: ws["A2"] = sub; ws["A2"].font = Font(italic=True, color="5F6368")
    return ws
def table(ws, r0, df, fmt=None):
    for j, col in enumerate(df.columns, 1):
        x = ws.cell(r0, j, col); x.fill = HF; x.font = HFont; x.alignment = Alignment(wrap_text=True, vertical="center"); x.border = BD
    for i, row in enumerate(df.itertuples(index=False), r0 + 1):
        for j, v in enumerate(row, 1):
            v = None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v); x = ws.cell(i, j, v); x.border = BD
            if isinstance(v, float): x.number_format = (fmt or {}).get(df.columns[j-1], "0.0000")
    return r0 + len(df) + 1
def widths(ws, w):
    for i, x in enumerate(w, 1): ws.column_dimensions[get_column_letter(i)].width = x
ws = wb.active; ws.title = "README"; ws["A1"] = "Task 2: journal paper on the SAME dataset (UCI Predict Students' Dropout and Academic Success), replicated and compared"; ws["A1"].font = Font(bold=True, size=14, color="14284B")
L = [("Journal paper", "Praptomo, S., Risman, A., & Suri, R. M. (2026). Explainable Machine Learning for Predicting Student Dropout and Academic Success Using XGBoost and SHAP. Journal of Electrical Engineering and Computer (JEECOM), 8(1). DOI 10.33650/jeecom.v8i1.16978. A journal article (not conference)."),
 ("Same dataset", "UCI dataset 697 (Realinho et al., 2022), 4,424 students, 36 predictors, 3 classes (Graduate 2,209 / Dropout 1,421 / Enrolled 794)."),
 ("Their protocol", "Stratified 80/20 hold-out, random_state 42 (885 test students: 284 Dropout / 159 Enrolled / 442 Graduate); no resampling; XGBoost tuned by RandomizedSearchCV on the training part (best: 500 trees, depth 5, lr 0.1, min_child_weight 3, subsample 1.0, colsample 0.9); Random Forest 400 trees balanced weights; logistic regression with standard scaling. This is the same split as our fair protocol (Protocol C)."),
 ("What we did", "(1) Re-ran their three models on the identical split and checked the numbers. (2) Compared our method (v1, and a changed version v2) with them on that split, on 10 more stratified 80/20 splits and on repeated stratified 5-fold CV (3 repeats = 15 folds) with the Nadeau-Bengio corrected t-test."),
 ("Why the method was changed", "Version v1 (balanced class weights) had LOWER accuracy than the paper's models on their split (0.751 vs 0.761 to 0.772). Balanced weights trade accuracy for recall of the small Enrolled class. Version v2 uses square-root class weights, chosen ONLY from out-of-fold results on the training part (accuracy 0.7875 vs 0.7771, macro-F1 equal); the 885 test students were not used for the choice."),
 ("Sheets", "Replication | Holdout_seed42 | Holdout_10seeds | CV_15folds | Tests | Method_change | Notes")]
for i, (a, b) in enumerate(L, 3):
    ws.cell(i, 1, a).font = Font(bold=True); x = ws.cell(i, 2, b); x.alignment = Alignment(wrap_text=True, vertical="top"); ws.row_dimensions[i].height = 70
widths(ws, [24, 130])
# replication
ws = sheet("Replication", "Do we reproduce the numbers of the journal paper? (identical split, 885 test students)", "Reported = Tables 3 and 4 of the paper (as quoted from the article page). Reproduced = our run with the stated settings.")
rep = h[h.split == 42].set_index("model"); rows = []
for nm, rk, rp in [("Paper XGB (JEECOM)", "XGBoost", (0.7605, 0.7008, 0.4654)), ("Paper RF (JEECOM)", "Random Forest", (0.7718, 0.6979, 0.3836)), ("Paper LR (JEECOM)", "Logistic Regression", (0.7684, 0.6826, 0.3333))]:
    rows.append([rk, rp[0], rep.loc[nm, "accuracy"], rp[1], rep.loc[nm, "macro_f1"], rp[2], rep.loc[nm, "recall_Enrolled"]])
R1 = pd.DataFrame(rows, columns=["Model", "Accuracy reported", "Accuracy reproduced", "Macro-F1 reported", "Macro-F1 reproduced", "Enrolled recall reported", "Enrolled recall reproduced"]); table(ws, 4, R1)
for r in range(5, 8): ws.cell(r, 8, f"=C{r}-B{r}").number_format = "+0.0000;-0.0000"
ws.cell(4, 8, "Accuracy diff").fill = HF; ws.cell(4, 8).font = HFont
ws.cell(9, 1, "XGBoost and logistic regression are reproduced to four decimals (accuracy, macro-F1 and Enrolled recall). The Random Forest differs (0.7593 vs 0.7718): the paper does not report all its settings, and its Enrolled recall (0.384) suggests different class-weight handling than ours.").alignment = Alignment(wrap_text=True)
ws.merge_cells("A9:H9"); ws.row_dimensions[9].height = 46; widths(ws, [22, 16, 18, 16, 18, 20, 22, 14])
# holdout seed 42
ws = sheet("Holdout_seed42", "Paper's own split (seed 42, 885 test students)", "Our two versions and the three models of the paper, same split.")
a = h[h.split == 42][["model", "accuracy", "macro_f1", "f1_Dropout", "f1_Enrolled", "f1_Graduate", "recall_Enrolled"]].copy(); a.columns = ["Model", "Accuracy", "Macro-F1", "F1 Dropout", "F1 Enrolled", "F1 Graduate", "Enrolled recall"]
table(ws, 4, a); widths(ws, [32, 11, 11, 11, 12, 12, 15])
for r in range(5, 5 + len(a)):
    if str(ws.cell(r, 1).value).startswith("OURS"):
        for k in range(1, 8): ws.cell(r, k).fill = BLUE
# holdout 10 seeds
ws = sheet("Holdout_10seeds", "Ten other stratified 80/20 splits (seeds 0 to 9): mean and SD", "Shows how much a single split can move the numbers.")
g = h[h.split != 42].groupby("model")[["accuracy", "macro_f1", "f1_Enrolled", "recall_Enrolled"]].agg(["mean", "std"]); g.columns = [f"{a} {b}" for a, b in g.columns]; g = g.reset_index()
g.columns = ["Model", "Accuracy mean", "Accuracy SD", "Macro-F1 mean", "Macro-F1 SD", "F1 Enrolled mean", "F1 Enrolled SD", "Enrolled recall mean", "Enrolled recall SD"]
table(ws, 4, g); widths(ws, [32, 14, 12, 14, 12, 16, 14, 20, 18])
for r in range(5, 5 + len(g)):
    if str(ws.cell(r, 1).value).startswith("OURS"):
        for k in range(1, 10): ws.cell(r, k).fill = BLUE
# CV
ws = sheet("CV_15folds", "Repeated stratified 5-fold CV (3 repeats): mean and SD over 15 folds", "Everything fitted on the training fold only.")
g = c.groupby("model")[["accuracy", "macro_f1", "f1_Enrolled", "recall_Enrolled"]].agg(["mean", "std"]); g.columns = [f"{a} {b}" for a, b in g.columns]; g = g.reset_index()
g.columns = ["Model", "Accuracy mean", "Accuracy SD", "Macro-F1 mean", "Macro-F1 SD", "F1 Enrolled mean", "F1 Enrolled SD", "Enrolled recall mean", "Enrolled recall SD"]
table(ws, 4, g); widths(ws, [32, 14, 12, 14, 12, 16, 14, 20, 18])
for r in range(5, 5 + len(g)):
    if str(ws.cell(r, 1).value).startswith("OURS"):
        for k in range(1, 10): ws.cell(r, k).fill = BLUE
# tests
ws = sheet("Tests", "Paired differences over 15 folds, Nadeau-Bengio corrected t-test", "Positive = our version is higher. p < 0.05 highlighted.")
tt = []
for ours in [O2, O1]:
    for base in [b for b in c.model.unique() if b.startswith("Paper")]:
        for m in ["accuracy", "macro_f1", "f1_Enrolled", "recall_Enrolled"]:
            x = c[c.model == ours].sort_values("split")[m].values; y2 = c[c.model == base].sort_values("split")[m].values; df = x - y2; t, p = nadeau_bengio(df, c.n_train.iloc[0], c.n_test.iloc[0])
            tt.append([ours, base, m, x.mean(), y2.mean(), df.mean(), p, int((df > 0).sum())])
for m in ["accuracy", "macro_f1", "f1_Enrolled", "recall_Enrolled"]:
    x = c[c.model == O2].sort_values("split")[m].values; y2 = c[c.model == O1].sort_values("split")[m].values; df = x - y2; t, p = nadeau_bengio(df, c.n_train.iloc[0], c.n_test.iloc[0]); tt.append([O2, O1, m, x.mean(), y2.mean(), df.mean(), p, int((df > 0).sum())])
TT = pd.DataFrame(tt, columns=["Ours", "Compared with", "Metric", "Ours mean", "Other mean", "Diff", "Corrected p", "Wins /15"]); table(ws, 4, TT, {"Wins /15": "0", "Diff": "+0.0000;-0.0000", "Corrected p": "0.0000"}); widths(ws, [28, 30, 16, 11, 11, 10, 12, 9])
for r in range(5, 5 + len(TT)):
    if (ws.cell(r, 7).value or 1) < 0.05:
        for k in range(1, 9): ws.cell(r, k).fill = YEL
# method change
ws = sheet("Method_change", "Why and how the method was changed (v1 to v2)", "Selection used only out-of-fold predictions on the training part of the paper's split.")
s2 = sel.copy(); s2["config"] = s2.config.map({"v1": "v1: balanced class weights", "v2": "v2: square-root class weights"}); s2.columns = ["Version", "OOF accuracy", "OOF macro-F1", "OOF F1 Enrolled"]; e = table(ws, 4, s2)
ws.cell(e + 1, 1, "Same four members in both versions (LightGBM, CatBoost, XGBoost, ExtraTrees) and the same 18 engineered features; only the class weights change (balanced = n/(K n_c), square-root = its square root). Fixed hand-set hyper-parameters; no tuning on any test data.").alignment = Alignment(wrap_text=True)
ws.merge_cells(start_row=e + 1, start_column=1, end_row=e + 1, end_column=6); ws.row_dimensions[e + 1].height = 46
ws.cell(e + 3, 1, "Trade-off: v2 has higher accuracy; v1 finds more Enrolled students (recall 0.60 vs 0.51 in CV). Both are reported; use v1 when catching Enrolled students matters most.").alignment = Alignment(wrap_text=True)
ws.merge_cells(start_row=e + 3, start_column=1, end_row=e + 3, end_column=6); ws.row_dimensions[e + 3].height = 34; widths(ws, [34, 16, 16, 18, 12, 12])
ws = sheet("Notes", "Caveats")
N = ["The journal paper reports Random Forest 0.7718 on this split; our replica of its Random Forest gives 0.7593, and our v2 gets 0.7706. Against the REPORTED figure v2 is 0.0012 lower on that single split; on 10 other splits and on 15-fold CV v2 is higher than all three models.",
 "Gains in accuracy are modest (about +1 to +2 points) and significant at p < 0.05 in the 15-fold CV; a single split of 885 students moves accuracy by about +-1.4 points (SD over 10 splits).",
 "v2 improves accuracy but finds fewer Enrolled students than v1 (recall 0.51 vs 0.60). Against the paper's models, v2 still has higher Enrolled F1 than XGBoost and logistic regression, and equal to Random Forest.",
 "The choice between v1 and v2 was made from out-of-fold results on the training part of the paper's own split. The 10 other splits and the CV folds share students with that training part, so they are not completely independent of the choice; the choice is a single binary decision.",
 "The journal article is from a regional Indonesian journal (JEECOM); results are quoted from the article page, whose result tables we could read; the numbers we reproduced match them for XGBoost and logistic regression."]
for i, n in enumerate(N, 3):
    x = ws.cell(i, 1, f"{i-2}. {n}"); x.alignment = Alignment(wrap_text=True, vertical="top"); ws.row_dimensions[i].height = 52
widths(ws, [150])
wb.save("/home/claude/outputs/dropout_journal_paper_comparison.xlsx")
