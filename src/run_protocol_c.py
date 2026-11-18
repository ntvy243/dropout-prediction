"""Protocol C - leakage-free evaluation (split first; every fitted step uses the training part only).

Stages: (1) baseline-1 pipeline run fairly; (2) Optuna tuning of LightGBM; (3) 4-model ensemble; (4) hold-out metrics with
bootstrap CIs and McNemar test; (5) repeated stratified 5x2 CV with the Nadeau-Bengio test; (6) SHAP; (7) fairness by group.
Usage:  python run_protocol_c.py [--fast]
"""
import json, sys
import numpy as np, pandas as pd
import optuna, shap
from scipy import stats
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from common import *

FAST = "--fast" in sys.argv
N_TRIALS, N_BOOT = (3, 200) if FAST else (80, 2000)
optuna.logging.set_verbosity(optuna.logging.WARNING)
_, X, y = load_data()

# ------------------------------------------------------------------ 1) split and baseline 1 run fairly
idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
Xtr, Xte, ytr, yte = X.iloc[idx_tr], X.iloc[idx_te], y[idx_tr], y[idx_te]
print("train", Xtr.shape, "test", Xte.shape, "test class counts", np.bincount(yte))

lo, hi = iqr_limits(Xtr)                                              # limits from the training part only
Xtr_c, Xte_c = iqr_clip(Xtr, lo, hi), iqr_clip(Xte, lo, hi)
top20, _ = top_k_features(Xtr_c, ytr)                                 # selection on the training part only
Xs, ys = SMOTE(random_state=SEED).fit_resample(Xtr_c[top20], ytr)     # SMOTE on the training part only
scaler = MinMaxScaler().fit(Xs)
paper_xgb = XGBClassifier(random_state=SEED, eval_metric="mlogloss", n_jobs=N_JOBS).fit(scaler.transform(Xs), ys)
pred_paper = paper_xgb.predict(scaler.transform(Xte_c[top20]))

# ------------------------------------------------------------------ 2) Optuna tuning of LightGBM (training part only)
def oof(make, A, yy, seed=SEED, B=None):
    out = np.zeros((len(yy), 3))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(A, yy):
        out[b] = make().fit(A.iloc[a], yy[a]).predict_proba(A.iloc[b])
    return out

def objective(trial):
    p = dict(n_estimators=trial.suggest_int("n_estimators", 150, 600), learning_rate=trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
             num_leaves=trial.suggest_int("num_leaves", 4, 40), max_depth=trial.suggest_categorical("max_depth", [-1, 3, 4, 5, 6, 8]),
             min_child_samples=trial.suggest_int("min_child_samples", 5, 60), subsample=trial.suggest_float("subsample", 0.5, 1.0),
             colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0), reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 30, log=True),
             reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True), min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.5))
    cw = trial.suggest_categorical("cw", ["balanced", "sqrt"])
    return f1_score(ytr, oof(lambda: Model("lgbm", p, cw), Xtr, ytr, seed=7).argmax(1), average="macro")   # tuning folds: seed 7

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS)
lgb_params = dict(study.best_params); lgb_cw = lgb_params.pop("cw")
json.dump(dict(params=lgb_params, class_weight=lgb_cw, best_oof_macro_f1=study.best_value), open(RESULTS / "optuna_best.json", "w"), indent=1)
print("best tuning macro-F1", round(study.best_value, 4), lgb_cw, lgb_params)

# ------------------------------------------------------------------ 3-4) ensemble and hold-out comparison
members = ensemble_members(lgb_params, lgb_cw)
proba = np.mean([m.fit(Xtr, ytr).predict_proba(Xte) for m in members], axis=0)
pred_final = proba.argmax(1)
m_final, m_paper = metrics(yte, pred_final), metrics(yte, pred_paper)

def fast_metrics(yt, yp):
    cm = np.bincount(3 * yt + yp, minlength=9).reshape(3, 3).astype(float)
    tp = np.diag(cm); prec = tp / np.maximum(cm.sum(0), 1); rec = tp / np.maximum(cm.sum(1), 1); f = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return dict(accuracy=tp.sum() / cm.sum(), macro_f1=f.mean(), f1_Enrolled=f[1], recall_Enrolled=rec[1])

rng = np.random.RandomState(0); diffs, bf, bp = [], [], []
for _ in range(N_BOOT):
    ix = rng.randint(0, len(yte), len(yte)); a, b = fast_metrics(yte[ix], pred_final[ix]), fast_metrics(yte[ix], pred_paper[ix])
    bf.append(a); bp.append(b); diffs.append({k: a[k] - b[k] for k in a})
ci = lambda lst, k: np.percentile([d[k] for d in lst], [2.5, 97.5])
ok_f, ok_p = pred_final == yte, pred_paper == yte
n01, n10 = int((ok_f & ~ok_p).sum()), int((~ok_f & ok_p).sum())
mcnemar_p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
rows = []
for label, m, boot in [("Paper method run fairly: XGBoost", m_paper, bp), ("Proposed 4-model ensemble", m_final, bf)]:
    r = dict(model=label, **{k: m[k] for k in ["accuracy", "macro_f1", "f1_Dropout", "f1_Enrolled", "f1_Graduate", "recall_Enrolled"]})
    for k in ["accuracy", "macro_f1", "f1_Enrolled"]:
        r[f"{k}_ci_low"], r[f"{k}_ci_high"] = ci(boot, k)
    rows.append(r)
pd.DataFrame(rows).to_csv(RESULTS / "protocolC_holdout.csv", index=False)
pd.DataFrame([dict(metric=k, mean_diff=np.mean([d[k] for d in diffs]), ci_low=ci(diffs, k)[0], ci_high=ci(diffs, k)[1],
                   mcnemar_exact_p_accuracy=mcnemar_p) for k in ["accuracy", "macro_f1", "f1_Enrolled", "recall_Enrolled"]]
             ).to_csv(RESULTS / "protocolC_holdout_diff.csv", index=False)
json.dump({"paper_method_fair": confusion_matrix(yte, pred_paper, labels=[0, 1, 2]).tolist(),
           "proposed_ensemble": confusion_matrix(yte, pred_final, labels=[0, 1, 2]).tolist(),
           "paper1_fig7_xgb_reported": [[353, 54, 37], [28, 363, 48], [15, 47, 381]]}, open(RESULTS / "confusion_matrices.json", "w"))
print("hold-out:", {k: round(m_final[k], 4) for k in ["accuracy", "macro_f1", "f1_Enrolled", "recall_Enrolled"]}, "| paper method:",
      {k: round(m_paper[k], 4) for k in ["accuracy", "macro_f1", "f1_Enrolled", "recall_Enrolled"]}, "| McNemar p", round(mcnemar_p, 3))

# ------------------------------------------------------------------ 5) repeated stratified 5x2 CV
folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED).split(X, y))
if FAST: folds = folds[:2]
rows = []
for k, (a, b) in enumerate(folds, 1):
    Xa, Xb, ya, yb = X.iloc[a], X.iloc[b], y[a], y[b]
    paper_pipe = ImbPipeline([("sel", SelectFromModel(ExtraTreesClassifier(n_estimators=200, random_state=SEED, n_jobs=N_JOBS), max_features=20, threshold=-np.inf)),
                              ("smote", SMOTE(random_state=SEED)), ("clf", XGBClassifier(eval_metric="mlogloss", random_state=SEED, n_jobs=N_JOBS))])
    p_paper = paper_pipe.fit(Xa, ya).predict(Xb)
    p_ens = soft_vote(ensemble_members(lgb_params, lgb_cw), Xa, ya, Xb).argmax(1)
    for name, p in [("Paper method: top-20 + SMOTE + XGBoost", p_paper), ("Proposed 4-model ensemble", p_ens)]:
        rows.append(dict(fold=k, model=name, **metrics(yb, p)))
    print("fold", k, flush=True)
cv = pd.DataFrame(rows); cv.to_csv(RESULTS / "protocolC_cv_folds.csv", index=False)
cv.groupby("model")[["accuracy", "macro_f1", "f1_Enrolled"]].agg(["mean", "std"]).to_csv(RESULTS / "protocolC_cv_summary.csv")
tests = []
for metric in ["accuracy", "macro_f1", "f1_Enrolled"]:
    pv = cv.pivot(index="fold", columns="model", values=metric); d = pv["Proposed 4-model ensemble"] - pv["Paper method: top-20 + SMOTE + XGBoost"]
    t, p = nadeau_bengio(d, 0.8 * len(y), 0.2 * len(y))
    tests.append(dict(metric=metric, mean_diff=d.mean(), corrected_t=t, corrected_p=p, naive_p=stats.ttest_1samp(d, 0)[1], wins=int((d > 0).sum()), folds=len(d)))
pd.DataFrame(tests).to_csv(RESULTS / "protocolC_cv_tests.csv", index=False)
print(pd.DataFrame(tests).round(4).to_string())

# ------------------------------------------------------------------ 6) SHAP for the tuned LightGBM member, hold-out students
lgbm = members[0].m
sv = np.asarray(shap.TreeExplainer(lgbm).shap_values(Xte))
if sv.ndim == 3 and sv.shape[0] != 3: sv = np.transpose(sv, (2, 0, 1))             # -> (classes, students, features)
imp = pd.DataFrame({"feature": Xte.columns, "mean_abs_shap": np.abs(sv).mean(axis=(0, 1))})
for c, name in enumerate(CLASS_NAMES): imp[f"shap_{name}"] = np.abs(sv[c]).mean(axis=0)
imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True); imp["rank"] = imp.index + 1
imp.to_csv(RESULTS / "shap_importance.csv", index=False)
np.save(RESULTS / "shap_values.npy", sv); Xte.to_csv(RESULTS / "shap_holdout_features.csv", index=False)

# ------------------------------------------------------------------ 7) fairness by group (ensemble, hold-out)
groups = {"Gender": ("Gender", {0: "Female", 1: "Male"}), "Displaced": ("Displaced", {0: "Not displaced", 1: "Displaced"}),
          "International": ("International", {0: "Domestic", 1: "International"})}
fair = []
for gname, (col, lab) in groups.items():
    for v, vname in lab.items():
        mask = (Xte[col] == v).to_numpy(); m = metrics(yte[mask], pred_final[mask])
        fair.append(dict(variable=gname, group=vname, n=int(mask.sum()), accuracy=m["accuracy"], macro_f1=m["macro_f1"],
                         recall_Dropout=m["recall_Dropout"], recall_Enrolled=m["recall_Enrolled"]))
pd.DataFrame(fair).to_csv(RESULTS / "fairness.csv", index=False)
print(pd.DataFrame(fair).round(3).to_string())
