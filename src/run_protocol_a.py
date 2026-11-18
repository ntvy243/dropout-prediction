"""Protocol A - the first baseline's own protocol (SMOTE before the train/test split; leaky).

Selection of the ensemble members uses 5-fold CV on the training part of ONE split (seed 42) and is fixed before the
ten evaluation seeds are run. Each seed draws a new SMOTE sample and a new stratified 80/20 split; the baseline's XGBoost
(default settings) is evaluated on exactly the same split. Usage:  python run_protocol_a.py [--fast]
"""
import sys
import numpy as np, pandas as pd
from scipy import stats
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from common import *

FAST = "--fast" in sys.argv
N_SEEDS = 2 if FAST else 10
_, X, y = load_data()

# As in the baseline: IQR limits and the top-20 selection are computed on the full data.
lo, hi = iqr_limits(X)
Xc = iqr_clip(X, lo, hi)
top20, _ = top_k_features(Xc, y)


def protocol_a_split(seed):
    Xs, ys = SMOTE(random_state=seed).fit_resample(Xc[top20], y)         # balancing BEFORE the split (leaky)
    Xs = MinMaxScaler().fit_transform(Xs)
    return train_test_split(Xs, ys, test_size=0.2, stratify=ys, random_state=seed)


CANDIDATES = {
    "ExtraTrees (sqrt)": lambda: ExtraTreesClassifier(n_estimators=1000, max_features="sqrt", random_state=SEED, n_jobs=N_JOBS),
    "ExtraTrees (0.5)": lambda: ExtraTreesClassifier(n_estimators=1000, max_features=0.5, random_state=SEED, n_jobs=N_JOBS),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=1000, random_state=SEED, n_jobs=N_JOBS),
    "kNN k=1": lambda: KNeighborsClassifier(1),
    "kNN k=5 (distance)": lambda: KNeighborsClassifier(5, weights="distance"),
    "LightGBM (63 leaves)": lambda: LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                                                   colsample_bytree=0.8, random_state=SEED, n_jobs=N_JOBS, verbose=-1),
    "CatBoost (depth 8)": lambda: CatBoostClassifier(iterations=800, depth=8, learning_rate=0.08, random_seed=SEED, verbose=0, thread_count=N_JOBS),
    "XGBoost (depth 8)": lambda: XGBClassifier(n_estimators=600, max_depth=8, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8,
                                               eval_metric="mlogloss", random_state=SEED, n_jobs=N_JOBS),
    "XGBoost paper default": lambda: XGBClassifier(eval_metric="mlogloss", random_state=SEED, n_jobs=N_JOBS),
}
if FAST:
    CANDIDATES = {k: CANDIDATES[k] for k in ["ExtraTrees (sqrt)", "LightGBM (63 leaves)", "XGBoost (depth 8)", "XGBoost paper default"]}
REF, ENS = "XGBoost paper default", "ENSEMBLE top-3 (ours)"

# --- 1) pre-registered selection: 5-fold CV on the training part of seed 42 (test part untouched)
Xa, Xb, ya, yb = protocol_a_split(SEED)
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
sel = []
for name, make in CANDIDATES.items():
    acc = [accuracy_score(ya[v], make().fit(Xa[t], ya[t]).predict(Xa[v])) for t, v in skf.split(Xa, ya)]
    sel.append(dict(model=name, cv_accuracy=np.mean(acc), cv_std=np.std(acc, ddof=1)))
    print(f"{name:24s} CV acc {np.mean(acc):.4f}", flush=True)
sel = pd.DataFrame(sel).sort_values("cv_accuracy", ascending=False).reset_index(drop=True)
top3 = [m for m in sel.model if m != REF][:3]
sel["selected_for_ensemble"] = sel.model.isin(top3)
sel.to_csv(RESULTS / "protocolA_cv_selection.csv", index=False)
print("ensemble members:", top3)

# --- 2) ten evaluation seeds, paired on identical splits
rows = []
for seed in range(N_SEEDS):
    Xa, Xb, ya, yb = protocol_a_split(seed)
    proba = {nm: CANDIDATES[nm]().fit(Xa, ya).predict_proba(Xb) for nm in [REF] + top3}
    proba[ENS] = np.mean([proba[k] for k in top3], axis=0)
    for nm, p in proba.items():
        rows.append(dict(seed=seed, model=nm, **metrics(yb, p.argmax(1))))
    print(seed, {k: round(accuracy_score(yb, p.argmax(1)), 4) for k, p in proba.items()}, flush=True)
seeds = pd.DataFrame(rows); seeds.to_csv(RESULTS / "protocolA_seeds.csv", index=False)

# --- 3) tests: paired t-test versus the baseline model, one-sample t-test versus the reported 0.83
pw = seeds.pivot(index="seed", columns="model", values="accuracy")
out = []
for nm in [REF] + top3 + [ENS]:
    sub = seeds[seeds.model == nm]
    r = dict(model=nm, mean_accuracy=pw[nm].mean(), std_accuracy=pw[nm].std(ddof=1),
             mean_macro_f1=sub.macro_f1.mean(), mean_f1_Enrolled=sub.f1_Enrolled.mean())
    t83, p83 = stats.ttest_1samp(pw[nm], 0.83); r.update(diff_vs_0_83=pw[nm].mean() - 0.83, p_vs_0_83=p83)
    if nm != REF:
        d = pw[nm] - pw[REF]; r.update(diff_vs_paper_model=d.mean(), paired_p=stats.ttest_1samp(d, 0)[1], wins=int((d > 0).sum()), seeds=len(d))
    out.append(r)
pd.DataFrame(out).to_csv(RESULTS / "protocolA_summary.csv", index=False)
print(pd.DataFrame(out).round(4).to_string())
