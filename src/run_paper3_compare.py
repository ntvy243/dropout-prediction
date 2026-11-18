"""Journal-paper comparison on the dropout dataset (UCI 697): proposed ensemble v1 (balanced weights) and v2 (square-root class weights, chosen from
out-of-fold accuracy/macro-F1 on the TRAINING part of the seed-42 split) against the three models of Praptomo et al. (2026, JEECOM).
Fixed, hand-set hyper-parameters (BASE_LGB; no tuning on any test data). Two evaluations, everything fitted on the training part only:
 (1) 10 stratified 80/20 hold-out splits (seeds 0-9 plus the paper's seed 42), (2) repeated stratified 5-fold CV, 3 repeats (15 folds, Nadeau-Bengio test)."""
import sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import *

FAST = "--fast" in sys.argv
_, X, y = load_data(); XF = add_features(X)
CATP = dict(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=5); XGBP = dict(n_estimators=300, max_depth=4, learning_rate=0.05, min_child_weight=3, reg_lambda=5)
def v1(): return [Model("lgbm", BASE_LGB, "balanced"), Model("cat", CATP, "balanced"), Model("xgb", XGBP, "balanced"), Model("et", {}, "balanced")]
def v2(): return [Model("lgbm", BASE_LGB, "sqrt"), Model("cat", CATP, "sqrt"), Model("xgb", XGBP, "sqrt"), Model("et", {}, "sqrt")]
class Vote:
    def __init__(self, mk): self.mk = mk
    def fit(self, A, yy): self.ms = [m.fit(A, yy) for m in self.mk()]; return self
    def predict(self, A): return np.mean([m.predict_proba(A) for m in self.ms], 0).argmax(1)
class Sk:
    def __init__(self, est): self.est = est
    def fit(self, A, yy): self.est.fit(A, yy); return self
    def predict(self, A): return self.est.predict(A)
def paper_models():
    return {"Paper LR (JEECOM)": ("raw", lambda: Sk(make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)))),
            "Paper RF (JEECOM)": ("raw", lambda: Sk(make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=SEED, n_jobs=N_JOBS)))),
            "Paper XGB (JEECOM)": ("raw", lambda: Sk(XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.1, min_child_weight=3, subsample=1.0, colsample_bytree=0.9, gamma=0, eval_metric="mlogloss", random_state=SEED, n_jobs=N_JOBS)))}
MODELS = {"OURS v1 (balanced weights)": ("eng", lambda: Vote(v1)), "OURS v2 (sqrt weights)": ("eng", lambda: Vote(v2)), **paper_models()}
data = {"raw": X, "eng": XF}

# ---- selection record on the training part of the paper's split (OOF), v1 vs v2
tr0, _ = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED); sel = []
for name, mk in [("v1", v1), ("v2", v2)]:
    o = np.zeros((len(tr0), 3))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=11).split(XF.iloc[tr0], y[tr0]):
        o[b] = np.mean([m.fit(XF.iloc[tr0].iloc[a], y[tr0][a]).predict_proba(XF.iloc[tr0].iloc[b]) for m in mk()], 0)
    sel.append(dict(config=name, oof_accuracy=float((o.argmax(1) == y[tr0]).mean()), **{k: v for k, v in metrics(y[tr0], o.argmax(1)).items() if k in ("macro_f1", "f1_Enrolled")}))
pd.DataFrame(sel).round(4).to_csv(RESULTS / "paper3_selection_oof.csv", index=False); print(pd.DataFrame(sel).round(4), flush=True)

rows = []; t0 = time.time()
seeds = [SEED] + ([] if FAST else list(range(10)))
for s in seeds:                                       # (1) hold-out splits
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=s)
    for n, (fx, mk) in MODELS.items():
        pred = mk().fit(data[fx].iloc[tr], y[tr]).predict(data[fx].iloc[te]); rows.append(dict(kind="holdout", split=s, model=n, n_train=len(tr), n_test=len(te), **metrics(y[te], pred)))
    print("holdout seed", s, f"{time.time() - t0:.0f}s", flush=True)
pd.DataFrame(rows).to_csv(RESULTS / "paper3_holdout.csv", index=False)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1 if FAST else 3, random_state=SEED)
for f, (tr, te) in enumerate(cv.split(X, y)):         # (2) repeated CV
    for n, (fx, mk) in MODELS.items():
        pred = mk().fit(data[fx].iloc[tr], y[tr]).predict(data[fx].iloc[te]); rows.append(dict(kind="cv", split=f, model=n, n_train=len(tr), n_test=len(te), **metrics(y[te], pred)))
    print("cv fold", f, f"{time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / ("paper3_all_fast.csv" if FAST else "paper3_all.csv"), index=False)
print("done")
