"""Exploration on the TRAINING part only (5-fold out-of-fold predictions), seed-42 split of the dropout data. Nothing here looks at the 885 test students."""
import json, time, itertools
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from common import *

_, X, y = load_data(); tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
Xtr, ytr = X.iloc[tr], y[tr]; F = add_features(Xtr); opt = json.load(open(RESULTS / "optuna_best.json"))
class LR:
    def fit(self, A, yy): self.m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.3)).fit(A, yy); return self
    def predict_proba(self, A): return self.m.predict_proba(A)
class RF:
    def __init__(self, cw=None): self.cw = cw
    def fit(self, A, yy): self.m = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight=self.cw, random_state=SEED, n_jobs=N_JOBS).fit(A, yy); return self
    def predict_proba(self, A): return self.m.predict_proba(A)
P = opt["params"]
cands = {"lgbm_none": lambda: Model("lgbm", P, None), "lgbm_sqrt": lambda: Model("lgbm", P, "sqrt"), "lgbm_bal": lambda: Model("lgbm", P, "balanced"),
         "cat_none": lambda: Model("cat", dict(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=5), None), "cat_sqrt": lambda: Model("cat", dict(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=5), "sqrt"),
         "xgb_none": lambda: Model("xgb", dict(n_estimators=300, max_depth=4, learning_rate=0.05, min_child_weight=3, reg_lambda=5), None),
         "xgb_sqrt": lambda: Model("xgb", dict(n_estimators=300, max_depth=4, learning_rate=0.05, min_child_weight=3, reg_lambda=5), "sqrt"),
         "et_sqrt": lambda: Model("et", {}, "sqrt"), "et_bal": lambda: Model("et", {}, "balanced"), "rf_none": lambda: RF(None), "rf_bal": lambda: RF("balanced"), "lr": lambda: LR()}
skf = StratifiedKFold(5, shuffle=True, random_state=11); oof = {}
t0 = time.time()
for n, mk in cands.items():
    o = np.zeros((len(ytr), 3))
    for a, b in skf.split(F, ytr): o[b] = mk().fit(F.iloc[a], ytr[a]).predict_proba(F.iloc[b])
    oof[n] = o; print(f"{n:10s} acc {accuracy_score(ytr, o.argmax(1)):.4f} macroF1 {f1_score(ytr, o.argmax(1), average='macro'):.4f}  {time.time() - t0:.0f}s", flush=True)
np.savez(RESULTS / "oof_train_v2.npz", y=ytr, **oof)
