"""Ablation of the proposed ensemble under the leakage-free protocol, on the same 11 hold-outs and 15 CV folds as run_paper3_compare.py / run_islam_fair.py.
Variants: v2 on the 36 raw features only; single LightGBM, single XGBoost, single CatBoost, single ExtraTrees (sqrt weights, 54 features); v2 with balanced weights = v1 (already in paper3_all.csv)."""
import time
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from common import *
_, X, y = load_data(); XF = add_features(X)
CATP = dict(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=5); XGBP = dict(n_estimators=300, max_depth=4, learning_rate=0.05, min_child_weight=3, reg_lambda=5)
def v2(): return [Model("lgbm", BASE_LGB, "sqrt"), Model("cat", CATP, "sqrt"), Model("xgb", XGBP, "sqrt"), Model("et", {}, "sqrt")]
class Vote:
    def __init__(self, mk): self.mk = mk
    def fit(self, A, yy): self.ms = [m.fit(A, yy) for m in self.mk()]; return self
    def predict(self, A): return np.mean([m.predict_proba(A) for m in self.ms], 0).argmax(1)
class One(Vote): pass
V = {"v2, 36 raw features only": ("raw", lambda: Vote(v2)),
     "v2 single LightGBM": ("eng", lambda: One(lambda: [Model("lgbm", BASE_LGB, "sqrt")])),
     "v2 single XGBoost": ("eng", lambda: One(lambda: [Model("xgb", XGBP, "sqrt")])),
     "v2 single CatBoost": ("eng", lambda: One(lambda: [Model("cat", CATP, "sqrt")])),
     "v2 single ExtraTrees": ("eng", lambda: One(lambda: [Model("et", {}, "sqrt")]))}
data = {"raw": X, "eng": XF}; rows = []; t0 = time.time()
def go(tr, te, kind, split):
    for n, (fx, mk) in V.items():
        pred = mk().fit(data[fx].iloc[tr], y[tr]).predict(data[fx].iloc[te]); rows.append(dict(kind=kind, split=split, model=n, n_train=len(tr), n_test=len(te), **metrics(y[te], pred)))
for s in [SEED] + list(range(10)):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=s); go(tr, te, "holdout", s); print("holdout", s, f"{time.time()-t0:.0f}s", flush=True)
for f, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(X, y)):
    go(tr, te, "cv", f); print("cv", f, f"{time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "ablation_all.csv", index=False)
print("done")
