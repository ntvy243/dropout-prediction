"""Protocol A (leaky protocol of the first baseline) with the shallow v2 ensemble. Same 10 seeds / splits as run_protocol_a.py; paired with the baseline XGBoost (default).
Variants: v2 on the baseline's SMOTE'd top-20 features; v2 with the 18 engineered features added before IQR/top-k/SMOTE. SMOTE equalises the classes, so sqrt weights = 1."""
import numpy as np, pandas as pd
from scipy import stats
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from common import *
CATP = dict(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=5); XGBP = dict(n_estimators=300, max_depth=4, learning_rate=0.05, min_child_weight=3, reg_lambda=5)
def v2(): return [Model("lgbm", BASE_LGB, "sqrt"), Model("cat", CATP, "sqrt"), Model("xgb", XGBP, "sqrt"), Model("et", {}, "sqrt")]
_, X, y = load_data()
def prep(Xf, cont):
    q1, q3 = Xf[cont].quantile(.25), Xf[cont].quantile(.75); lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    Xc = Xf.copy(); Xc[cont] = Xc[cont].clip(lower=lo, upper=hi, axis=1); top, _ = top_k_features(Xc, y); return Xc, top
X1c, top1 = prep(X, CONT_COLS)
XE = add_features(X); engc = [c for c in XE.columns if c not in X.columns]; X2c, top2 = prep(XE, CONT_COLS + [c for c in engc if XE[c].nunique() > 2])
def split(Xc, top, seed):
    Xs, ys = SMOTE(random_state=seed).fit_resample(Xc[top], y); Xs = MinMaxScaler().fit_transform(Xs)
    return train_test_split(Xs, ys, test_size=0.2, stratify=ys, random_state=seed)
rows = []
for seed in range(10):
    Xa, Xb, ya, yb = split(X1c, top1, seed)
    px = XGBClassifier(eval_metric="mlogloss", random_state=SEED, n_jobs=N_JOBS).fit(Xa, ya).predict_proba(Xb); pv = soft_vote(v2(), Xa, ya, Xb)
    Xa2, Xb2, ya2, yb2 = split(X2c, top2, seed); pe = soft_vote(v2(), Xa2, ya2, Xb2)
    for nm, p, yy in [("XGBoost paper default", px, yb), ("OURS v2 (same inputs)", pv, yb), ("OURS v2 (+18 features)", pe, yb2)]:
        rows.append(dict(seed=seed, model=nm, **metrics(yy, p.argmax(1))))
    print(seed, flush=True)
d = pd.DataFrame(rows); d.to_csv(RESULTS / "protocolA_v2_seeds.csv", index=False); pw = d.pivot(index="seed", columns="model", values="accuracy"); out = []
for nm in pw.columns:
    r = dict(model=nm, mean_accuracy=pw[nm].mean(), std=pw[nm].std(ddof=1), mean_macro_f1=d[d.model == nm].macro_f1.mean(), p_vs_0_83=stats.ttest_1samp(pw[nm], .83)[1])
    if nm != "XGBoost paper default": df_ = pw[nm] - pw["XGBoost paper default"]; r.update(diff_vs_paper=df_.mean(), paired_p=stats.ttest_1samp(df_, 0)[1], wins=int((df_ > 0).sum()))
    out.append(r)
pd.DataFrame(out).to_csv(RESULTS / "protocolA_v2_summary.csv", index=False); print(pd.DataFrame(out).round(4).to_string())
