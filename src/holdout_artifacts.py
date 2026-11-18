"""Seed-42 hold-out artifacts for the paper: predictions of Islam XGB (fair), proposed v1 and v2; SHAP of the v2 LightGBM member; subgroup metrics with bootstrap CIs."""
import json
import shap
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from common import *
_, X, y = load_data(); XF = add_features(X)
CATP = dict(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=5); XGBP = dict(n_estimators=300, max_depth=4, learning_rate=0.05, min_child_weight=3, reg_lambda=5)
def ens(cw): return [Model("lgbm", BASE_LGB, cw), Model("cat", CATP, cw), Model("xgb", XGBP, cw), Model("et", {}, cw)]
tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
Xtr, Xte = X.iloc[tr], X.iloc[te]; lo, hi = iqr_limits(Xtr); Xtr_c, Xte_c = iqr_clip(Xtr, lo, hi), iqr_clip(Xte, lo, hi)
top, _ = top_k_features(Xtr_c, y[tr]); Xs, ys = SMOTE(random_state=SEED).fit_resample(Xtr_c[top], y[tr]); sc = MinMaxScaler().fit(Xs)
px = XGBClassifier(random_state=SEED, eval_metric="mlogloss", n_jobs=N_JOBS).fit(sc.transform(Xs), ys).predict_proba(sc.transform(Xte_c[top]))
P = {"Islam XGB (fair)": px}; members = {}
for nm, cw in [("Proposed v1", "balanced"), ("Proposed v2", "sqrt")]:
    ms = ens(cw); [m.fit(XF.iloc[tr], y[tr]) for m in ms]; members[nm] = ms; P[nm] = np.mean([m.predict_proba(XF.iloc[te]) for m in ms], 0)
yt = y[te]; np.savez(RESULTS / "holdout42_probs.npz", y=yt, te=te, **{k.replace(" ", "_"): v for k, v in P.items()})
cm = {k: confusion_matrix(yt, v.argmax(1), labels=[0, 1, 2]).tolist() for k, v in P.items()}; json.dump(cm, open(RESULTS / "holdout42_confusion.json", "w"))
print({k: metrics(yt, v.argmax(1))["accuracy"] for k, v in P.items()})
# SHAP on the LightGBM member of v2
lgb = members["Proposed v2"][0].m; ex = shap.TreeExplainer(lgb); sv = ex.shap_values(XF.iloc[te])
sv = np.stack(sv, -1) if isinstance(sv, list) else sv                      # (n, features, classes)
imp = pd.DataFrame(np.abs(sv).mean(0), index=XF.columns, columns=CLASS_NAMES); imp["Overall"] = imp.mean(1); imp.sort_values("Overall", ascending=False).to_csv(RESULTS / "shap_v2_importance.csv")
np.save(RESULTS / "shap_v2_values.npy", sv); XF.iloc[te].to_csv(RESULTS / "shap_v2_features.csv", index=False)
# subgroup metrics with 2000-resample bootstrap CIs (v2)
rng = np.random.default_rng(SEED); pred = P["Proposed v2"].argmax(1); g = X.iloc[te]; rows = []
for var, col, lab in [("Gender", "Gender", {0: "Female", 1: "Male"}), ("Displaced", "Displaced", {0: "Not displaced", 1: "Displaced"}), ("International", "International", {0: "Domestic", 1: "International"})]:
    for v, nm in lab.items():
        idx = np.where(g[col].values == v)[0]
        if len(idx) == 0: continue
        m = metrics(yt[idx], pred[idx]); boots = [metrics(yt[b], pred[b])["accuracy"] for b in (rng.choice(idx, len(idx)) for _ in range(2000))]
        rows.append(dict(variable=var, group=nm, n=len(idx), accuracy=m["accuracy"], acc_lo=np.percentile(boots, 2.5), acc_hi=np.percentile(boots, 97.5), macro_f1=m["macro_f1"], recall_Dropout=m["recall_Dropout"], recall_Enrolled=m["recall_Enrolled"]))
pd.DataFrame(rows).to_csv(RESULTS / "fairness_v2.csv", index=False); print(pd.DataFrame(rows).round(3).to_string()); print(imp.sort_values("Overall", ascending=False).head(10).round(3))
