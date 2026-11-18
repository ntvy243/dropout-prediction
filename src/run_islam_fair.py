"""Islam et al. (2025) pipeline run WITHOUT leakage, on exactly the same splits as run_paper3_compare.py (so the proposed v1/v2 rows of
paper3_all.csv are paired with it). Per split, fitted on the training part only: IQR limits -> ExtraTrees top-20 -> SMOTE -> Min-Max -> classifier.
Splits: 11 stratified 80/20 hold-outs (seeds 42 and 0-9) and repeated stratified 5-fold CV x3 (15 folds, random_state 42)."""
import time
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from common import *
_, X, y = load_data()
CLF = {"Islam XGB (fair)": lambda: XGBClassifier(random_state=SEED, eval_metric="mlogloss", n_jobs=N_JOBS),
       "Islam RF (fair)": lambda: RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS),
       "Islam GB (fair)": lambda: GradientBoostingClassifier(random_state=SEED),
       "Islam DT (fair)": lambda: DecisionTreeClassifier(random_state=SEED)}
def run(tr, te, kind, split, rows):
    Xtr, Xte = X.iloc[tr], X.iloc[te]; lo, hi = iqr_limits(Xtr); Xtr, Xte = iqr_clip(Xtr, lo, hi), iqr_clip(Xte, lo, hi)
    top, _ = top_k_features(Xtr, y[tr]); Xs, ys = SMOTE(random_state=SEED).fit_resample(Xtr[top], y[tr]); sc = MinMaxScaler().fit(Xs)
    for n, mk in CLF.items():
        pred = mk().fit(sc.transform(Xs), ys).predict(sc.transform(Xte[top])); rows.append(dict(kind=kind, split=split, model=n, n_train=len(tr), n_test=len(te), **metrics(y[te], pred)))
rows = []; t0 = time.time()
for s in [SEED] + list(range(10)):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=s); run(tr, te, "holdout", s, rows); print("holdout", s, f"{time.time()-t0:.0f}s", flush=True)
for f, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(X, y)):
    run(tr, te, "cv", f, rows); print("cv", f, f"{time.time()-t0:.0f}s", flush=True)
pd.DataFrame(rows).to_csv(RESULTS / "islam_fair_all.csv", index=False); print("done")
