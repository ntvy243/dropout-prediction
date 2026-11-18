"""Protocol B - the second baseline's protocol: drop 8 low-correlation columns, plain random 80/20 split, no balancing.

Repeated on 30 random splits; the proposed ensemble (hand-set parameters) is evaluated on exactly the same splits, with the
same 28 columns and with the 18 engineered features added to all columns.  Usage:  python run_protocol_b.py [--fast]
"""
import sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from common import *

N_SPLITS = 3 if "--fast" in sys.argv else 30
_, X, y = load_data()
X28 = X.drop(columns=DROP_PAPER2)
XF = add_features(X)
rows = []
for seed in range(N_SPLITS):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)          # not stratified, as in the baseline
    A, B, ya, yb = X28.iloc[tr], X28.iloc[te], y[tr], y[te]
    pred = {
        "Paper 2 Random Forest": RandomForestClassifier(random_state=seed, n_jobs=N_JOBS).fit(A, ya).predict(B),
        "Paper 2 AdaBoost": AdaBoostClassifier(random_state=seed).fit(A, ya).predict(B),
        "Paper 2 SVM": make_pipeline(StandardScaler(), SVC()).fit(A, ya).predict(B),
        "Paper 2 Decision Tree": DecisionTreeClassifier(random_state=seed).fit(A, ya).predict(B),
        "Paper 2 KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()).fit(A, ya).predict(B),
        "Ours: ensemble (same 28 features)": soft_vote(ensemble_members(), A, ya, B).argmax(1),
        "Ours: ensemble (+engineered features)": soft_vote(ensemble_members(), XF.iloc[tr], ya, XF.iloc[te]).argmax(1),
    }
    for name, p in pred.items():
        rows.append(dict(seed=seed, model=name, **metrics(yb, p)))
    print(seed, {k: round(accuracy_score(yb, p), 4) for k, p in pred.items()}, flush=True)
R = pd.DataFrame(rows); R.to_csv(RESULTS / "protocolB_splits.csv", index=False)

summary = R.groupby("model")[["accuracy", "macro_f1", "f1_Enrolled"]].agg(["mean", "std"])
summary.columns = ["_".join(c) for c in summary.columns]; summary.to_csv(RESULTS / "protocolB_summary.csv")
print(summary.round(4).to_string())

# paired tests against the Random Forest of the second baseline
rf, tests = "Paper 2 Random Forest", []
for model in ["Ours: ensemble (same 28 features)", "Ours: ensemble (+engineered features)"]:
    for metric in ["accuracy", "macro_f1", "f1_Enrolled"]:
        pv = R.pivot(index="seed", columns="model", values=metric); d = pv[model] - pv[rf]
        tests.append(dict(model=model, metric=metric, mean_diff=d.mean(), p_value=stats.ttest_1samp(d, 0)[1], wins=int((d > 0).sum()), splits=len(d)))
acc = R.pivot(index="seed", columns="model", values="accuracy")
tests.append(dict(model="Paper 2 Random Forest (re-run)", metric="splits above reported 0.8057", mean_diff=int((acc[rf] > 0.8057).sum()), p_value=np.nan, wins=np.nan, splits=len(acc)))
pd.DataFrame(tests).to_csv(RESULTS / "protocolB_tests.csv", index=False)
print(pd.DataFrame(tests).round(4).to_string())
