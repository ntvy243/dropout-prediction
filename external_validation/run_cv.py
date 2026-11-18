"""Repeated stratified 5-fold CV (3 repeats = 15 folds) on UCI Student Performance for the proposed method and the published pipelines.
Everything (feature selection, scaling, SMOTE, fitting) is done on the training fold only. Usage: python run_cv.py [--fast]"""
import sys, time, itertools
from sklearn.model_selection import RepeatedStratifiedKFold
from sp_common import *

FAST = "--fast" in sys.argv
rows = []
MODELS = {"OURS (ensemble + engineered features)": ("eng", lambda: Ensemble()),
          "OURS without engineered features": ("raw", lambda: Ensemble()),
          "Islam et al. (2025) pipeline": ("raw", lambda: Islam()),
          "Kesgin et al. (2025) Logistic Regression": ("raw", lambda: Kesgin("LR")),
          "Kesgin et al. (2025) Random Forest": ("raw", lambda: Kesgin("RF")),
          "Kesgin et al. (2025) XGBoost": ("raw", lambda: Kesgin("XGB"))}
t0 = time.time()
for subject, grades, target in itertools.product(["mat", "por"], [True, False], ["binary", "five"]):
    df = load(subject); y = make_target(df.G3, target)
    X = {"eng": prepare(df, grades, True), "raw": prepare(df, grades, False)}
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1 if FAST else 3, random_state=SEED)
    for fold, (tr, te) in enumerate(rskf.split(X["raw"], y)):
        for name, (fx, mk) in MODELS.items():
            m = mk().fit(X[fx].iloc[tr], y[tr]); p = m.predict_proba(X[fx].iloc[te])
            if p.shape[1] != len(np.unique(y)):          # a class missing from the training fold
                full = np.zeros((len(te), len(np.unique(y)))); full[:, m.classes_] = p; p = full
            rows.append(dict(subject=subject, grades=("G1+G2 included" if grades else "G1+G2 excluded"), target=target, fold=fold, model=name, n_train=len(tr), n_test=len(te), **metrics(y[te], p)))
    print(subject, grades, target, f"{time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / ("cv_folds_fast.csv" if FAST else "cv_folds.csv"), index=False)
print("done")
