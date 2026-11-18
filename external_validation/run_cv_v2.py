"""Same repeated stratified 5-fold CV (15 folds, same folds as run_cv.py) with the v2 ensemble: square-root class weights (the setting chosen on the dropout dataset's
training part, not on this data). Output: results/cv_folds_v2.csv"""
import itertools
from sklearn.model_selection import RepeatedStratifiedKFold
from sp_common import *
rows = []
for subject, grades, target in itertools.product(["mat", "por"], [True, False], ["binary", "five"]):
    df = load(subject); y = make_target(df.G3, target); X = prepare(df, grades, True)
    for fold, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(X, y)):
        m = Ensemble(power=0.5).fit(X.iloc[tr], y[tr]); p = m.predict_proba(X.iloc[te])
        if p.shape[1] != len(np.unique(y)): full = np.zeros((len(te), len(np.unique(y)))); full[:, m.classes_] = p; p = full
        rows.append(dict(subject=subject, grades=("G1+G2 included" if grades else "G1+G2 excluded"), target=target, fold=fold, model="OURS v2 (sqrt class weights)", n_train=len(tr), n_test=len(te), **metrics(y[te], p)))
    print(subject, grades, target, flush=True); pd.DataFrame(rows).to_csv(RESULTS / "cv_folds_v2.csv", index=False)
print("done")
