"""Replication of the single train/test split protocol of Kesgin et al. (2025) and comparison with the proposed ensemble.

(1) 'Paper text' protocol: Mathematics file (395 students), all predictors incl. G1, G2, pass = G3 >= 10, stratified 80/20 (316/79), SMOTE on the
    training part, LR / RF(300, 10, 5) / XGB(200, 5, 0.1). Run on 10 different seeds (the paper reports one split).
(2) 'Repository code' protocol (github.com/kdrksgn/fairsynedu): Portuguese file (649), 10 predictors, plain random 80/20 split seed 42, SMOTE, grid-searched RF and XGB.
Output: results/split_paper_text.csv, results/split_repo_code.csv"""
from sklearn.model_selection import train_test_split, GridSearchCV
from sp_common import *

# ---- (1) paper text, 10 seeds
df = load("mat"); y = make_target(df.G3, "binary"); Xraw = prepare(df, True, False); Xeng = prepare(df, True, True); rows = []
for seed in range(10):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    for name, fx, mk in [("Kesgin LR", Xraw, lambda: Kesgin("LR", seed)), ("Kesgin RF", Xraw, lambda: Kesgin("RF", seed)), ("Kesgin XGBoost", Xraw, lambda: Kesgin("XGB", seed)),
                         ("Islam et al. pipeline", Xraw, lambda: Islam(seed)), ("OURS", Xeng, lambda: Ensemble(seed))]:
        p = mk().fit(fx.iloc[tr], y[tr]).predict_proba(fx.iloc[te]); rows.append(dict(seed=seed, model=name, n_test=len(te), **metrics(y[te], p)))
    print("seed", seed, flush=True)
pd.DataFrame(rows).to_csv(RESULTS / "split_paper_text.csv", index=False)

# ---- (2) repository code (por file, 10 predictors, non-stratified split seed 42, grid search with f1 scoring)
df = load("por"); y = (df.G3 >= 10).astype(int).values
feats = ["G1", "G2", "absences", "failures", "Medu", "Fedu", "sex", "school", "Pstatus", "famsize"]
X = pd.get_dummies(df[feats], columns=["sex", "school", "Pstatus", "famsize"], drop_first=True, dtype=float)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
Xs, ys = SMOTE(random_state=42).fit_resample(Xtr, ytr); out = []
lr = LogisticRegression(max_iter=1000).fit(Xs, ys)
rf = GridSearchCV(RandomForestClassifier(random_state=42), dict(n_estimators=[100, 200], max_depth=[10, 20, None]), cv=5, scoring="f1").fit(Xs, ys).best_estimator_
xg = GridSearchCV(XGBClassifier(eval_metric="logloss"), dict(n_estimators=[100, 200], max_depth=[3, 6, 9], learning_rate=[0.01, 0.1]), cv=5, scoring="f1").fit(Xs, ys).best_estimator_
for name, m in [("Repo Logistic Regression", lr), ("Repo Random Forest (grid)", rf), ("Repo XGBoost (grid)", xg)]:
    out.append(dict(model=name, n_test=len(yte), **metrics(yte, m.predict_proba(Xte))))
Xe = prepare(df, True, True); eo = Ensemble(42).fit(Xe.iloc[Xtr.index], y[Xtr.index]).predict_proba(Xe.iloc[Xte.index])
out.append(dict(model="OURS (same split, all predictors)", n_test=len(yte), **metrics(yte, eo)))
pd.DataFrame(out).to_csv(RESULTS / "split_repo_code.csv", index=False); print(pd.DataFrame(out).round(3).to_string())
