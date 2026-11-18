"""Replicate Praptomo, Risman & Suri (2026), J. Electr. Eng. Comput. (JEECOM) 8(1), DOI 10.33650/jeecom.v8i1.16978, on the SAME dataset (UCI 697) and the
SAME split (stratified 80/20, random_state 42 -> 885 test students 284/159/442): Logistic Regression (standard scaling), Random Forest (400 trees, balanced
class weights), XGBoost (500 trees, depth 5, lr 0.1, min_child_weight 3, subsample 1.0, colsample 0.9, gamma 0; no resampling).
Reported by the paper: XGB acc 0.7605 / macro-F1 0.7008; RF 0.7718 / 0.6979; LR 0.7684 / 0.6826; XGB Enrolled recall 0.4654."""
import json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from common import *

_, X, y = load_data()
tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]; print("test class counts", np.bincount(yte))
models = {"Paper LR": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)),
          "Paper RF": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=SEED, n_jobs=N_JOBS)),
          "Paper XGB": XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.1, min_child_weight=3, subsample=1.0, colsample_bytree=0.9, gamma=0, eval_metric="mlogloss", random_state=SEED, n_jobs=N_JOBS)}
rows = []
for n, m in models.items():
    p = m.fit(Xtr, ytr).predict(Xte); rows.append(dict(model=n, **metrics(yte, p)))
opt = json.load(open(RESULTS / "optuna_best.json")); mem = ensemble_members(opt["params"], opt["class_weight"])
Xa, Xb = add_features(Xtr), add_features(Xte); p = soft_vote(mem, Xa, ytr, Xb).argmax(1); rows.append(dict(model="OURS v1 (current ensemble)", **metrics(yte, p)))
d = pd.DataFrame(rows)[["model", "accuracy", "macro_f1", "f1_Dropout", "f1_Enrolled", "f1_Graduate", "recall_Enrolled"]]; print(d.round(4).to_string(index=False))
d.round(5).to_csv(RESULTS / "paper3_replication_holdout.csv", index=False)
