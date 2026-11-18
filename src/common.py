"""Shared utilities for the student dropout / academic success study.

Data: UCI "Predict Students' Dropout and Academic Success" (Realinho et al., 2022), 4,424 students, 36 predictors.
The file is data/data.csv (semicolon separated); run `python src/download_data.py` to fetch it again, or set DATA_PATH.
"""
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]            # encoded 0, 1, 2
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # repository root
DATA_PATH = Path(os.environ.get("DATA_PATH", ROOT / "data" / "data.csv"))
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
RESULTS.mkdir(parents=True, exist_ok=True)
N_JOBS = int(os.environ.get("N_JOBS", 2))

# Continuous columns that receive IQR clipping in the first baseline's pipeline.
CONT_COLS = (["Previous qualification (grade)", "Admission grade", "Age at enrollment"]
             + [f"Curricular units {s} sem ({k})" for s in ("1st", "2nd")
                for k in ("credited", "enrolled", "evaluations", "approved", "grade", "without evaluations")]
             + ["Unemployment rate", "Inflation rate", "GDP"])

# Columns removed by the second baseline (correlation with the target below 0.1).
DROP_PAPER2 = ["Nacionality", "Mother's qualification", "Father's qualification", "Educational special needs",
               "International", "Curricular units 1st sem (without evaluations)", "Unemployment rate", "Inflation rate"]

# Hand-set LightGBM parameters (never tuned on any test data).
BASE_LGB = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                subsample=0.85, colsample_bytree=0.8, reg_lambda=1)


def load_data():
    """Return the raw frame, the 36 predictors and the encoded target (Dropout 0, Enrolled 1, Graduate 2)."""
    df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    y = df["Target"].map({"Dropout": 0, "Enrolled": 1, "Graduate": 2}).to_numpy()
    return df, df.drop(columns="Target"), y


# ----------------------------------------------------------------------------- preprocessing
def iqr_limits(frame):
    """Tukey fences (Q1 - 1.5 IQR, Q3 + 1.5 IQR) for the continuous columns."""
    q1, q3 = frame[CONT_COLS].quantile(.25), frame[CONT_COLS].quantile(.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def iqr_clip(frame, lo, hi):
    out = frame.copy()
    out[CONT_COLS] = out[CONT_COLS].clip(lower=lo, upper=hi, axis=1)
    return out


def top_k_features(Xc, y, k=20, seed=SEED):
    """Top-k features by ExtraTrees impurity importance (first baseline's selection step)."""
    et = ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=N_JOBS).fit(Xc, y)
    imp = pd.Series(et.feature_importances_, index=Xc.columns).sort_values(ascending=False)
    return imp.head(k).index.tolist(), imp


def add_features(X):
    """18 engineered academic-progress features (ratios, deltas, flags)."""
    X = X.copy()
    g = lambda s, k: X[f"Curricular units {s} sem ({k})"]
    a1, e1, v1, g1, c1, w1 = [g("1st", k) for k in ("approved", "enrolled", "evaluations", "grade", "credited", "without evaluations")]
    a2, e2, v2, g2, c2, w2 = [g("2nd", k) for k in ("approved", "enrolled", "evaluations", "grade", "credited", "without evaluations")]
    X["appr_rate_1"] = np.where(e1 > 0, a1 / e1.replace(0, 1), 0)
    X["appr_rate_2"] = np.where(e2 > 0, a2 / e2.replace(0, 1), 0)
    X["appr_total"] = a1 + a2
    X["enrolled_total"] = e1 + e2
    X["appr_rate_total"] = np.where(e1 + e2 > 0, (a1 + a2) / (e1 + e2).replace(0, 1), 0)
    X["eval_succ_1"] = np.where(v1 > 0, a1 / v1.replace(0, 1), 0)
    X["eval_succ_2"] = np.where(v2 > 0, a2 / v2.replace(0, 1), 0)
    X["grade_delta"] = g2 - g1
    X["appr_delta"] = a2 - a1
    X["fail_units_1"] = e1 - a1
    X["fail_units_2"] = e2 - a2
    X["fail_total"] = (e1 - a1) + (e2 - a2)
    X["no_eval_total"] = w1 + w2
    X["credited_total"] = c1 + c2
    X["adm_minus_prev"] = X["Admission grade"] - X["Previous qualification (grade)"]
    X["fees_debt"] = X["Tuition fees up to date"] * 2 + X["Debtor"]
    X["no_sem2_activity"] = ((e1 > 0) & (e2 == 0)).astype(int)
    X["zero_appr_2"] = (a2 == 0).astype(int)
    return X


# ----------------------------------------------------------------------------- models
def class_weights(y, mode):
    """'balanced' = n / (K * n_c); 'sqrt' = its square root; None = uniform."""
    if mode is None:
        return {0: 1.0, 1: 1.0, 2: 1.0}
    n = np.bincount(y, minlength=3).astype(float)
    bal = n.sum() / (3 * n)
    return {i: float(bal[i] if mode == "balanced" else np.sqrt(bal[i])) for i in range(3)}


class Model:
    """Uniform fit / predict_proba wrapper for LightGBM, CatBoost, XGBoost and ExtraTrees with class weights."""

    def __init__(self, kind, params=None, cw=None, seed=SEED):
        self.kind, self.params, self.cw, self.seed = kind, dict(params or {}), cw, seed

    def fit(self, X, y):
        cw = class_weights(y, self.cw)
        if self.kind == "lgbm":
            self.m = LGBMClassifier(class_weight=cw, subsample_freq=1, random_state=self.seed, n_jobs=N_JOBS,
                                    verbose=-1, **self.params).fit(X, y)
        elif self.kind == "cat":
            p = dict(iterations=400, depth=6, learning_rate=0.05); p.update(self.params)
            self.m = CatBoostClassifier(class_weights=[cw[0], cw[1], cw[2]], random_seed=self.seed, verbose=0,
                                        thread_count=N_JOBS, **p).fit(X, y)
        elif self.kind == "xgb":
            p = dict(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.85, colsample_bytree=0.8); p.update(self.params)
            self.m = XGBClassifier(eval_metric="mlogloss", random_state=self.seed, n_jobs=N_JOBS, **p).fit(
                X, y, sample_weight=np.array([cw[i] for i in y]))
        elif self.kind == "et":
            p = dict(n_estimators=500, min_samples_leaf=2, max_features="sqrt"); p.update(self.params)
            self.m = ExtraTreesClassifier(class_weight=cw, random_state=self.seed, n_jobs=N_JOBS, **p).fit(X, y)
        else:
            raise ValueError(self.kind)
        return self

    def predict_proba(self, X):
        return np.asarray(self.m.predict_proba(X))


def ensemble_members(lgb_params=None, lgb_cw="balanced"):
    """The four members of the proposed soft-voting ensemble."""
    return [Model("lgbm", lgb_params or BASE_LGB, lgb_cw),
            Model("cat", dict(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=5), "balanced"),
            Model("xgb", dict(n_estimators=300, max_depth=4, learning_rate=0.05, min_child_weight=3, reg_lambda=5), "balanced"),
            Model("et", {}, "balanced")]


def soft_vote(members, X_train, y_train, X_test):
    """Average the class probabilities of the fitted members."""
    return np.mean([m.fit(X_train, y_train).predict_proba(X_test) for m in members], axis=0)


# ----------------------------------------------------------------------------- metrics and tests
def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    return dict(accuracy=accuracy_score(y_true, y_pred), macro_f1=f.mean(),
                f1_Dropout=f[0], f1_Enrolled=f[1], f1_Graduate=f[2],
                recall_Dropout=r[0], recall_Enrolled=r[1], recall_Graduate=r[2],
                precision_Dropout=p[0], precision_Enrolled=p[1], precision_Graduate=p[2])


def nadeau_bengio(d, n_train, n_test):
    """Corrected resampled t-test (Nadeau & Bengio, 2003) for paired per-fold differences d."""
    from scipy import stats
    d = np.asarray(d, float); J = len(d); var = d.var(ddof=1)
    if J < 2 or var == 0:
        return np.nan, np.nan
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    return t, 2 * stats.t.sf(abs(t), J - 1)


def paper3_all_path():
    """paper3_all.csv (full run) or paper3_all_fast.csv (--fast run)."""
    p = RESULTS / "paper3_all.csv"
    return p if p.exists() else RESULTS / "paper3_all_fast.csv"
