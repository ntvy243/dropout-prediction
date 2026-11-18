"""Student Performance (UCI id 320; Cortez & Silva, 2008): data loading, features and the compared pipelines.

Pipelines (all fitted on the training fold only, then applied to the test fold):
  OURS      : engineered features + soft-voting ensemble (LightGBM, CatBoost, XGBoost, ExtraTrees) with balanced class weights
  ISLAM     : Islam et al. (2025) pipeline: IQR clipping -> Extra-Trees top-20 -> SMOTE -> Min-Max -> XGBoost (default)
  KESGIN_LR / KESGIN_RF / KESGIN_XGB : Kesgin et al. (2025, Applied Sciences 15, 8409) as described in the paper:
      one-hot (drop first), StandardScaler, SMOTE on the training part, L2 logistic regression / RF(300, depth 10, split 5) / XGB(200, depth 5, lr 0.1)
"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA, RESULTS = HERE / "data", HERE / "results"
SEED = 42
N_JOBS = 2
YESNO = ["schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]


def load(subject):
    """subject: 'mat' (395 students, Mathematics) or 'por' (649 students, Portuguese)."""
    return pd.read_csv(DATA / f"student-{subject}.csv", sep=";")


def make_target(g3, kind):
    if kind == "binary":                                   # pass (G3 >= 10) vs fail, as in Kesgin et al. and Cortez & Silva
        return (g3 >= 10).astype(int).values               # 1 = pass, 0 = fail
    if kind == "five":                                     # Cortez & Silva 5-level scale: 0 fail, 1 sufficient, 2 satisfactory, 3 good, 4 excellent
        return np.select([g3 < 10, g3 < 12, g3 < 14, g3 < 16], [0, 1, 2, 3], 4)
    raise ValueError(kind)


def add_features(df, grades=True):
    """18 engineered academic-progress / background features (computed from the raw columns of one row only)."""
    d = df.copy(); yn = lambda c: (d[c] == "yes").astype(int)
    d["parents_edu_mean"] = (d.Medu + d.Fedu) / 2; d["parents_edu_max"] = d[["Medu", "Fedu"]].max(axis=1)
    d["alcohol_total"] = d.Dalc + d.Walc; d["social_index"] = d.goout + d.freetime; d["study_minus_free"] = d.studytime - d.freetime
    d["support_total"] = yn("schoolsup") + yn("famsup") + yn("paid"); d["log_absences"] = np.log1p(d.absences)
    d["failures_x_absences"] = d.failures * (d.absences + 1); d["family_stability"] = d.famrel + (d.Pstatus == "T").astype(int)
    d["ambition_study"] = yn("higher") * d.studytime; d["health_goout"] = d.health * d.goout
    d["travel_study"] = d.traveltime - d.studytime; d["no_failures"] = (d.failures == 0).astype(int)
    if grades:
        d["g_mean"] = (d.G1 + d.G2) / 2; d["g_trend"] = d.G2 - d.G1; d["g_min"] = d[["G1", "G2"]].min(axis=1)
        d["g2_below10"] = (d.G2 < 10).astype(int); d["g2_x_failures"] = d.G2 * (d.failures + 1); d["g2_minus_parents"] = d.G2 - 2 * d.parents_edu_mean
    return d


def prepare(df, grades=True, engineered=True):
    """Numeric design matrix: one-hot encoding of the text columns (+ engineered features when requested)."""
    d = df.drop(columns=["G3"])
    if engineered: d = add_features(d, grades)
    if not grades: d = d.drop(columns=["G1", "G2"])
    return pd.get_dummies(d, dtype=float)


def sw(y):  # balanced sample weights
    return compute_sample_weight("balanced", y)


class Ensemble:
    """OURS: soft voting of four class-weighted models."""
    def __init__(self, seed=SEED, power=1.0): self.seed, self.power = seed, power      # power 1 = balanced class weights, 0.5 = square root (v2), 0 = none
    def fit(self, X, y):
        k = len(np.unique(y)); cw = {c: float((len(y) / (k * (y == c).sum())) ** self.power) for c in np.unique(y)}; w = np.array([cw[i] for i in y])
        self.ms = [
            LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, max_depth=5, min_child_samples=8, subsample=0.85, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=1.0, class_weight=cw, random_state=self.seed, n_jobs=N_JOBS, verbose=-1).fit(X, y),
            CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05, l2_leaf_reg=5, class_weights=[cw[c] for c in sorted(cw)], random_seed=self.seed,
                               verbose=0, thread_count=N_JOBS).fit(X, y),
            XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.85, colsample_bytree=0.7, min_child_weight=2,
                          reg_lambda=3, eval_metric="logloss" if k == 2 else "mlogloss", random_state=self.seed, n_jobs=N_JOBS).fit(X, y, sample_weight=w),
            ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt", class_weight=cw, random_state=self.seed, n_jobs=N_JOBS).fit(X, y)]
        self.classes_ = np.unique(y); return self
    def predict_proba(self, X): return np.mean([m.predict_proba(X) for m in self.ms], axis=0)


def _smote(X, y, seed):
    m = np.bincount(y)[np.bincount(y) > 0].min()
    if m < 2: return X, y
    return SMOTE(random_state=seed, k_neighbors=int(min(5, m - 1))).fit_resample(X, y)


class Islam:
    """ISLAM: IQR clipping -> Extra-Trees top-20 -> SMOTE -> Min-Max -> XGBoost (all fitted on the training fold only)."""
    def __init__(self, seed=SEED): self.seed = seed
    def fit(self, X, y):
        X = X.copy(); q1, q3 = X.quantile(0.25), X.quantile(0.75); iqr = q3 - q1
        self.lo, self.hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr; Xc = X.clip(self.lo, self.hi, axis=1)
        et = ExtraTreesClassifier(n_estimators=200, random_state=self.seed, n_jobs=N_JOBS).fit(Xc, y)
        self.cols = list(Xc.columns[np.argsort(et.feature_importances_)[::-1][:20]])
        Xs, ys = _smote(Xc[self.cols].values, y, self.seed); self.sc = MinMaxScaler().fit(Xs)
        self.m = XGBClassifier(random_state=self.seed, n_jobs=N_JOBS, eval_metric="logloss" if len(np.unique(y)) == 2 else "mlogloss").fit(self.sc.transform(Xs), ys)
        self.classes_ = np.unique(y); return self
    def predict_proba(self, X): return self.m.predict_proba(self.sc.transform(X.clip(self.lo, self.hi, axis=1)[self.cols].values))


class Kesgin:
    """KESGIN et al. (2025) as described: standardise numeric columns, SMOTE on the training part, then LR / RF / XGB with the published settings."""
    def __init__(self, kind, seed=SEED): self.kind, self.seed = kind, seed
    def fit(self, X, y):
        self.num = [c for c in X.columns if X[c].nunique() > 2]; self.sc = StandardScaler().fit(X[self.num])
        Z = self._z(X); Zs, ys = _smote(Z.values, y, self.seed)
        k = len(np.unique(y)); self.classes_ = np.unique(y)
        if self.kind == "LR": self.m = LogisticRegression(penalty="l2", max_iter=2000)
        elif self.kind == "RF": self.m = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, random_state=self.seed, n_jobs=N_JOBS)
        else: self.m = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=self.seed, n_jobs=N_JOBS, eval_metric="logloss" if k == 2 else "mlogloss")
        self.m.fit(Zs, ys); return self
    def _z(self, X): Z = X.copy(); Z[self.num] = self.sc.transform(X[self.num]); return Z
    def predict_proba(self, X): return self.m.predict_proba(self._z(X).values)


def metrics(y, p):
    """p = predicted class probabilities (n, K)."""
    pred = p.argmax(1); K = p.shape[1]
    pr, rc, f, _ = precision_recall_fscore_support(y, pred, labels=list(range(K)), zero_division=0)
    out = dict(accuracy=accuracy_score(y, pred), balanced_acc=balanced_accuracy_score(y, pred), macro_f1=f.mean())
    if K == 2: out.update(f1_fail=f[0], recall_fail=rc[0], precision_fail=pr[0], auc=roc_auc_score(y, p[:, 1]))
    return out


def nadeau_bengio(d, n_train, n_test):
    d = np.asarray(d, float); J = len(d); var = d.var(ddof=1)
    if J < 2 or var == 0: return np.nan, np.nan
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var); return t, 2 * stats.t.sf(abs(t), J - 1)
