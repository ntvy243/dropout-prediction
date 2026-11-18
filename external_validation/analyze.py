"""Summaries, corrected resampled t-tests (Nadeau & Bengio) and figures from results/cv_folds.csv, split_paper_text.csv and split_repo_code.csv."""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sp_common import *

src = "cv_folds_fast.csv" if "--fast" in sys.argv else "cv_folds.csv"
d = pd.read_csv(RESULTS / src); OURS = "OURS (ensemble + engineered features)"
keys = ["subject", "grades", "target"]
summ = d.groupby(keys + ["model"]).agg(accuracy=("accuracy", "mean"), accuracy_sd=("accuracy", "std"), balanced_acc=("balanced_acc", "mean"), macro_f1=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"), f1_fail=("f1_fail", "mean"), auc=("auc", "mean"), folds=("fold", "count")).reset_index()
summ.round(4).to_csv(RESULTS / "cv_summary.csv", index=False)
tests = []
for k, g in d.groupby(keys):
    a = g[g.model == OURS].sort_values("fold")
    for m in g.model.unique():
        if m == OURS: continue
        b = g[g.model == m].sort_values("fold")
        for metric in ["accuracy", "macro_f1"] + (["f1_fail"] if k[2] == "binary" else []):
            diff = a[metric].values - b[metric].values; t, p = nadeau_bengio(diff, a.n_train.iloc[0], a.n_test.iloc[0])
            tests.append(dict(zip(keys, k), baseline=m, metric=metric, ours=a[metric].mean(), baseline_mean=b[metric].mean(), mean_diff=diff.mean(), corrected_t=t,
                              corrected_p=p, naive_p=stats.ttest_1samp(diff, 0)[1], wins=int((diff > 0).sum()), folds=len(diff)))
T = pd.DataFrame(tests); T.round(5).to_csv(RESULTS / "cv_tests.csv", index=False)
print(summ[summ.target == "binary"].round(3).to_string())

# ---- figure: accuracy by setting, binary target
col = {OURS: "#2A78D6", "OURS without engineered features": "#A9C8EE", "Islam et al. (2025) pipeline": "#EB6834", "Kesgin et al. (2025) Logistic Regression": "#8A8985",
       "Kesgin et al. (2025) Random Forest": "#B5B3AD", "Kesgin et al. (2025) XGBoost": "#5F6368"}
for tgt in ["binary", "five"]:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=False)
    for ax, (k, g) in zip(axes, d[d.target == tgt].groupby(["subject", "grades"])):
        s = g.groupby("model").accuracy.agg(["mean", "std"]).reindex(list(col)); ax.bar(range(len(s)), s["mean"], yerr=s["std"], color=[col[m] for m in s.index], capsize=2)
        lo = s["mean"].min() - 0.08; ax.set_ylim(max(0, lo), min(1, s["mean"].max() + 0.06)); ax.set_xticks([])
        ax.set_title(f"{'Mathematics' if k[0] == 'mat' else 'Portuguese'}, {k[1].replace('G1+G2 ', 'G1/G2 ')}", fontsize=10)
        for i, v in enumerate(s["mean"]): ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=7.5)
    axes[0].set_ylabel("accuracy (15 folds)"); fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in col.values()], [m.replace(" (ensemble + engineered features)", " (proposed)") for m in col], loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"UCI Student Performance, {'pass/fail' if tgt == 'binary' else '5-level grade'}: repeated stratified 5-fold CV", fontsize=11); fig.tight_layout(rect=[0, 0.12, 1, 0.94])
    for ext in ("png", "pdf"): fig.savefig(RESULTS / f"fig_cv_{tgt}.{ext}", dpi=200)
    plt.close(fig)
try:
    s1 = pd.read_csv(RESULTS / "split_paper_text.csv"); print(s1.groupby("model")[["accuracy", "f1_fail", "macro_f1"]].agg(["mean", "std"]).round(3))
    print(pd.read_csv(RESULTS / "split_repo_code.csv").round(3))
except FileNotFoundError: pass
