"""All figures of the manuscript, generated from the result files (300 dpi PNG + vector PDF). Okabe-Ito colour-blind-safe palette; thin marks, light grid."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from common import *
OUT = RESULTS.parent / "paper_figures"; OUT.mkdir(exist_ok=True)
C = {"islam": "#D55E00", "islam2": "#999999", "v2": "#0072B2", "v1": "#56B4E9", "grid": "#E6E6E6", "ink": "#222222"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#888", "axes.linewidth": .8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": .6, "axes.axisbelow": True, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})
def save(fig, name): fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)
df, X, y = load_data(); isl = pd.read_csv(RESULTS / "islam_fair_all.csv"); p3 = pd.read_csv(paper3_all_path()); p3 = p3[p3.model.str.startswith("OURS")]
abl = pd.read_csv(RESULTS / "ablation_all.csv"); allr = pd.concat([isl, p3, abl]); allr["model"] = allr.model.replace({"OURS v1 (balanced weights)": "Proposed v1", "OURS v2 (sqrt weights)": "Proposed v2"})
A = pd.read_csv(RESULTS / "protocolA_seeds.csv"); Av2 = pd.read_csv(RESULTS / "protocolA_v2_seeds.csv")
cv = allr[allr.kind == "cv"]; ho = allr[allr.kind == "holdout"]
def ms(d, m, k): s = d[d.model == m][k]; return s.mean(), s.std(ddof=1)

# ---- Fig 1 workflow
fig, ax = plt.subplots(figsize=(7.6, 3.5)); ax.axis("off"); ax.set_xlim(0, 101); ax.set_ylim(0, 48)
def box(x, y_, w, h, t, fc, ec="#555", fs=7.2): ax.add_patch(FancyBboxPatch((x, y_), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=.9)); ax.text(x + w / 2, y_ + h / 2, t, ha="center", va="center", fontsize=fs, color=C["ink"], linespacing=1.25)
def arr(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=.9, color="#555"))
box(0.5, 18, 14, 12, "UCI dataset\n4,424 students\n36 predictors", "#F2F2F2"); arr(15.2, 24, 18.3, 24)
box(18.5, 18, 17, 12, "Stratified split\ninto train / test\nbefore any\nfitted step", "#FFF2CC"); arr(36, 24, 40.5, 34); arr(36, 24, 40.5, 14)
box(40.5, 28.5, 27, 12, "Baseline (Islam et al.)\nIQR clip → top-20 features\n→ SMOTE → Min-Max → XGBoost", "#FBE5D6", C["islam"], fs=6.6); box(40.5, 7.5, 27, 12, "Proposed\n36 raw + 18 engineered\nfeatures → 4-model cost-\nsensitive soft-voting", "#DDEBF7", C["v2"])
arr(68, 34.5, 72.5, 25.5); arr(68, 13.5, 72.5, 22.5)
box(72.5, 17, 17, 14, "Evaluation\n15-fold CV,\n11 hold-outs,\nper-class metrics", "#E2F0D9"); arr(90.2, 24, 92.2, 24)
box(92.2, 18.5, 8, 11, "SHAP,\nsub-\ngroups", "#EDEDED", fs=6.6)
ax.text(0.5, 46, "Protocol A (baseline's own): SMOTE and scaling on all data, then split; used for the like-for-like comparison only", fontsize=7, color=C["islam"], va="top")
ax.text(0.5, 1.5, "Protocol C (leakage-free): clipping limits, feature selection, SMOTE and scaling are fitted on the training part only", fontsize=7, color=C["v2"])
save(fig, "fig1_workflow")

# ---- Fig 2 EDA
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.7)); n = y.size
cnt = np.bincount(y); axs[0].bar(CLASS_NAMES, cnt, color=[C["islam"], C["islam2"], C["v2"]], width=.6)
for i, c in enumerate(cnt): axs[0].text(i, c + 30, f"{c}\n({c/n:.1%})", ha="center", fontsize=7.5)
axs[0].set_title("(a) Class distribution"); axs[0].set_ylim(0, 2700); axs[0].set_ylabel("Students")
a2 = "Curricular units 2nd sem (approved)"
for k, nm in enumerate(CLASS_NAMES): axs[1].boxplot(X[a2][y == k], positions=[k], widths=.5, patch_artist=True, boxprops=dict(facecolor=[C["islam"], C["islam2"], C["v2"]][k], alpha=.55), medianprops=dict(color="k"), flierprops=dict(markersize=2))
axs[1].set_xticks(range(3)); axs[1].set_xticklabels(CLASS_NAMES); axs[1].set_title("(b) 2nd-semester units approved"); axs[1].set_ylabel("Units approved")
rates = []
for col, labs in [("Tuition fees up to date", ("Fees\nunpaid", "Fees\npaid")), ("Scholarship holder", ("No\nschol.", "Schol.\nholder")), ("Debtor", ("Not\ndebtor", "Debtor"))]:
    for v in (0, 1): m = X[col] == v; rates.append((labs[v], (y[m] == 0).mean()))
axs[2].bar([r[0] for r in rates], [r[1] for r in rates], color=C["islam"], width=.62, alpha=.85); axs[2].set_title("(c) Dropout rate by status"); axs[2].set_ylabel("Share who dropped out"); axs[2].tick_params(axis="x", labelsize=6.2); axs[2].set_ylim(0, 1)
for i, r in enumerate(rates): axs[2].text(i, r[1] + .02, f"{r[1]:.2f}", ha="center", fontsize=7)
fig.tight_layout(); save(fig, "fig2_eda")

# ---- Fig 3 headline: direct comparison
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.1), gridspec_kw=dict(width_ratios=[1, 1.25]))
sa = A.groupby("model").accuracy.agg(["mean", "std"]); top3 = "ENSEMBLE top-3 (ours)"
vals = [("Reported by\nIslam et al.", .83, 0, C["islam2"]), ("Islam XGB\nre-run", sa.loc["XGBoost paper default", "mean"], sa.loc["XGBoost paper default", "std"], C["islam"]), ("Proposed\n(top-3)", sa.loc[top3, "mean"], sa.loc[top3, "std"], C["v2"])]
for i, (l, m, s, c) in enumerate(vals): axs[0].bar(i, m, yerr=s if s else None, color=c, width=.6, capsize=3, error_kw=dict(lw=.9)); axs[0].text(i, m + (s or 0) + .004, f"{m:.3f}", ha="center", fontsize=8, fontweight="bold")
axs[0].set_xticks(range(3)); axs[0].set_xticklabels([v[0] for v in vals], fontsize=7.5); axs[0].set_ylim(.7, .9); axs[0].set_ylabel("Accuracy"); axs[0].set_title("(a) Protocol A: baseline's own protocol")
mods = [("Islam XGB", "Islam XGB (fair)", C["islam"]), ("Islam RF", "Islam RF (fair)", C["islam2"]), ("Proposed v1", "Proposed v1", C["v1"]), ("Proposed v2", "Proposed v2", C["v2"])]
for i, (l, m, c) in enumerate(mods):
    mu, sd = ms(cv, m, "accuracy"); axs[1].bar(i, mu, yerr=sd, color=c, width=.6, capsize=3, error_kw=dict(lw=.9)); axs[1].text(i, mu + sd + .003, f"{mu:.3f}", ha="center", fontsize=8, fontweight="bold")
axs[1].set_xticks(range(4)); axs[1].set_xticklabels([m[0] for m in mods], fontsize=7.5); axs[1].set_ylim(.7, .82); axs[1].set_title("(b) Protocol C: leakage-free, 15-fold CV"); axs[1].set_ylabel("Accuracy")
fig.tight_layout(); save(fig, "fig3_headline")

# ---- Fig 4 protocol A per seed
pw = A.pivot(index="seed", columns="model", values="accuracy"); fig, ax = plt.subplots(figsize=(5.6, 2.9))
ax.plot(pw.index, pw["XGBoost paper default"], "o-", color=C["islam"], lw=1.2, ms=4, label="Islam et al. XGBoost (re-run)"); ax.plot(pw.index, pw[top3], "s-", color=C["v2"], lw=1.4, ms=4, label="Proposed ensemble (top-3)")
ax.axhline(.83, color="#777", ls="--", lw=1, label="0.83 reported by Islam et al."); ax.set_xlabel("Seed (new SMOTE sample and split)"); ax.set_ylabel("Accuracy"); ax.set_xticks(range(10)); ax.legend(fontsize=7.5, loc="upper right", ncol=1); ax.set_ylim(.81, .895); ax.set_title("Protocol A: proposed model above the baseline on all 10 seeds")
fig.tight_layout(); save(fig, "fig4_protocolA_seeds")

# ---- Fig 5 leakage effect (same model: Islam XGBoost)
fig, ax = plt.subplots(figsize=(5.8, 2.9)); mets = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("f1_Enrolled", "Enrolled F1")]
xa = A[A.model == "XGBoost paper default"]; xf = ho[ho.model == "Islam XGB (fair)"]
for j, (k, lab) in enumerate(mets):
    a_, f_ = xa[k].mean(), xf[k].mean(); ax.bar(j - .18, a_, .34, color=C["islam"], alpha=.45, hatch="//", edgecolor=C["islam"]); ax.bar(j + .18, f_, .34, color=C["islam"])
    ax.text(j - .18, a_ + .01, f"{a_:.3f}", ha="center", fontsize=7.5); ax.text(j + .18, f_ + .01, f"{f_:.3f}", ha="center", fontsize=7.5); ax.text(j, max(a_, f_) + .10, f"drop {a_-f_:.3f}", ha="center", fontsize=8, fontweight="bold", color="#444")
ax.set_xticks(range(3)); ax.set_xticklabels([m[1] for m in mets]); ax.set_ylim(0, 1.2); ax.set_ylabel("Score"); ax.set_title("Islam XGBoost: SMOTE before vs. after the split")
from matplotlib.patches import Patch; ax.legend(handles=[Patch(fc=C["islam"], alpha=.45, hatch="//", ec=C["islam"], label="Protocol A (leaky)"), Patch(fc=C["islam"], label="Protocol C (leakage-free, 11 splits)")], fontsize=7.5, loc="upper center", bbox_to_anchor=(.5, -0.12), ncol=2)
fig.tight_layout(); save(fig, "fig5_leakage")

# ---- Fig 6 fair metrics (CV)
fig, axs = plt.subplots(1, 3, figsize=(7.8, 2.9)); mm = [("accuracy", "Accuracy", (.70, .82)), ("macro_f1", "Macro-F1", (.60, .76)), ("recall_Enrolled", "Enrolled recall", (.30, .68))]
for ax, (k, lab, yl) in zip(axs, mm):
    for i, (l, m, c) in enumerate(mods): mu, sd = ms(cv, m, k); ax.bar(i, mu, yerr=sd, color=c, width=.62, capsize=2.5, error_kw=dict(lw=.8)); ax.text(i, mu + sd + (yl[1] - yl[0]) * .02, f"{mu:.3f}", ha="center", fontsize=7)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Islam\nXGB", "Islam\nRF", "Prop.\nv1", "Prop.\nv2"], fontsize=7); ax.set_ylim(*yl); ax.set_title(lab)
fig.suptitle("Leakage-free comparison, repeated stratified 5-fold CV × 3 (15 folds; mean ± SD)", fontsize=9, y=1.02); fig.tight_layout(); save(fig, "fig6_fair_metrics")

# ---- Fig 7 paired per-fold differences
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=False)
for ax, (k, lab) in zip(axs, [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")]):
    b = cv[cv.model == "Islam XGB (fair)"].set_index("split")[k]
    for i, (m, c, l) in enumerate([("Proposed v1", C["v1"], "v1 − Islam XGB"), ("Proposed v2", C["v2"], "v2 − Islam XGB")]):
        d = cv[cv.model == m].set_index("split")[k] - b; ax.scatter(np.full(len(d), i) + np.random.default_rng(1).uniform(-.12, .12, len(d)), d, s=14, color=c, alpha=.8, zorder=3); ax.hlines(d.mean(), i - .25, i + .25, color="k", lw=1.6, zorder=4)
        ax.text(i, d.max() + .004, f"mean {d.mean():+.3f}\nwins {int((d > 0).sum())}/15", ha="center", fontsize=7, va="bottom")
    ax.axhline(0, color="#555", lw=.9); ax.set_xticks([0, 1]); ax.set_xticklabels(["Proposed v1", "Proposed v2"]); ax.set_xlim(-.6, 1.6); ax.set_title(f"{lab}: difference per fold"); ax.set_ylabel("Proposed − Islam XGBoost"); ax.margins(y=.25)
fig.tight_layout(); save(fig, "fig7_paired_folds")

# ---- Fig 8 confusion matrices (seed-42 hold-out)
cm = json.load(open(RESULTS / "holdout42_confusion.json")); fig, axs = plt.subplots(1, 3, figsize=(7.8, 2.8))
for ax, (k, lab) in zip(axs, [("Islam XGB (fair)", "Islam XGBoost"), ("Proposed v1", "Proposed v1"), ("Proposed v2", "Proposed v2")]):
    m = np.array(cm[k]); r = m / m.sum(1, keepdims=True); ax.imshow(r, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    for i in range(3):
        for j in range(3): ax.text(j, i, f"{m[i, j]}\n({r[i, j]:.0%})", ha="center", va="center", fontsize=7.5, color="white" if r[i, j] > .55 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(["Drop.", "Enrol.", "Grad."], fontsize=7.5); ax.set_yticks(range(3)); ax.set_yticklabels(["Dropout", "Enrolled", "Graduate"], fontsize=7.5); ax.set_xlabel("Predicted"); ax.set_title(f"{lab}\nacc. {np.trace(m)/m.sum():.3f}", fontsize=8.5)
axs[0].set_ylabel("True"); fig.tight_layout(); save(fig, "fig8_confusion")

# ---- Fig 9 ablation
ab_models = [("Islam XGB (fair)", "Islam\nXGB", C["islam"]), ("v2 single ExtraTrees", "ET", "#BBBBBB"), ("v2 single XGBoost", "XGB", "#BBBBBB"), ("v2 single CatBoost", "Cat", "#BBBBBB"), ("v2 single LightGBM", "LGBM", "#BBBBBB"),
             ("v2, 36 raw features only", "Ens.\nraw 36", C["v1"]), ("Proposed v2", "Ens. v2\n+18 feat.", C["v2"])]
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.0))
for ax, (k, lab, yl) in zip(axs, [("accuracy", "Accuracy", (.72, .815)), ("macro_f1", "Macro-F1", (.64, .765))]):
    for i, (m, l, c) in enumerate(ab_models): mu, sd = ms(cv, m, k); ax.bar(i, mu, yerr=sd, color=c, width=.65, capsize=2, error_kw=dict(lw=.7)); ax.text(i, mu + sd + .002, f"{mu:.3f}", ha="center", fontsize=6.5, rotation=90, va="bottom")
    ax.set_xticks(range(len(ab_models))); ax.set_xticklabels([m[1] for m in ab_models], fontsize=7); ax.set_ylim(*yl); ax.set_title(f"{lab} (15-fold CV); grey = single models (v2 weights)", fontsize=8)
fig.tight_layout(); save(fig, "fig9_ablation")

# ---- Fig 10 SHAP importance (stacked by class) + beeswarm-like dependence for the top feature
imp = pd.read_csv(RESULTS / "shap_v2_importance.csv", index_col=0).head(12)[::-1]; fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.5), gridspec_kw=dict(width_ratios=[1.35, 1]))
left = np.zeros(len(imp))
for cl, col in zip(CLASS_NAMES, [C["islam"], C["islam2"], C["v2"]]): axs[0].barh(imp.index, imp[cl], left=left, color=col, height=.68, label=cl); left += imp[cl].values
axs[0].set_xlabel("Mean |SHAP| (log-odds), summed over classes"); axs[0].tick_params(axis="y", labelsize=6.8); axs[0].legend(fontsize=7, loc="lower right"); axs[0].set_title("(a) Global importance by class")
sv = np.load(RESULTS / "shap_v2_values.npy"); Xh = pd.read_csv(RESULTS / "shap_v2_features.csv"); f = "Curricular units 2nd sem (approved)"; j = list(Xh.columns).index(f)
sc_ = axs[1].scatter(Xh[f], sv[:, j, 0], c=Xh["Tuition fees up to date"], cmap="coolwarm_r", s=8, alpha=.75); axs[1].axhline(0, color="#555", lw=.8); axs[1].set_xlabel("2nd-semester units approved"); axs[1].set_ylabel("SHAP value, class Dropout"); axs[1].set_title("(b) Units approved vs. SHAP (colour: fees paid)")
cb = fig.colorbar(sc_, ax=axs[1], fraction=.05, pad=.02); cb.set_ticks([0, 1]); cb.ax.tick_params(labelsize=7); fig.tight_layout(); save(fig, "fig10_shap")

# ---- Fig 11 subgroups
fa = pd.read_csv(RESULTS / "fairness_v2.csv"); fig, ax = plt.subplots(figsize=(6.2, 2.9)); lab = [f"{r.group}\n(n={r.n})" for r in fa.itertuples()]; xs = np.arange(len(fa))
ax.bar(xs - .27, fa.accuracy, .26, color=C["v2"], label="Accuracy", yerr=[fa.accuracy - fa.acc_lo, fa.acc_hi - fa.accuracy], capsize=2, error_kw=dict(lw=.7)); ax.bar(xs, fa.recall_Dropout, .26, color=C["islam"], label="Dropout recall"); ax.bar(xs + .27, fa.recall_Enrolled, .26, color=C["islam2"], label="Enrolled recall")
ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=7); ax.set_ylim(0, 1.0); ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.0)); ax.set_title("Hold-out performance of the proposed model by group (accuracy: 95% bootstrap CI)", fontsize=8.5, pad=14)
fig.tight_layout(); save(fig, "fig11_subgroups"); print("figures done")
