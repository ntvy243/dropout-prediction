"""Publication figures from the CSV/JSON files written by run_protocol_{a,b,c}.py.
Output: ../figures/fig*.png (300 dpi) and .pdf   (usage: python make_figures.py)"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np, pandas as pd
import shap
from common import *

FIG = RESULTS.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OURS, BASE1, BASE2, GREY = "#2a78d6", "#eb6834", "#eda100", "#8a8985"
CLS = {"Dropout": "#eb6834", "Enrolled": "#eda100", "Graduate": "#2a78d6"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "axes.titlesize": 10, "axes.titleweight": "bold"})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig); print("saved", name)


def tag(ax, letter):
    ax.text(-0.1, 1.07, f"({letter})", transform=ax.transAxes, fontsize=10, fontweight="bold")


df, X, y = load_data()
A_seeds = pd.read_csv(RESULTS / "protocolA_seeds.csv"); A_sum = pd.read_csv(RESULTS / "protocolA_summary.csv").set_index("model")
B_split = pd.read_csv(RESULTS / "protocolB_splits.csv")
C_hold = pd.read_csv(RESULTS / "protocolC_holdout.csv").set_index("model")
C_cv = pd.read_csv(RESULTS / "protocolC_cv_folds.csv")
REF, ENS = "XGBoost paper default", "ENSEMBLE top-3 (ours)"

# ------------------------------------------------------------------ Fig 1: workflow
fig, ax = plt.subplots(figsize=(11, 3.7)); ax.axis("off"); ax.set_xlim(0, 11.2); ax.set_ylim(0, 3.7)
def box(x, y0, w, txt, col):
    ax.add_patch(FancyBboxPatch((x, y0), w, 1.0, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=col, lw=1.8))
    ax.text(x + w / 2, y0 + 0.5, txt, ha="center", va="center", fontsize=7.6)
box(0.1, 1.35, 1.6, "UCI data\n4,424 students\n36 predictors", GREY)
box(1.95, 1.35, 1.7, "Pre-processing\nIQR clipping, feature\nselection, balancing,\nengineered features", GREY)
box(4.0, 2.55, 1.9, "Protocol A\nSMOTE before split\n(baseline 1)", BASE1)
box(4.0, 1.35, 1.9, "Protocol B\nrandom split, 28 columns\n(baseline 2)", BASE2)
box(4.0, 0.15, 1.9, "Protocol C\nsplit first, train-only\nfitting (leakage-free)", OURS)
box(6.25, 1.35, 1.7, "Models\nbaselines vs proposed\n4-model soft-voting\nensemble", GREY)
box(8.2, 1.35, 1.7, "Evaluation\naccuracy, macro-F1,\nEnrolled recall,\nrepeated splits + tests", GREY)
box(10.1, 1.35, 1.0, "SHAP and\ndecision\nsupport", GREY)
arrow = dict(arrowstyle="->", color="#444", lw=1.2)
for p0, p1 in [((1.7, 1.85), (1.95, 1.85)), ((3.65, 1.85), (4.0, 3.05)), ((3.65, 1.85), (4.0, 1.85)), ((3.65, 1.85), (4.0, 0.65)),
               ((5.9, 3.05), (6.25, 2.15)), ((5.9, 1.85), (6.25, 1.85)), ((5.9, 0.65), (6.25, 1.55)), ((7.95, 1.85), (8.2, 1.85)), ((9.9, 1.85), (10.1, 1.85))]:
    ax.annotate("", xy=p1, xytext=p0, arrowprops=arrow)
save(fig, "fig1_workflow")

# ------------------------------------------------------------------ Fig 2: class distribution and outcome by student characteristic
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=[1, 2.3]))
cnt = df["Target"].value_counts()[CLASS_NAMES]
ax = axes[0]; bars = ax.bar(CLASS_NAMES, cnt.values, color=[CLS[c] for c in CLASS_NAMES])
for b, v in zip(bars, cnt.values): ax.text(b.get_x() + b.get_width() / 2, v + 30, f"{v:,}\n({v / len(df):.1%})", ha="center", fontsize=8)
ax.set_ylim(0, 2700); ax.set_ylabel("students"); ax.set_title("Outcome distribution"); tag(ax, "a")
ax = axes[1]
spec = [("Tuition fees up to date", {0: "Fees not\nup to date", 1: "Fees up\nto date"}), ("Scholarship holder", {0: "No\nscholarship", 1: "Scholarship"}),
        ("Debtor", {0: "Not debtor", 1: "Debtor"}), ("Gender", {0: "Female", 1: "Male"}), ("Displaced", {0: "Not\ndisplaced", 1: "Displaced"})]
xs, labels, pos = [], [], 0
for var, lab in spec:
    for v, name in lab.items():
        share = df[df[var] == v]["Target"].value_counts(normalize=True).reindex(CLASS_NAMES).fillna(0).values
        left = 0
        for c, s in zip(CLASS_NAMES, share):
            ax.barh(pos, s, left=left, color=CLS[c], edgecolor="white", linewidth=1.2, label=c if pos == 0 else None)
            if s > 0.07: ax.text(left + s / 2, pos, f"{s:.0%}", ha="center", va="center", fontsize=7, color="white" if c != "Enrolled" else "black")
            left += s
        labels.append(name.replace("\n", " ")); xs.append(pos); pos += 1
    pos += 0.5
ax.set_yticks(xs); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 1); ax.grid(False)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); ax.set_title("Outcome by student characteristic")
ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.17), fontsize=8); tag(ax, "b")
fig.tight_layout(); save(fig, "fig2_eda")

# ------------------------------------------------------------------ Fig 3: Protocol A
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=[1.5, 1]))
ax = axes[0]; pw = A_seeds.pivot(index="seed", columns="model", values="accuracy")
ax.plot(pw.index, pw[REF], "o-", color=BASE1, label="Baseline 1 XGBoost (re-run)"); ax.plot(pw.index, pw[ENS], "o-", color=OURS, label="Proposed ensemble")
ax.axhline(0.83, color="#333", ls="--", lw=1); ax.text(pw.index.min() - 0.4, 0.8315, "reported 0.83", ha="left", fontsize=8)
ax.set_xticks(list(pw.index)); ax.set_xlabel("random seed"); ax.set_ylabel("test accuracy"); ax.set_title("Per-seed accuracy, Protocol A"); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2); tag(ax, "a")
ax = axes[1]; order = [REF] + [m for m in A_sum.index if m not in (REF, ENS)] + [ENS]
m_, s_ = A_sum.loc[order, "mean_accuracy"], A_sum.loc[order, "std_accuracy"]
ax.barh(range(len(order)), m_, xerr=s_, color=[BASE1] + [GREY] * (len(order) - 2) + [OURS], capsize=3, error_kw=dict(lw=1))
for i, v in enumerate(m_): ax.text(0.8, i, f"{v:.3f}", va="center", fontsize=8, color="white")
ax.set_yticks(range(len(order))); ax.set_yticklabels([o.replace("ENSEMBLE top-3 (ours)", "Proposed ensemble").replace("XGBoost paper default", "Baseline 1 XGBoost") for o in order], fontsize=8)
ax.invert_yaxis(); ax.set_xlim(0.78, 0.88); ax.axvline(0.83, color="#333", ls="--", lw=1); ax.set_xlabel("mean accuracy +- SD"); ax.set_title("Mean over seeds"); tag(ax, "b")
fig.tight_layout(); save(fig, "fig3_protocolA")

# ------------------------------------------------------------------ Fig 4: Protocol B
models = ["Paper 2 Random Forest", "Paper 2 SVM", "Paper 2 AdaBoost", "Ours: ensemble (+engineered features)"]
short = ["RF\n(baseline 2)", "SVM\n(baseline 2)", "AdaBoost\n(baseline 2)", "Proposed\nensemble"]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, metric, ttl, l in zip(axes, ["accuracy", "macro_f1", "f1_Enrolled"], ["Accuracy", "Macro-F1", "Enrolled F1"], "abc"):
    data = [B_split[B_split.model == m][metric].values for m in models]
    bp = ax.boxplot(data, widths=.55, patch_artist=True, showmeans=True, medianprops=dict(color="black"),
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=4))
    for patch, c in zip(bp["boxes"], [BASE2, GREY, GREY, OURS]): patch.set_facecolor(c); patch.set_alpha(.6)
    ax.set_xticks(range(1, 5)); ax.set_xticklabels(short, fontsize=7.5); ax.set_title(f"{ttl} (n={len(data[0])} splits)"); tag(ax, l)
    if metric == "accuracy": ax.axhline(0.8057, color="#333", ls="--", lw=1); ax.text(0.55, 0.8075, "reported 0.8057", fontsize=7.5)
fig.tight_layout(); save(fig, "fig4_protocolB")

# ------------------------------------------------------------------ Fig 5: leakage inflation (same model family, Protocol A vs C)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
mets = [("accuracy", "mean_accuracy", "Accuracy"), ("macro_f1", "mean_macro_f1", "Macro-F1"), ("f1_Enrolled", "mean_f1_Enrolled", "Enrolled F1")]
a_vals = [A_sum.loc[ENS, c] for _, c, _ in mets]; c_vals = [C_hold.loc["Proposed 4-model ensemble", k] for k, _, _ in mets]
x = np.arange(3); w = .36
b1 = ax.bar(x - w / 2, a_vals, w, color=BASE1, label="Protocol A: SMOTE before split (leaky)"); b2 = ax.bar(x + w / 2, c_vals, w, color=OURS, label="Protocol C: split first (fair)")
for b, v in list(zip(b1, a_vals)) + list(zip(b2, c_vals)): ax.text(b.get_x() + b.get_width() / 2, v + .01, f"{v:.3f}", ha="center", fontsize=8)
for i, (a, c) in enumerate(zip(a_vals, c_vals)): ax.text(i, 0.95, f"leakage inflation +{a - c:.3f}", ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([m[2] for m in mets]); ax.set_ylim(0, 1.05); ax.set_ylabel("score"); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.1))
ax.set_title("Same model family, two evaluation protocols"); fig.tight_layout(); save(fig, "fig5_leakage")

# ------------------------------------------------------------------ Fig 6: SHAP importance (per class) and beeswarm for Dropout
imp = pd.read_csv(RESULTS / "shap_importance.csv").head(12).iloc[::-1]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1, 1.1]))
ax = axes[0]; left = np.zeros(len(imp))
for c in CLASS_NAMES:
    ax.barh(imp.feature, imp[f"shap_{c}"], left=left, color=CLS[c], label=c, edgecolor="white", linewidth=1); left += imp[f"shap_{c}"].values
ax.set_xlabel("mean |SHAP value| (hold-out students)"); ax.set_title("Global importance by outcome class"); ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.tick_params(axis="y", labelsize=8); tag(ax, "a")
sv = np.load(RESULTS / "shap_values.npy"); Xte = pd.read_csv(RESULTS / "shap_holdout_features.csv")
plt.sca(axes[1]); shap.summary_plot(sv[0], Xte, max_display=12, show=False, plot_size=None, color_bar=True)
axes[1].set_title("SHAP values for the Dropout class"); axes[1].tick_params(axis="y", labelsize=8); axes[1].grid(False); tag(axes[1], "b")
fig.tight_layout(); save(fig, "fig6_shap")

# ------------------------------------------------------------------ Fig 7: confusion matrices
cm = json.load(open(RESULTS / "confusion_matrices.json"))
panels = [("paper1_fig7_xgb_reported", "Baseline 1, reported (Fig. 7)\nSMOTE-balanced test set"), ("paper_method_fair", "Baseline 1 method, leakage-free\nreal test set"), ("proposed_ensemble", "Proposed ensemble, leakage-free\nreal test set")]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
for ax, (k, ttl), l in zip(axes, panels, "abc"):
    m = np.array(cm[k]); pc = m / m.sum(1, keepdims=True); ax.imshow(pc, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    for i in range(3):
        for j in range(3): ax.text(j, i, f"{pc[i, j]:.0%}\n({m[i, j]})", ha="center", va="center", fontsize=8, color="white" if pc[i, j] > .5 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_NAMES, fontsize=8); ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{ttl}\nn = {m.sum()}, Enrolled recall {m[1, 1] / m[1].sum():.0%}", fontsize=8.5); tag(ax, l)
fig.tight_layout(); save(fig, "fig7_confusion")

# ------------------------------------------------------------------ Fig 8: repeated 5x2 CV
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4)); names = ["Paper method: top-20 + SMOTE + XGBoost", "Proposed 4-model ensemble"]
for ax, metric, ttl, l in zip(axes, ["accuracy", "macro_f1"], ["Accuracy per fold", "Macro-F1 per fold"], "ab"):
    data = [C_cv[C_cv.model == n][metric].values for n in names]
    bp = ax.boxplot(data, widths=.5, patch_artist=True, showmeans=True, medianprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], [BASE1, OURS]): patch.set_facecolor(c); patch.set_alpha(.6)
    rng = np.random.RandomState(0)
    for i, d in enumerate(data): ax.scatter(rng.normal(i + 1, .04, len(d)), d, s=14, color=[BASE1, OURS][i], zorder=3)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Baseline 1\nmethod", "Proposed\nensemble"]); ax.set_title(ttl); tag(ax, l)
fig.tight_layout(); save(fig, "fig8_cv")

# ------------------------------------------------------------------ Fig 9: fairness
fa = pd.read_csv(RESULTS / "fairness.csv"); fa["label"] = fa.group + " (n=" + fa.n.astype(str) + ")"
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
for ax, col, ttl, l in zip(axes, ["accuracy", "recall_Dropout", "recall_Enrolled"], ["Accuracy", "Dropout recall", "Enrolled recall"], "abc"):
    ax.barh(range(len(fa)), fa[col], color=[OURS if i % 2 == 0 else GREY for i in range(len(fa))])
    for i, v in enumerate(fa[col]): ax.text(v + .01, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1); ax.set_title(ttl); ax.set_yticks(range(len(fa))); ax.set_yticklabels(fa.label, fontsize=8); ax.invert_yaxis(); tag(ax, l)
fig.tight_layout(); save(fig, "fig9_fairness")
