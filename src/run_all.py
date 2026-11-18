"""Run the whole study in order.

    python src/run_all.py            full run (several hours on 2 CPU cores; Optuna 80 trials, 30 splits, 15 CV folds)
    python src/run_all.py --fast     reduced settings, quick smoke test (numbers will differ from the paper)

Stages 1-3 are the three evaluation protocols, 4-8 the direct comparison with Islam et al. (2025) used in the
manuscript, 9-11 the tables/figures.  Every stage writes CSV/JSON into results/ and figures into figures/ and paper_figures/.
"""
import subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = [
    "run_protocol_a.py",        # 1  Islam et al. protocol (SMOTE before the split, leaky), 10 seeds
    "run_protocol_b.py",        # 2  second published protocol, 30 random splits
    "run_protocol_c.py",        # 3  leakage-free hold-out + 5x2 CV, Optuna, SHAP, fairness (writes optuna_best.json)
    "run_paper3_compare.py",    # 4  proposed model v1 / v2 vs the JEECOM paper models, leakage-free
    "run_islam_fair.py",        # 5  Islam pipeline without leakage on the same splits
    "run_ablation.py",          # 6  single learners and raw-feature ensemble
    "run_protocol_a_v2.py",     # 7  proposed model under the baseline's own (leaky) protocol
    "stats_islam.py",           # 8  paired and Nadeau-Bengio tests
    "holdout_artifacts.py",     # 9  seed-42 hold-out probabilities, SHAP, subgroup metrics
    "make_figures.py",          # 10 figures 1-9 -> figures/
    "make_paper_figures.py",    # 11 manuscript figures 1-11 -> paper_figures/
]
if __name__ == "__main__":
    t0 = time.time()
    for i, script in enumerate(STAGES, 1):
        print(f"\n{'=' * 20} [{i}/{len(STAGES)}] {script}  ({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)
        subprocess.run([sys.executable, str(HERE / script)] + sys.argv[1:], check=True, cwd=HERE)
    print(f"\nAll stages finished in {(time.time() - t0) / 60:.1f} min")
