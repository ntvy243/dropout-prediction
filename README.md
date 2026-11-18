# Student dropout & academic success prediction

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/dropout-prediction/blob/main/notebooks/01_full_pipeline_colab.ipynb)

Code for a cost-sensitive soft-voting ensemble (LightGBM, CatBoost, XGBoost, ExtraTrees; 18 engineered features) that predicts **Dropout / Enrolled / Graduate** on the UCI dataset 697 (4,424 students, 36 predictors), and for its direct comparison with Islam et al. (2025), *Telematics and Informatics Reports* 18, 100203 (XGBoost, accuracy ≈ 0.83).

> Replace `YOUR-USERNAME` in this file and in both notebooks after you create the GitHub repository.

## Main findings (all numbers regenerate from the code)

| Evaluation | Result |
|---|---|
| Baseline's own protocol (IQR clipping, top-20 features and SMOTE on all data *before* the split) | the proposed shallow ensemble scores 0.824; Islam et al. report ≈ 0.83; a higher-capacity top-3 ensemble reaches 0.8495 |
| Same pipeline but leakage-free (split first, everything fitted on the training part) | SMOTE-before-split inflates baseline accuracy by 0.074 and Enrolled F1 by 0.332 |
| Leakage-free, 15 CV folds, same splits | proposed model (v2) beats Islam's pipeline: accuracy 0.783 vs 0.764 (+0.019), macro-F1 +0.027, won 15 of 15 folds (Nadeau–Bengio test) |
| Enrolled class (17.9 % of students) | balanced class weights (v1) raise Enrolled recall by 0.147 |

v2 uses square-root class weights, w_c = (n / (K·n_c))^0.5; it was chosen after seeing hold-out results, so its gains may be slightly optimistic. An ablation shows that a single boosted model captures most of the gain.

## Quick start on Google Colab

1. Push this folder to GitHub (see below).
2. Open `notebooks/01_full_pipeline_colab.ipynb` in Colab (use the badge above), set your user name in the first cell and run the cells in order. Start with the *smoke test* (`--fast`); the full run takes hours, and results can be saved to Google Drive.
3. `notebooks/02_bert_baseline_colab.ipynb` needs a GPU runtime (T4) and runs the BERT baseline.

## Run locally

```bash
pip install -r requirements.txt
python src/download_data.py          # only if data/data.csv is missing
python src/run_all.py --fast         # quick test
python src/run_all.py                # full run, writes results/, figures/, paper_figures/
```

Optional environment variables: `DATA_PATH`, `RESULTS_DIR` (figures go to its parent folder), `N_JOBS`. `requirements-frozen.txt` lists the exact versions used for the paper.

## Repository layout

```
data/                 data.csv (UCI 697, CC BY 4.0) + source note
src/
  common.py             data loading, IQR clipping, 18 engineered features, models, soft vote, metrics, Nadeau–Bengio test
  run_all.py            runs the 11 stages below in order
  run_protocol_a.py     1  Islam et al. protocol (SMOTE before split, leaky), 10 seeds
  run_protocol_b.py     2  second published protocol, 30 random splits
  run_protocol_c.py     3  leakage-free hold-out + 5x2 CV, Optuna, McNemar, bootstrap, SHAP, fairness
  run_paper3_compare.py 4  proposed model v1 / v2 vs the JEECOM paper models (leakage-free)
  run_islam_fair.py     5  Islam pipeline run without leakage on the same 11 hold-outs and 15 CV folds
  run_ablation.py       6  single learners and raw-feature ensemble
  run_protocol_a_v2.py  7  proposed model under the baseline's own protocol
  stats_islam.py        8  paired and Nadeau–Bengio tests
  holdout_artifacts.py  9  seed-42 hold-out probabilities, SHAP values, subgroup metrics
  make_figures.py       10 figures 1–9
  make_paper_figures.py 11 the 11 manuscript figures
  run_bert.py           BERT baseline (GPU): each student written as a sentence, fine-tuned and compared (McNemar, bootstrap)
  download_data.py      fetch the dataset from UCI
external_validation/  same method on two other datasets (see its README)
notebooks/            Colab notebooks
results/              CSV/JSON behind every table in the manuscript
figures/, paper_figures/   PNG (300 dpi) + PDF
```

## Evaluation protocols

- **A – baseline's own (leaky):** IQR clipping, top-20 feature selection and SMOTE on the full data, then Min-Max scaling and the split.
- **C – leakage-free:** split first; clipping, feature selection, scaling, class weights and tuning use the training part only. All headline claims use this protocol.
- Significance: paired t-test over seeds or hold-outs; Nadeau–Bengio corrected t-test (n_test/n_train = 0.25) for the 15 repeated-CV folds. p-values are not adjusted for multiple comparisons.

## Put it on GitHub

```bash
cd dropout-prediction
git init && git add . && git commit -m "Dropout prediction: code, data, results and Colab notebooks"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/dropout-prediction.git
git push -u origin main
```

## Data and licence

Data: Realinho, V., Machado, J., Baptista, L., Martins, M.V. (2022). Predicting student dropout and academic success. *Data* 7(11), 146. https://doi.org/10.3390/data7110146 (CC BY 4.0). Code: MIT (`LICENSE`; fill in the author names).
