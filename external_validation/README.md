# New dataset: UCI Student Performance (Cortez & Silva, 2008) - same method, plus a journal paper on the same data
Files: `sp_common.py` (data, features, pipelines), `run_cv.py` (repeated stratified 5-fold CV, 15 folds, 8 settings), `run_split.py` (replication of the journal paper's single 80/20 split on 10 seeds and of its repository code),
`analyze.py` (tables, Nadeau-Bengio tests, figures), `build_excel.py` (Excel). Data in `data/` (two identical GitHub copies of the UCI files). Run: `python run_cv.py && python run_split.py && python analyze.py`.
Journal paper: Kesgin et al. (2025), Applied Sciences 15(15), 8409, DOI 10.3390/app15158409.
