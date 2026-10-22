# WSN anomaly detection and attack classification: implementation

Code, result files and figures behind the manuscript "Enhanced Security for Wireless Sensor Networks Through Lightweight Machine Learning-Based Anomaly Detection and Attack Classification". The final manuscript is in `paper/Paper_final.docx`.

Every number in the paper's tables and figures comes from the scripts below. Nothing is copied from earlier publications.

## Layout
| Path | What it is |
|---|---|
| `sim.py` | LEACH-Sim simulator written for this study: 80 networks of 100 nodes, 60 rounds, four attacks (Blackhole, Grayhole, Flooding, Scheduling). `python3 sim.py 80` writes `leach_sim.csv` (git-ignored). |
| `feats.py` | Feature engineering for LEACH-Sim (36 candidate features in four groups). |
| `run2.py` | One full pipeline per data source: `python3 run2.py leach\|nsl\|unsw`. Split, RFECV (LEACH-Sim), robust scaling, SMOTE on training rows only, Optuna TPE tuning, RF / SVM-RBF / XGBoost / LSTM / k-NN, 80-trial TPE soft-vote weights, isolation forest and autoencoder, FGSM / PGD transfer test, SHAP, packet-loss sweep, latency and model size. Writes `results2_<name>.json` and `arrays_<name>.npz` (the latter is git-ignored). |
| `figs.py` | Redraws every figure in `figs/` from the result files (no retraining needed). |
| `results2_*.json` | The numbers reported in the paper (metrics, confusion matrices, weights, tuning, SHAP, sweeps, cost). |
| `scores_*.npz` | Compact test-set scores used only by `figs.py` (attack score per model for ROC / PR, ensemble predictions for the UNSW-NB15 confusion matrix). `run2.py` also writes the full `arrays_<name>.npz` probabilities; `figs.py` uses them when present and gives identical figures. |
| `leach_ch_rows.csv.gz` | Cluster-head rows of LEACH-Sim (the part `figs.py` needs for Figure 6). The full 448,294-record dataset is not stored, to keep the repository small: `python3 sim.py 80` regenerates it exactly in under a minute (checked against the dataset used in the paper). |
| `figs/` | Final figures: `fig2`-`fig4` and `fig_method` (Figure 5) are the diagrams, `fig5`-`fig11` are Figures 6-12 in the paper. |
| `logs/` | Console logs of the runs that produced the result files. |
| `data/` | Where to put NSL-KDD and UNSW-NB15 (see `data/README.md`). |
| `run_all.sh` | Full re-run in one command. |

## Quick start
```bash
pip install -r requirements.txt
python3 figs.py            # redraw all figures from the shipped results (about 1 minute)
./run_all.sh               # or retrain everything (needs the two public datasets in data/)
```

## Design points worth knowing
- LEACH-Sim split is by whole simulation: 48 train / 8 validation / 24 test networks, so rounds of one network never appear on both sides.
- NSL-KDD and UNSW-NB15 use the official training and test files; 15% of the training file is held out (stratified) for validation.
- Scaler, feature selection and SMOTE are fitted on training rows only. The validation part decides early stopping, tuning and ensemble weights. The test part is scored once.
- The two detectors (isolation forest, autoencoder) are trained on normal rows only and are not part of the vote.
- Seed 42 throughout. Latency and model-size figures are measured on the machine that runs the script (x86-64 CPU), so they will differ on other hardware; they say nothing about real sensor motes.
- LEACH-Sim is a simulation. Results on it do not show performance on real deployments.

## Reported test results

### LEACH-Sim

| Model | Accuracy (%) | Macro-F1 (%) | MCC |
|---|---|---|---|
| RF | 99.53 | 92.68 | 0.941 |
| SVM | 99.28 | 88.86 | 0.909 |
| XGB | 99.54 | 92.90 | 0.942 |
| LSTM | 99.51 | 92.28 | 0.938 |
| kNN | 99.15 | 87.25 | 0.891 |
| Ensemble | 99.55 | 92.94 | 0.943 |

Ensemble weights: RF 0.03, SVM 0.03, XGB 0.49, LSTM 0.38, kNN 0.06

### NSL-KDD

| Model | Accuracy (%) | Macro-F1 (%) | MCC |
|---|---|---|---|
| RF | 74.51 | 53.86 | 0.633 |
| SVM | 75.53 | 54.03 | 0.647 |
| XGB | 78.06 | 58.09 | 0.683 |
| LSTM | 74.80 | 55.74 | 0.629 |
| kNN | 76.73 | 56.11 | 0.663 |
| Ensemble | 76.26 | 58.03 | 0.657 |

Ensemble weights: RF 0.07, SVM 0.02, XGB 0.36, LSTM 0.25, kNN 0.30

### UNSW-NB15

| Model | Accuracy (%) | Macro-F1 (%) | MCC |
|---|---|---|---|
| RF | 75.00 | 47.00 | 0.678 |
| SVM | 75.51 | 41.90 | 0.674 |
| XGB | 76.51 | 51.57 | 0.697 |
| LSTM | 76.62 | 46.50 | 0.698 |
| kNN | 72.21 | 42.06 | 0.639 |
| Ensemble | 76.20 | 51.03 | 0.693 |

Ensemble weights: RF 0.07, SVM 0.01, XGB 0.74, LSTM 0.00, kNN 0.17
