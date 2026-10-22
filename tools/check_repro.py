"""Consistency checks used by CI. Read-only: nothing is modified.

    python tools/check_repro.py artifacts    # shipped files, README tables vs result files
    python tools/check_repro.py simulation   # run after `python sim.py 80`

`artifacts`   every required file exists, the result files agree with each other, and the
              numbers in the README result tables equal the numbers in results2_*.json.
`simulation`  leach_sim.csv has exactly the row count recorded in results2_leach.json and its
              cluster-head rows are identical to the shipped leach_ch_rows.csv.gz.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {"leach": "LEACH-Sim", "nsl": "NSL-KDD", "unsw": "UNSW-NB15"}
MODELS = ["RF", "SVM", "XGB", "LSTM", "kNN", "Ensemble"]
FIGURES = ["fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8", "fig9", "fig10", "fig11", "fig_method"]
SCORE_KEYS = {"leach": {"yte"} | {f"score_{m}" for m in MODELS},
              "nsl": {"yte"} | {f"score_{m}" for m in MODELS},
              "unsw": {"yte", "pred_Ensemble"}}
errors = []


def fail(msg):
    errors.append(msg)
    print("FAIL:", msg)


def readme_tables():
    """{dataset: ({model: (acc, macro_f1, mcc)}, {model: weight})} from the README."""
    text = (ROOT / "README.md").read_text()
    out = {}
    for key, title in DATASETS.items():
        m = re.search(rf"### {re.escape(title)}\n(.*?)(?=\n### |\Z)", text, re.S)
        if not m:
            fail(f"README has no '### {title}' section")
            continue
        rows = {r[0]: tuple(float(x) for x in r[1:4]) for r in
                (re.findall(r"\|\s*([^|\s]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", m.group(1)))}
        w = re.search(r"Ensemble weights:\s*(.+)", m.group(1))
        weights = {k: float(v) for k, v in re.findall(r"(\w+)\s+([\d.]+)", w.group(1))} if w else {}
        out[key] = (rows, weights)
    return out


def check_artifacts():
    required = ["README.md", "sim.py", "feats.py", "run2.py", "figs.py", "requirements.txt", "run_all.sh", "leach_ch_rows.csv.gz"]
    required += [f"results2_{k}.json" for k in DATASETS] + [f"scores_{k}.npz" for k in DATASETS]
    required += [f"figs/{f}.png" for f in FIGURES]
    for f in required:
        if not (ROOT / f).is_file():
            fail(f"missing file: {f}")

    tables = readme_tables()
    for key in DATASETS:
        rf, sf = ROOT / f"results2_{key}.json", ROOT / f"scores_{key}.npz"
        if not rf.is_file() or not sf.is_file():
            continue
        res = json.loads(rf.read_text())
        if res.get("dataset") != key:
            fail(f"{rf.name}: dataset field is {res.get('dataset')!r}, expected {key!r}")
        if sum(res["test_support"]) != res["n_test"]:
            fail(f"{rf.name}: test_support sums to {sum(res['test_support'])}, n_test is {res['n_test']}")
        cm = np.array(res["confusion_matrix"])
        if cm.sum() != res["n_test"]:
            fail(f"{rf.name}: confusion matrix sums to {cm.sum()}, n_test is {res['n_test']}")
        if abs(sum(res["ensemble_weights"].values()) - 1) > 1e-6:
            fail(f"{rf.name}: ensemble weights do not sum to 1")
        with np.load(sf) as z:
            missing = SCORE_KEYS[key] - set(z.files)
            if missing:
                fail(f"{sf.name}: missing arrays {sorted(missing)}")
            elif len(z["yte"]) != res["n_test"]:
                fail(f"{sf.name}: {len(z['yte'])} test labels, results say {res['n_test']}")
        if key not in tables:
            continue
        rows, weights = tables[key]
        for model in MODELS:
            if model not in rows:
                fail(f"README {DATASETS[key]}: no row for {model}")
                continue
            t = res["test_metrics"][model]
            want = (t["acc"] * 100, t["f1_macro"] * 100, t["mcc"])
            for name, got, exp, tol in zip(("accuracy", "macro-F1", "MCC"), rows[model], want, (0.0051, 0.0051, 0.00051)):
                if abs(got - exp) > tol:
                    fail(f"README {DATASETS[key]} {model} {name}: README {got}, results file {exp:.4f}")
        for model, w in res["ensemble_weights"].items():
            if abs(weights.get(model, -1) - w) > 0.0051:
                fail(f"README {DATASETS[key]} weight {model}: README {weights.get(model)}, results file {w:.4f}")
    leach = json.loads((ROOT / "results2_leach.json").read_text())
    print(f"LEACH-Sim records in results file: {leach['n_train_raw'] + leach['n_val'] + leach['n_test']}")


def check_simulation():
    csv = ROOT / "leach_sim.csv"
    if not csv.is_file():
        fail("leach_sim.csv not found; run `python sim.py 80` first")
        return
    res = json.loads((ROOT / "results2_leach.json").read_text())
    expected = res["n_train_raw"] + res["n_val"] + res["n_test"]
    df = pd.read_csv(csv)
    print(f"regenerated records: {len(df)}  (results file: {expected})")
    if len(df) != expected:
        fail(f"sim.py 80 produced {len(df)} records, results2_leach.json expects {expected}")
    if df["sim"].nunique() != 80:
        fail(f"expected 80 simulated networks, found {df['sim'].nunique()}")
    shipped = pd.read_csv(ROOT / "leach_ch_rows.csv.gz")
    ch = df.loc[df["is_ch"] == 1, list(shipped.columns)].reset_index(drop=True)
    print(f"cluster-head rows: regenerated {len(ch)}, shipped {len(shipped)}")
    if ch.shape != shipped.shape:
        fail("cluster-head rows differ in shape from leach_ch_rows.csv.gz")
        return
    for col in shipped.columns:
        if pd.api.types.is_numeric_dtype(shipped[col]):
            same = np.allclose(ch[col].to_numpy(float), shipped[col].to_numpy(float), rtol=0, atol=1e-9)
        else:
            same = bool((ch[col] == shipped[col]).all())
        if not same:
            fail(f"cluster-head column '{col}' differs from leach_ch_rows.csv.gz")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "artifacts":
        check_artifacts()
    elif mode == "simulation":
        check_simulation()
    else:
        sys.exit(__doc__)
    if errors:
        sys.exit(f"{len(errors)} check(s) failed")
    print(f"{mode}: all checks passed")
