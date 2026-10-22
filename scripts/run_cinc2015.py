"""External test 2: PPG witness for false-alarm suppression on PhysioNet/CinC Challenge 2015.

    wget -r -N -c -np https://physionet.org/files/challenge-2015/1.0.0/training/
    pip install wfdb
    python scripts/run_cinc2015.py --dir physionet.org/files/challenge-2015/1.0.0/training --out results
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ecgppg.data import load_cinc2015_record  # noqa: E402
from ecgppg.alarm import ppg_witness, decide, challenge_score  # noqa: E402

TYPES = ["Asystole", "Bradycardia", "Tachycardia", "Ventricular_Tachycardia", "Ventricular_Flutter_Fib"]
SHORT = {"Asystole": "ASY", "Bradycardia": "EBR", "Tachycardia": "ETC",
         "Ventricular_Tachycardia": "VTA", "Ventricular_Flutter_Fib": "VFB"}


def summarise(rows, beta):
    out = {}
    for ty in TYPES + ["All"]:
        rr = [r for r in rows if ty == "All" or r["type"] == ty]
        keep = [decide(r["p"], r["sqi"], beta) for r in rr]
        tp = sum(k and r["truth"] == 1 for k, r in zip(keep, rr))
        fn = sum((not k) and r["truth"] == 1 for k, r in zip(keep, rr))
        tn = sum((not k) and r["truth"] == 0 for k, r in zip(keep, rr))
        fp = sum(k and r["truth"] == 0 for k, r in zip(keep, rr))
        out[ty] = {"n": len(rr), "n_true": tp + fn, "n_false": tn + fp,
                   "tpr": 100.0 * tp / (tp + fn) if tp + fn else float("nan"),
                   "tnr": 100.0 * tn / (tn + fp) if tn + fp else float("nan"),
                   "score": challenge_score(tp, tn, fp, fn),
                   "score_keep_all": challenge_score(tp + fn, 0, tn + fp, 0)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    rows = []
    for hea in sorted(glob.glob(os.path.join(args.dir, "*.hea"))):
        rec = load_cinc2015_record(hea[:-4])
        if rec["truth"] is None:
            continue
        p, sqi = ppg_witness(rec["ppg"], rec["fs"], rec["alarm_t"])
        # longest names first: "Tachycardia" is a substring of "Ventricular_Tachycardia"
        ty = next((t for t in sorted(TYPES, key=len, reverse=True)
                   if t.lower() in rec["type"].lower()), rec["type"])
        rows.append({"id": rec["id"], "type": ty, "truth": rec["truth"], "p": p, "sqi": sqi,
                     "has_ppg": rec["ppg"] is not None})
        print(rec["id"], ty, rec["truth"], f"p={p:.3f} sqi={sqi:.2f}")
    res = {str(b): summarise(rows, b) for b in [0.05, 0.10, 0.20, 0.30]}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results_cinc.json"), "w") as fh:
        json.dump({"rows": rows, "summary": res}, fh, indent=1)
    main_ = res["0.1"]
    lines = []
    for ty in TYPES + ["All"]:
        m = main_[ty]
        name = "All alarms" if ty == "All" else SHORT[ty]
        lines.append(f"{name} & {m['n_true']} & {m['n_false']} & {m['tpr']:.1f} & {m['tnr']:.1f} & "
                     f"{m['score']:.1f} & {m['score_keep_all']:.1f} \\\\")
    with open(os.path.join(args.out, "tab_cinc.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    a = main_["All"]
    with open(os.path.join(args.out, "macros_cinc.tex"), "w") as fh:
        fh.write(f"\\newcommand{{\\resCincN}}{{{a['n']}}}\n")
        fh.write(f"\\newcommand{{\\resCincTpr}}{{{a['tpr']:.1f}}}\n")
        fh.write(f"\\newcommand{{\\resCincTnr}}{{{a['tnr']:.1f}}}\n")
        fh.write(f"\\newcommand{{\\resCincScore}}{{{a['score']:.1f}}}\n")
        fh.write(f"\\newcommand{{\\resCincNoPpg}}{{{sum(not r['has_ppg'] for r in rows)}}}\n")
    print(json.dumps(main_["All"], indent=1))


if __name__ == "__main__":
    main()
