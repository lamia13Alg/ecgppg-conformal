"""Simulation study: validity and power of sliding conformal monitoring with contaminated calibration
under independence, autocorrelation, drift and clustered arrhythmia (synthetic score streams).

    python scripts/simulation_study.py --out results --reps 30
"""
import argparse
import zlib
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ecgppg.conformal import run_monitor  # noqa: E402

N, NCAL, DELTA = 6000, 600, 4.0


def stream(rng, dep, eps, cluster):
    if dep == "iid":
        z = rng.normal(size=N)
    elif dep.startswith("ar"):
        rho = float(dep[2:]) / 10
        e = rng.normal(size=N) * np.sqrt(1 - rho ** 2)
        z = np.empty(N); z[0] = rng.normal()
        for i in range(1, N):
            z[i] = rho * z[i - 1] + e[i]
    elif dep == "drift":
        z = rng.normal(size=N) + np.linspace(0, 1.0, N)
    lab = np.zeros(N, int)
    n_ab = int(round(eps * N))
    if n_ab:
        if cluster:
            L = 20
            starts = rng.choice(np.arange(0, N - L, L), size=max(1, n_ab // L), replace=False)
            for s in starts:
                lab[s:s + L] = 1
        else:
            lab[rng.choice(N, size=n_ab, replace=False)] = 1
    s = z.copy()
    s[lab == 1] = DELTA + rng.normal(size=(lab == 1).sum())
    r = {"t": np.arange(N) * 0.8, "label": lab, "cal_mask": np.arange(N) < NCAL,
         "eval_mask": np.arange(N) >= NCAL}
    return s, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    deps = ["iid", "ar5", "ar9", "drift"]
    epss = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
    Ms = [100, 300, 1000]
    modes = [("labelfree", "labelfree", 0.0), ("adj0", "adjudicated", 0.0), ("adj10", "adjudicated", 0.10)]
    res = []
    for dep, eps, cl, M in itertools.product(deps, epss, [False, True], Ms):
        if eps == 0 and cl:
            continue
        acc = {m[0]: {"fp": 0, "nn": 0, "tp": 0, "na": 0, "b": []} for m in modes}
        for k in range(args.reps):
            rng = np.random.default_rng(zlib.crc32(f'{dep}|{eps}|{cl}|{M}|{k}'.encode()))
            s, r = stream(rng, dep, eps, cl)
            for name, mode, q in modes:
                o = run_monitor(s, r, alpha=0.01, mode=mode, q=q, M=M, G=20)
                ev = r["eval_mask"] & np.isfinite(o["p"])
                a = acc[name]
                a["fp"] += int((o["alarm"] & ev & (r["label"] == 0)).sum())
                a["nn"] += int((ev & (r["label"] == 0)).sum())
                a["tp"] += int((o["alarm"] & ev & (r["label"] == 1)).sum())
                a["na"] += int((ev & (r["label"] == 1)).sum())
                b = o["bound"][ev & (r["label"] == 0)]
                a["b"].append(float(np.nanmean(b)) if np.isfinite(b).any() else np.nan)
        for name, a in acc.items():
            res.append({"dep": dep, "eps": eps, "cluster": cl, "M": M, "mode": name,
                        "far": 100 * a["fp"] / a["nn"], "bound": 100 * float(np.nanmean(a["b"])),
                        "se": 100 * a["tp"] / a["na"] if a["na"] else np.nan})
        print(dep, eps, cl, M, [(x["mode"], round(x["far"], 2), round(x["bound"], 2), round(x["se"], 1) if x["se"] == x["se"] else None)
                                for x in res[-3:]], flush=True)
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, "simulation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
