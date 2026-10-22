"""Complement to run_revision2.py: a fair comparison with ACI.

  * ACI grid (targets 0.25-2 %, step sizes 0.0005-0.02) on the VitalDB VALIDATION patients, so that
    an ACI setting can be selected by the same pre-specified rule as ours (no test data);
  * the conservative ACI settings under the same imperfect reviewers as the patient-specific monitor
    (independent and task-level errors, false- and missed-confirmation-only errors, random and
    periodic delays), 3 seeds;
  * the patient-specific monitor under task-level false-only / missed-only errors.

    python scripts/run_revision2b.py --cache cache --out results_rev2 --hours 11
Writes vitaldb_val_aci.json, vitaldb_rows_b.json and vilnius_rows_b.jsonl (resumable).
"""
import argparse
import glob
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_revision2 as R  # noqa: E402
from ecgppg.pipeline import load_record  # noqa: E402
from ecgppg.scores import PopulationModels  # noqa: E402
from ecgppg.experiments import split_patients  # noqa: E402
from ecgppg.revision2 import aci_monitor_v2, run_monitor_v2  # noqa: E402

A_GRID = [0.0025, 0.005, 0.01, 0.02]
G_GRID = [0.0005, 0.001, 0.002, 0.005, 0.02]
ACI_SET = [(0.0025, 0.001), (0.005, 0.001), (0.01, 0.001)]
SEEDS = [0, 1, 2]
REV = [("err5_beat", {"fp_rate": .05, "fn_rate": .05, "error_model": "beat"}, True),
       ("err5_task", {"fp_rate": .05, "fn_rate": .05, "error_model": "group"}, True),
       ("err10_task", {"fp_rate": .10, "fn_rate": .10, "error_model": "group"}, True),
       ("fp10_task", {"fp_rate": .10, "error_model": "group"}, True),
       ("fn10_task", {"fn_rate": .10, "error_model": "group"}, True),
       ("expdelay60", {"delay_s": 60.0, "delay_model": "exp"}, False),
       ("expdelay300", {"delay_s": 300.0, "delay_model": "exp"}, True),
       ("batch15", {"delay_s": 900.0, "delay_model": "batch"}, False)]


def analyse_b(path, full=True):
    r = load_record(path)
    rid = str(r["id"])
    sc = R.scores_of(r)
    ctx = R.record_context(r)
    rows = {}

    def sd(name, k):
        return zlib.crc32(f"{name}|{rid}|{k}".encode())
    for a in A_GRID:
        for g in G_GRID:
            if g in (0.001, 0.005, 0.02) and full:
                continue                      # already computed by run_revision2
            o = aci_monitor_v2(sc["early"], r, R.G["aci_cal"]["early"], a, g)
            rows[f"aci_a{a}_g{g}"] = R.row_for(r, o["alarm"], ctx, {"tasks": o["tasks"], "reviews": o["reviews"]})
    if not full:
        for a in A_GRID:
            for g in (0.001, 0.005, 0.02):
                o = aci_monitor_v2(sc["early"], r, R.G["aci_cal"]["early"], a, g)
                rows[f"aci_a{a}_g{g}"] = R.row_for(r, o["alarm"], ctx, {"tasks": o["tasks"], "reviews": o["reviews"]})
        return rid, rows, ctx["desc"]
    for (a, g) in ACI_SET:
        for name, kw, multi in REV:
            for k in (SEEDS if multi else [0]):
                o = aci_monitor_v2(sc["early"], r, R.G["aci_cal"]["early"], a, g, seed=sd(f"aci_{name}", k), **kw)
                key = f"aci_a{a}_g{g}_{name}" + (f"_s{k}" if multi else "")
                rows[key] = R.row_for(r, o["alarm"], ctx, {"tasks": o["tasks"], "reviews": o["reviews"]})
    s = sc[R.SEL_FUSION]
    mon = run_monitor_v2(s, r, R.ALPHA, R.SEL_MODE, R.SEL_Q)      # reproduction check of the earlier run
    rows["repro_ps"] = R.row_for(r, mon["alarm"], ctx, mon)
    for name, kw in [("rev_fp10_task", {"fp_rate": .10, "error_model": "group"}),
                     ("rev_fn10_task", {"fn_rate": .10, "error_model": "group"})]:
        for k in SEEDS:
            mon = run_monitor_v2(s, r, R.ALPHA, R.SEL_MODE, R.SEL_Q, seed=sd(name, k), **kw)
            rows[f"{name}_s{k}"] = R.row_for(r, mon["alarm"], ctx, mon)
    return rid, rows, ctx["desc"]


def analyse_val(path):
    return analyse_b(path, full=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="results_rev2")
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    workers = args.workers or max(1, (os.cpu_count() or 2))
    paths = sorted(glob.glob(os.path.join(args.cache, "vitaldb", "*.npz")))
    allrecs = [load_record(p) for p in paths]
    recs = {r["id"]: r for r in allrecs
            if r["eval_mask"].sum() > 0 and (r["label"][r["fit_mask"]] == 0).sum() >= 50}
    path_of = {r["id"]: p for r, p in zip(allrecs, paths)}
    vil_paths = sorted(glob.glob(os.path.join(args.cache, "vilnius", "*.npz")))
    tr, va, te = split_patients(list(recs), seed=R.SEED)
    models = PopulationModels(R.SEED).fit([recs[i] for i in tr])
    R.G["models"] = models
    thr, cal = {}, {}
    for key in ("ecg", "early"):
        v = np.concatenate([models.raw_scores(recs[i])[key][recs[i]["eval_mask"] & (recs[i]["label"] == 0) & recs[i]["sinus"]] for i in va])
        thr[key] = {a: float(np.quantile(v, 1 - a)) for a in R.ALPHAS}
        cal[key] = v
    R.G["thr"], R.G["aci_cal"] = thr, cal
    R.self_test(recs[te[0]])
    R.log(f"VitalDB val {len(va)} test {len(te)}; Vilnius {len(vil_paths)} days; {workers} workers")
    fn = os.path.join(args.out, "vitaldb_val_aci.json")
    if not os.path.exists(fn):
        acc = {}
        R.run_pool(analyse_val, [path_of[i] for i in va], workers,
                   lambda res: acc.__setitem__(res[0], {"rows": res[1], "desc": res[2]}), 1e9)
        json.dump(acc, open(fn, "w"), default=float)
        R.log("validation ACI grid done")
    fn = os.path.join(args.out, "vitaldb_rows_b.json")
    if not os.path.exists(fn):
        acc = {}
        R.run_pool(analyse_b, [path_of[i] for i in te], workers,
                   lambda res: acc.__setitem__(res[0], {"rows": res[1], "desc": res[2]}), 1e9)
        json.dump(acc, open(fn, "w"), default=float)
        R.log("VitalDB test done")
    fn = os.path.join(args.out, "vilnius_rows_b.jsonl")
    done = set()
    if os.path.exists(fn):
        for ln in open(fn):
            try:
                done.add(json.loads(ln)["id"])
            except Exception:  # noqa: BLE001
                pass
    todo = [p for p in vil_paths if os.path.basename(p)[:-4] not in done]
    fh = open(fn, "a")
    cnt = [len(done)]

    def save(res):
        fh.write(json.dumps({"id": res[0], "rows": res[1], "desc": res[2]}, default=float) + "\n"); fh.flush()
        cnt[0] += 1
        if cnt[0] % 20 == 0:
            R.log(f"  Vilnius {cnt[0]} / {len(vil_paths)}")
    finished = R.run_pool(analyse_b, todo, workers, save, args.hours)
    fh.close()
    n_done = sum(1 for _ in open(fn))
    st = {"vilnius_days_done_b": n_done, "vilnius_days_total": len(vil_paths),
          "finished": bool(finished and n_done >= len(vil_paths))}
    json.dump(st, open(os.path.join(args.out, "status_b.json"), "w"))
    R.log("status", st)


if __name__ == "__main__":
    main()
