"""Episode-level alarms under an explicit false-alarm budget (Vilnius and VitalDB test patients).

For the patient-specific monitor (selected configuration, exact reviewer) and for ACI (target 0.25 %,
step 0.001, and the primary setting), beat-level p-values / alarms are aggregated into episode alarms by
  * the conformal CUSUM (eps = 0.2) over a grid of thresholds h, 60-beat refractory period;
  * a k-of-60 run rule (an episode alarm when at least k of the last 60 beats alarmed), same refractory.
An episode alarm is false if none of the 60 beats ending at it is a target beat. For every threshold we
store, per record, the number of target episodes (>= 30 and >= 120 target beats), the number detected
(an alarm between the first beat and 5 beats after the last), the detection delays and the number of
false episode alarms, so that operating points meeting a budget (e.g. <= 1 false alarm per patient-day)
can be read off afterwards.

    python scripts/run_revision2c.py --cache cache --out results_rev2 --hours 11
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_revision2 as R  # noqa: E402
from ecgppg.pipeline import load_record  # noqa: E402
from ecgppg.scores import PopulationModels  # noqa: E402
from ecgppg.experiments import split_patients  # noqa: E402
from ecgppg.metrics import episodes, eval_hours  # noqa: E402
from ecgppg.revision2 import aci_monitor_v2, run_monitor_v2  # noqa: E402

H_GRID = [math.log(x) for x in (1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 1e8, 1e9)]
K_GRID = [5, 10, 15, 20, 30, 40, 50]
EPS, REFR, WIN = 0.2, 60, 60
MIN_LENS = [30, 120]


def cusum_events(p, h_grid, eps=EPS, refr=REFR):
    """Run the CUSUM for every threshold in one pass (one statistic per threshold)."""
    nh = len(h_grid)
    S = [0.0] * nh
    hold = [0] * nh
    ev = [[] for _ in range(nh)]
    c1 = math.log(eps)
    for i, pi in enumerate(p):
        if not (pi == pi):        # NaN
            continue
        inc = c1 + (eps - 1.0) * math.log(max(pi, 1e-12))
        for j in range(nh):
            if hold[j] > 0:
                hold[j] -= 1
                continue
            s = S[j] + inc
            if s < 0.0:
                s = 0.0
            if s > h_grid[j]:
                ev[j].append(i); s = 0.0; hold[j] = refr
            S[j] = s
    return ev


def run_rule_events(alarm, k_grid, win=WIN, refr=REFR):
    a = alarm.astype(np.int32)
    c = np.convolve(a, np.ones(win, np.int32))[:len(a)]
    out = []
    for k in k_grid:
        idx = np.where(c >= k)[0]
        ev, last = [], -10 ** 9
        for i in idx:
            if i - last > refr:
                ev.append(int(i)); last = i
        out.append(ev)
    return out


def score_events(ev, tgt, t, eps_list):
    """ev: event positions (on the evaluated-beat sequence). Returns per min_len: n_ep, n_det, delays;
    and the number of false events."""
    ev = np.asarray(ev, dtype=np.int64)
    ctg = np.r_[0, np.cumsum(tgt.astype(np.int64))]
    if len(ev):
        lo = np.maximum(0, ev - WIN + 1)
        has_t = (ctg[ev + 1] - ctg[lo]) > 0
        n_false = int((~has_t).sum())
    else:
        n_false = 0
    res = {"false": n_false}
    for ml, eps in eps_list:
        det, delays = 0, []
        for (f, l) in eps:
            j = np.searchsorted(ev, f)
            if j < len(ev) and ev[j] <= l + 5:
                det += 1
                delays.append(float(t[ev[j]] - t[f]))
        res[f"ep{ml}"] = [len(eps), det, delays]
    return res


def analyse_c(path):
    r = load_record(path)
    rid = str(r["id"])
    sc = R.scores_of(r)
    pos = np.where(r["eval_mask"])[0]
    lab = r["label"][pos]
    tgt = np.isin(lab, (1, 2))
    t = r["t"][pos]
    eps_list = [(ml, episodes(tgt, min_len=ml, max_gap=10)) for ml in MIN_LENS]
    out = {"hours": eval_hours(r)}
    mon = run_monitor_v2(sc[R.SEL_FUSION], r, R.ALPHA, R.SEL_MODE, R.SEL_Q)
    p = mon["p"][pos]
    al = mon["alarm"][pos]
    out["ps_cusum"] = [score_events(e, tgt, t, eps_list) for e in cusum_events(p, H_GRID)]
    out["ps_run"] = [score_events(e, tgt, t, eps_list) for e in run_rule_events(al, K_GRID)]
    for (a, g, tag) in [(0.0025, 0.001, "aci_cons"), (0.01, 0.005, "aci_prim")]:
        o = aci_monitor_v2(sc["early"], r, R.G["aci_cal"]["early"], a, g)
        out[f"{tag}_run"] = [score_events(e, tgt, t, eps_list) for e in run_rule_events(o["alarm"][pos], K_GRID)]
    return rid, out


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
    cal = {}
    for key in ("ecg", "early"):
        cal[key] = np.concatenate([models.raw_scores(recs[i])[key][recs[i]["eval_mask"] & (recs[i]["label"] == 0) & recs[i]["sinus"]] for i in va])
    R.G["aci_cal"] = cal
    R.G["thr"] = {k: {a: 0.0 for a in R.ALPHAS} for k in ("ecg", "early")}
    meta = {"h_grid": H_GRID, "k_grid": K_GRID, "eps": EPS, "refractory": REFR, "window": WIN, "min_lens": MIN_LENS}
    json.dump(meta, open(os.path.join(args.out, "episodes_meta.json"), "w"))
    fn = os.path.join(args.out, "vitaldb_episodes.json")
    if not os.path.exists(fn):
        acc = {}
        R.run_pool(analyse_c, [path_of[i] for i in te], workers, lambda res: acc.__setitem__(res[0], res[1]), 1e9)
        json.dump(acc, open(fn, "w"))
        R.log("VitalDB episodes done")
    fn = os.path.join(args.out, "vilnius_episodes.jsonl")
    done = set()
    if os.path.exists(fn):
        for ln in open(fn):
            try:
                done.add(json.loads(ln)["id"])
            except Exception:  # noqa: BLE001
                pass
    todo = [p for p in vil_paths if os.path.basename(p)[:-4] not in done]
    fh = open(fn, "a")

    def save(res):
        fh.write(json.dumps({"id": res[0], **res[1]}) + "\n"); fh.flush()
    finished = R.run_pool(analyse_c, todo, workers, save, args.hours)
    fh.close()
    n_done = sum(1 for _ in open(fn))
    st = {"vilnius_days_done_c": n_done, "finished": bool(finished and n_done >= len(vil_paths))}
    json.dump(st, open(os.path.join(args.out, "status_c.json"), "w"))
    R.log("status", st)


if __name__ == "__main__":
    main()
