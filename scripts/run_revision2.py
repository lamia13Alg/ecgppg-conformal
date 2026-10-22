"""Second-revision analyses (main split, seed 42; all on cached features).

  T1  VitalDB test patients, every method below, per record           -> vitaldb_rows.json
  T2  appendix tables recomputed with the definitions of the main text -> appendix.json
      (false alarms counted on normal beats in sinus rhythm everywhere)
  T3  bounded-look-ahead and causal filtering check on the VitalDB test
      patients (ECG re-downloaded; needs Internet and the vitaldb package) -> causal.json
  T4  Vilnius, every method below, per day (resumable)                 -> vilnius_rows.jsonl

Methods per record:
  * patient-specific monitor (selected configuration) at alpha in {0.1, 0.25, 0.5, 1, 2}% with buffer
    sizes M = 300 and M = max(300, ceil(3/alpha));
  * population thresholds (validation quantile at the same alpha; P > 0.5);
  * ACI on the population joint score for a grid of target levels and step sizes;
  * reviewer scenarios: beat-, task- and score-correlated errors (3 seeds each), exponential delays,
    periodic (batch) review, provisional entry of beats awaiting review, partial availability with
    unreviewed alarms kept out of the buffer, episode (strip) exclusion, ACI with the same reviewers;
  * oracle-clean buffer (decomposition of the sensitivity loss);
  * day descriptors (AF burden, PPG coverage, ECG quality, rhythm irregularity, score drift).

    python scripts/run_revision2.py --cache cache --out results_rev2 --hours 11
"""
import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ecgppg.pipeline import load_record, CALIBRATIONS, auc_score_of  # noqa: E402
from ecgppg.scores import PopulationModels, FUSIONS, P_FUSIONS, fused_scores  # noqa: E402
from ecgppg.experiments import split_patients  # noqa: E402
from ecgppg.metrics import eval_hours, episodes  # noqa: E402
from ecgppg.conformal import run_monitor, morphology_screen  # noqa: E402
from ecgppg.revision import run_monitor_ext, aci_monitor, notifications  # noqa: E402
from ecgppg.revision2 import (run_monitor_v2, aci_monitor_v2, review_tasks,  # noqa: E402
                              ecg_features_bounded, ecg_features_causal)

ALPHA = 0.01
SEED = 42
SEL_FUSION, SEL_MODE, SEL_Q = "ecg", "adjudicated", 0.05     # configuration selected on validation
ALPHAS = [0.001, 0.0025, 0.005, 0.01, 0.02]
ACI_ALPHAS = [0.0025, 0.005, 0.01, 0.02]
ACI_GAMMAS = [0.001, 0.005, 0.02]
RSEEDS = [0, 1, 2]
LOOKAHEADS = [1.0, 2.0, 3.0]
T0 = time.time()
G = {}


def hrs():
    return (time.time() - T0) / 3600


def log(*a):
    print(f"[{hrs():5.2f} h]", *a, flush=True)


def m_for(alpha):
    return max(300, int(math.ceil(3.0 / alpha)))


# --------------------------------------------------------------------------------------------- helpers
def base_counts(r, alarm, target=(1, 2), ev_labels=(0, 1, 2)):
    lab = r["label"]
    ev = r["eval_mask"] & np.isin(lab, ev_labels)
    tgt = ev & np.isin(lab, target)
    nor = ev & (lab == 0) & r["sinus"]
    return ev, tgt, nor


def row_for(r, alarm, ctx_rec, mon=None, target=(1, 2)):
    ev, tgt, nor = base_counts(r, alarm, target)
    al = alarm & r["eval_mask"]
    out = {"tp": int((tgt & al).sum()), "n_tgt": int(tgt.sum()),
           "fp": int((nor & al).sum()), "n_nor": int(nor.sum())}
    nt, nf = notifications(al, r["t"], r["label"], r["eval_mask"], 60.0, target)
    out["nt"], out["nf"] = int(nt), int(nf)
    out["tasks_all"] = review_tasks(al, r["t"], r["eval_mask"])
    irr = ctx_rec["irr"]
    out["fp_irr"] = int((nor & irr & al).sum())
    near = ctx_rec["near_tgt"]
    out["fp_near"] = int((nor & near & al).sum())
    if mon is not None:
        for k in ("reviews", "tasks", "rev_ab", "miss_ab", "rev_nor", "fpconf_nor", "startup_reviews"):
            if k in mon:
                out[k] = int(mon[k])
        if "bound" in mon:
            b = mon["bound"][nor]
            out["bound_sum"] = float(np.nansum(b)); out["bound_n"] = int(np.isfinite(b).sum())
    return out


def record_context(r, target=(1, 2)):
    """Per-record descriptors and masks shared by all methods."""
    lab = r["label"]
    ev = r["eval_mask"]
    t = r["t"]
    Xe = r["Xe"]
    irr = (np.asarray(Xe[:, 64]) >= 0.2) | (np.asarray(Xe[:, 65]) >= 0.2)
    tg = np.isin(lab, target)
    # normal beats within 60 s of a target beat (peri-arrhythmia)
    near = np.zeros(len(t), bool)
    if tg.any():
        tt = t[tg]
        j = np.searchsorted(tt, t)
        d1 = np.abs(t - tt[np.clip(j - 1, 0, len(tt) - 1)])
        d2 = np.abs(tt[np.clip(j, 0, len(tt) - 1)] - t)
        near = np.minimum(d1, d2) <= 60.0
    nor = ev & (lab == 0) & r["sinus"]
    sqi = np.asarray(r["sqi"], float)
    rr = np.r_[np.nan, np.diff(t)]
    rrn = rr[nor & np.isfinite(rr) & (rr < 3)]
    pos = np.where(ev)[0]
    eps = episodes(tg[pos], min_len=30) if len(pos) else []
    desc = {"hours": eval_hours(r), "n_eval": int(ev.sum()), "n_nor": int(nor.sum()),
            "burden": float(tg[ev].mean()) if ev.any() else 0.0,
            "ppg_cov": float((sqi[ev] != 0).mean()) if ev.any() else 0.0,
            "sqi_mean": float(sqi[ev & (sqi != 0)].mean()) if (ev & (sqi != 0)).any() else float("nan"),
            "ecg_bad": float(np.asarray(r["ecg_bad"], bool)[ev].mean()) if ev.any() else 0.0,
            "hr": float(60.0 / np.median(rrn)) if len(rrn) else float("nan"),
            "rr_cv": float(np.std(rrn) / np.mean(rrn)) if len(rrn) > 2 else float("nan"),
            "irr_frac_nor": float(irr[nor].mean()) if nor.any() else float("nan"),
            "near_frac_nor": float(near[nor].mean()) if nor.any() else float("nan"),
            "n_episodes30": len(eps)}
    return {"irr": irr, "near_tgt": near, "desc": desc}


def scenario_list():
    S = [("rev_oracle", {}, [0])]
    for s in RSEEDS:
        S.append((f"rev_err5_beat_s{s}", {"fp_rate": .05, "fn_rate": .05, "error_model": "beat"}, [s]))
        S.append((f"rev_err5_task_s{s}", {"fp_rate": .05, "fn_rate": .05, "error_model": "group"}, [s]))
        S.append((f"rev_err10_task_s{s}", {"fp_rate": .10, "fn_rate": .10, "error_model": "group"}, [s]))
        S.append((f"rev_err5_score_s{s}", {"fp_rate": .05, "fn_rate": .05, "error_model": "score"}, [s]))
        S.append((f"rev_expdelay300_s{s}", {"delay_s": 300.0, "delay_model": "exp"}, [s]))
        S.append((f"rev_epiexcl_err5_task_s{s}", {"fp_rate": .05, "fn_rate": .05, "error_model": "group",
                                                  "episode_exclusion": True}, [s]))
    S += [("rev_expdelay60", {"delay_s": 60.0, "delay_model": "exp"}, [0]),
          ("rev_delay300_prov", {"delay_s": 300.0, "pending_policy": "provisional"}, [0]),
          ("rev_batch15", {"delay_s": 900.0, "delay_model": "batch"}, [0]),
          ("rev_batch15_prov", {"delay_s": 900.0, "delay_model": "batch", "pending_policy": "provisional"}, [0]),
          ("rev_avail50_withhold", {"avail": 0.5, "unreviewed": "withhold"}, [0]),
          ("rev_epiexcl_oracle", {"episode_exclusion": True}, [0])]
    return S


ACI_REV = [("aci_err5_beat", {"fp_rate": .05, "fn_rate": .05, "error_model": "beat"}, True),
           ("aci_err5_task", {"fp_rate": .05, "fn_rate": .05, "error_model": "group"}, True),
           ("aci_expdelay300", {"delay_s": 300.0, "delay_model": "exp"}, True),
           ("aci_batch15", {"delay_s": 900.0, "delay_model": "batch"}, False)]


# --------------------------------------------------------------------------------------------- per record
def scores_of(r):
    m = G["models"]
    Xe = np.asarray(r["Xe"], float); Xp = np.asarray(r["Xp"], float)
    # float32 rounding as in PopulationModels.raw_scores (first-revision runs)
    return {"ecg": PopulationModels._logit(m.ecg, Xe).astype(np.float32).astype(float),
            "early": PopulationModels._logit(m.joint, np.hstack([Xe, Xp])).astype(np.float32).astype(float)}


def analyse_record(path):
    r = load_record(path)
    rid = str(r["id"])
    sc = scores_of(r)
    ctx = record_context(r)
    ev = r["eval_mask"]
    rows = {}
    s = sc[SEL_FUSION]

    def sd(name, k=0):
        return zlib.crc32(f"{name}|{rid}|{k}".encode())

    # alpha sweep, patient-specific
    for a in ALPHAS:
        for M in sorted({300, m_for(a)}):
            if (M + 1) * a < 1.0 - 1e-12:      # p >= 1/(M+1) > alpha: no alarm is possible
                rows[f"ps_a{a}_M{M}"] = {"impossible": 1}
                continue
            mon = run_monitor_v2(s, r, a, SEL_MODE, SEL_Q, M=M, seed=sd("ps"))
            rows[f"ps_a{a}_M{M}"] = row_for(r, mon["alarm"], ctx, mon)
    # population thresholds
    for key in ("ecg", "early"):
        rows[f"pop_{key}_default"] = row_for(r, (sc[key] > 0) & ev, ctx)
        for a in ALPHAS:
            rows[f"pop_{key}_a{a}"] = row_for(r, (sc[key] > G["thr"][key][a]) & ev, ctx)
    # ACI grid (joint score, as in the first revision) and ECG score at the nominal level
    for a in ACI_ALPHAS:
        for g in ACI_GAMMAS:
            o = aci_monitor_v2(sc["early"], r, G["aci_cal"]["early"], a, g)
            rows[f"aci_a{a}_g{g}"] = row_for(r, o["alarm"], ctx, {"tasks": o["tasks"], "reviews": o["reviews"]})
    o = aci_monitor_v2(sc["ecg"], r, G["aci_cal"]["ecg"], ALPHA, 0.005)
    rows["aci_ecg_a0.01_g0.005"] = row_for(r, o["alarm"], ctx, {"tasks": o["tasks"], "reviews": o["reviews"]})
    # reviewer scenarios (patient-specific, alpha = 1 %)
    for name, kw, seeds in scenario_list():
        mon = run_monitor_v2(s, r, ALPHA, SEL_MODE, SEL_Q, seed=sd(name, seeds[0]), **kw)
        rows[name] = row_for(r, mon["alarm"], ctx, mon)
    for name, kw, multi in ACI_REV:
        for k in (RSEEDS if multi else [0]):
            o = aci_monitor_v2(sc["early"], r, G["aci_cal"]["early"], ALPHA, 0.005, seed=sd(name, k), **kw)
            rows[f"{name}_s{k}" if multi else name] = row_for(r, o["alarm"], ctx, {"tasks": o["tasks"], "reviews": o["reviews"]})
    # decomposition: oracle-clean buffer
    mon = run_monitor_v2(s, r, ALPHA, SEL_MODE, SEL_Q, oracle_clean=True)
    rows["oracle_clean"] = row_for(r, mon["alarm"], ctx, mon)
    # score drift of normal beats (selected score): eval median minus calibration-window median
    nor_cal = r["cal_mask"] & (r["label"] == 0)
    nor_ev = ev & (r["label"] == 0) & r["sinus"]
    desc = dict(ctx["desc"])
    if nor_cal.any() and nor_ev.any():
        desc["score_drift"] = float(np.median(s[nor_ev]) - np.median(s[nor_cal]))
        desc["score_q99_shift"] = float(np.percentile(s[nor_ev], 99) - np.percentile(s[nor_cal], 99))
    return rid, rows, desc


def appendix_record(path):
    """Every (calibration, fusion) configuration and the population thresholds on one record,
    with the definitions of the main text (FAR on normal beats in sinus rhythm)."""
    r = load_record(path)
    m = G["models"]
    raw = {k: v.astype(np.float32).astype(float) for k, v in m._raw_scores(r).items()}
    fs = fused_scores(raw, r)
    hours = eval_hours(r)
    lab = r["label"]
    ev = r["eval_mask"] & np.isin(lab, (0, 1, 2))
    bad = np.asarray(r["ecg_bad"], bool)
    out = {}

    def cnt(alarm, res=None, auc_s=None):
        al = alarm & r["eval_mask"]
        tgt = ev & (lab > 0)
        nor = ev & (lab == 0) & r["sinus"]
        d = {"tp": int((tgt & al).sum()), "n_tgt": int(tgt.sum()), "fp": int((nor & al).sum()),
             "n_nor": int(nor.sum()), "tp_v": int((ev & (lab == 2) & al).sum()), "n_v": int((ev & (lab == 2)).sum()),
             "tp_s": int((ev & (lab == 1) & al).sum()), "n_s": int((ev & (lab == 1)).sum()),
             "tp_poor": int((tgt & bad & al).sum()), "n_poor": int((tgt & bad).sum()), "hours": hours}
        if res is not None:
            d["reviews_grouped"] = int(res["reviews_grouped"])
        if auc_s is not None:
            y = (lab[ev] > 0).astype(int)
            if 0 < y.sum() < len(y):
                from sklearn.metrics import roc_auc_score
                d["auroc"] = float(roc_auc_score(y, np.asarray(auc_s)[ev]))
        return d
    screens = {}
    for (cname, mode, q, tau) in CALIBRATIONS:
        if mode == "screen" and tau not in screens:
            screens[tau] = morphology_screen(r, tau)
        for fu in FUSIONS:
            sc = fs[fu]
            res = run_monitor(sc, r, alpha=ALPHA, mode=mode, q=q, pfusion=fu if fu in P_FUSIONS else None,
                              screen=screens.get(tau))
            out[f"{cname}|{fu}"] = cnt(res["alarm"], res, auc_score_of(sc, r))
    res = run_monitor(fs[SEL_FUSION], r, alpha=ALPHA, mode=SEL_MODE, q=SEL_Q, static=True)
    out["static"] = cnt(res["alarm"], None, fs[SEL_FUSION])
    for key in ("ecg", "early"):
        out[f"global_{key}"] = cnt((raw[key] > G["thr"][key][ALPHA]) & r["eval_mask"], None, raw[key])
        out[f"default_{key}"] = cnt((raw[key] > 0) & r["eval_mask"], None, raw[key])
    return str(r["id"]), out


def _init_worker():
    m = G["models"]
    for est in (m.ecg, m.ppg, m.joint):
        est.n_jobs = 1


def run_pool(fn, paths, workers, on_result, deadline_h):
    if workers <= 1:
        _init_worker()
        for pth in paths:
            if hrs() > deadline_h:
                return False
            on_result(fn(pth))
        return True
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        for res in pool.imap_unordered(fn, paths, chunksize=1):
            on_result(res)
            if hrs() > deadline_h:
                pool.terminate()
                return False
    return True


# --------------------------------------------------------------------------------------------- causal check
def causal_check(test_ids, recs, out_path, deadline_h):
    try:
        import vitaldb
    except Exception as e:  # noqa: BLE001
        log("causal check skipped (vitaldb package missing):", repr(e)); return
    from ecgppg.features import ecg_features
    FS = 500.0
    m = G["models"]
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for rid in test_ids:
        if rid in res:
            continue
        if hrs() > deadline_h:
            log("causal check: time budget reached"); break
        r = recs[rid]
        try:
            case = int(rid.replace("vdb", ""))
            t = np.asarray(r["t"], float)
            t0, t1 = max(0.0, t[0] - 2.0), t[-1] + 2.0
            vals = vitaldb.load_case(case, ["SNUADC/ECG_II", "SNUADC/PLETH"], 1 / FS)
            i0, i1 = int(t0 * FS), min(len(vals), int(t1 * FS))
            ecg = vals[i0:i1, 0]
            off = i0 / FS
            tr = t - off
            blk, fit = r["block"], r["fit_mask"]
            X0 = ecg_features(ecg, FS, tr, blk, fit)
            Xc = ecg_features_causal(ecg, FS, tr, blk, fit)
            ref = np.asarray(r["Xe"], float)
            d = {"repro_maxabs": float(np.max(np.abs(X0 - ref))),
                 "repro_med_rel": float(np.median(np.linalg.norm(X0[:, :64] - ref[:, :64], axis=1) /
                                                  (np.linalg.norm(ref[:, :64], axis=1) + 1e-9)))}
            variants = [("ref", ref), ("full", X0), ("causal", Xc)]
            for D in LOOKAHEADS:
                variants.append((f"bounded{D:g}", ecg_features_bounded(ecg, FS, tr, blk, fit, chunk_s=0.5, lookahead_s=D)))
            ctx = record_context(r)
            ev = r["eval_mask"]
            alarms = {}
            for tag, X in variants:
                if tag not in ("ref", "full"):
                    rel = np.linalg.norm(X[:, :64] - X0[:, :64], axis=1) / (np.linalg.norm(X0[:, :64], axis=1) + 1e-9)
                    d[f"{tag}_rel"] = [float(np.median(rel[ev])), float(np.percentile(rel[ev], 95))]
                s = PopulationModels._logit(m.ecg, X).astype(np.float32).astype(float)
                mon = run_monitor_v2(s, r, ALPHA, SEL_MODE, SEL_Q)
                d[tag] = row_for(r, mon["alarm"], ctx, mon)
                alarms[tag] = mon["alarm"]
            for tag, _ in variants[1:]:
                d[f"agree_{tag}"] = float(np.mean(alarms[tag][ev] == alarms["ref"][ev]))
                d[f"agree_full_{tag}"] = float(np.mean(alarms[tag][ev] == alarms["full"][ev]))
            rr_next = np.r_[np.diff(tr), np.nan]
            rr_next = np.where(np.isfinite(rr_next) & (rr_next < 3), rr_next, 0)
            for D in LOOKAHEADS:
                chunk_end = (np.floor(tr / 0.5) + 1) * 0.5 + D
                lat = np.maximum(chunk_end - tr, rr_next)
                d[f"lat_bounded{D:g}"] = [float(np.median(lat[ev])), float(np.percentile(lat[ev], 95))]
            d["lat_ref"] = [float(np.median(np.maximum(rr_next[ev], .45))), float(np.percentile(np.maximum(rr_next[ev], .45), 95))]
            res[rid] = d
            log(f"  causal {rid}: repro {d['repro_maxabs']:.2e}  agree bounded2 {d['agree_bounded2']:.4f} causal {d['agree_causal']:.4f}")
        except Exception as e:  # noqa: BLE001
            res[rid] = {"error": repr(e)}
            log(f"  causal {rid}: error {e!r}")
        json.dump(res, open(out_path, "w"), default=float)


# --------------------------------------------------------------------------------------------- self test
def self_test(r):
    s = scores_of(r)[SEL_FUSION]
    for kw in ({}, {"fp_rate": .05, "fn_rate": .05}, {"delay_s": 300.0}, {"avail": 0.5}):
        a = run_monitor_ext(s, r, ALPHA, SEL_MODE, SEL_Q, seed=7, **kw)["alarm"]
        b = run_monitor_v2(s, r, ALPHA, SEL_MODE, SEL_Q, seed=7, **kw)["alarm"]
        assert np.array_equal(a, b), f"run_monitor_v2 differs from run_monitor_ext for {kw}"
    e = scores_of(r)["early"]
    a = aci_monitor(e, r, G["aci_cal"]["early"], ALPHA)["alarm"]
    b = aci_monitor_v2(e, r, G["aci_cal"]["early"], ALPHA, 0.005)["alarm"]
    assert np.array_equal(a, b), "aci_monitor_v2 differs from aci_monitor"
    log("self-test passed (v2 monitors reproduce the first-revision monitors)")


# --------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="results_rev2")
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no_causal", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    workers = args.workers or max(1, (os.cpu_count() or 2))
    paths = sorted(glob.glob(os.path.join(args.cache, "vitaldb", "*.npz")))
    allrecs = [load_record(p) for p in paths]
    recs = {r["id"]: r for r in allrecs
            if r["eval_mask"].sum() > 0 and (r["label"][r["fit_mask"]] == 0).sum() >= 50}
    path_of = {r["id"]: p for r, p in zip(allrecs, paths)}
    vil_paths = sorted(glob.glob(os.path.join(args.cache, "vilnius", "*.npz")))
    log(f"VitalDB {len(recs)} records; Vilnius {len(vil_paths)} days; {workers} workers")
    tr, va, te = split_patients(list(recs), seed=SEED)
    train, val, test = [recs[i] for i in tr], [recs[i] for i in va], [recs[i] for i in te]
    models = PopulationModels(SEED).fit(train)
    G["models"] = models
    # validation thresholds (quantile of normal sinus beats) and ACI calibration sets
    thr, cal = {}, {}
    for key in ("ecg", "early"):
        v = np.concatenate([models.raw_scores(r)[key][r["eval_mask"] & (r["label"] == 0) & r["sinus"]] for r in val])
        thr[key] = {a: float(np.quantile(v, 1 - a)) for a in ALPHAS}
        cal[key] = v
    G["thr"], G["aci_cal"] = thr, cal
    json.dump({"seed": SEED, "selected": [SEL_FUSION, SEL_MODE, SEL_Q], "test": te, "val": va,
               "thr": thr, "alphas": ALPHAS, "aci_alphas": ACI_ALPHAS, "aci_gammas": ACI_GAMMAS,
               "scenarios": [s[0] for s in scenario_list()], "workers": workers},
              open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    self_test(test[0])
    # ---- T1 VitalDB test
    fn = os.path.join(args.out, "vitaldb_rows.json")
    if not os.path.exists(fn):
        acc = {}
        run_pool(analyse_record, [path_of[i] for i in te], workers,
                 lambda res: acc.__setitem__(res[0], {"rows": res[1], "desc": res[2]}), 1e9)
        json.dump(acc, open(fn, "w"), default=float)
        log("T1 VitalDB test done")
    # ---- T2 appendix (test and validation)
    fn = os.path.join(args.out, "appendix.json")
    if not os.path.exists(fn):
        app = {"test": {}, "val": {}}
        run_pool(appendix_record, [path_of[i] for i in te], workers,
                 lambda res: app["test"].__setitem__(res[0], res[1]), 1e9)
        run_pool(appendix_record, [path_of[i] for i in va], workers,
                 lambda res: app["val"].__setitem__(res[0], res[1]), 1e9)
        json.dump(app, open(fn, "w"), default=float)
        log("T2 appendix done")
    # ---- T3 causal check
    if not args.no_causal:
        causal_check(te, recs, os.path.join(args.out, "causal.json"), args.hours * 0.25)
        log("T3 causal check finished")
    # ---- T4 Vilnius (resumable)
    fn = os.path.join(args.out, "vilnius_rows.jsonl")
    done = set()
    if os.path.exists(fn):
        for ln in open(fn):
            try:
                done.add(json.loads(ln)["id"])
            except Exception:  # noqa: BLE001
                pass
    todo = [p for p in vil_paths if os.path.basename(p)[:-4] not in done]
    log(f"T4 Vilnius: {len(done)} days done, {len(todo)} to do")
    fh = open(fn, "a")
    cnt = [len(done)]

    def save(res):
        fh.write(json.dumps({"id": res[0], "rows": res[1], "desc": res[2]}, default=float) + "\n"); fh.flush()
        cnt[0] += 1
        if cnt[0] % 10 == 0:
            log(f"  Vilnius {cnt[0]} / {len(vil_paths)} days")
    finished = run_pool(analyse_record, todo, workers, save, args.hours)
    fh.close()
    n_done = sum(1 for _ in open(fn))
    status = {"vilnius_days_done": n_done, "vilnius_days_total": len(vil_paths),
              "finished": bool(finished and n_done >= len(vil_paths))}
    json.dump(status, open(os.path.join(args.out, "status.json"), "w"))
    log("status", status)


if __name__ == "__main__":
    main()
