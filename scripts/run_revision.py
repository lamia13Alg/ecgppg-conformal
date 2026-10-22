"""Major-revision analyses (all on cached features; nothing is downloaded).

  * multi-seed stability: the whole protocol (training, selection on validation, test) repeated
    for several patient splits of VitalDB, and the frozen models of each split applied to Vilnius;
  * main split (seed 42): patient-level hierarchical bootstrap CIs and paired differences, imperfect
    reviewer scenarios, alarm notifications, CUSUM tuned on validation, conditional FAR,
    distribution-shift quantification, latency (no-lookahead variant), deep-learning baseline,
    AF-trained population model, adaptive conformal inference (ACI), per-patient normalisation,
    computational cost, inclusion flow, representative case.

    python scripts/run_revision.py --cache cache --out results_rev --seeds 0,1,2,3,4,5,6,7,8,9
Results are written to results_rev/*.json (resumable: finished seeds are skipped).
"""
import argparse
import glob
import zlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ecgppg.pipeline import load_record, LazyRecords, CALIBRATIONS, evaluate, select_configuration  # noqa
from ecgppg.scores import PopulationModels, FUSIONS, P_FUSIONS, fused_scores, stack_training, _rf  # noqa
from ecgppg.experiments import split_patients  # noqa
from ecgppg.metrics import eval_hours  # noqa
from ecgppg.conformal import run_monitor  # noqa
from ecgppg.revision import (run_monitor_ext, aci_monitor, per_patient_normalised, notifications,  # noqa
                             cusum_alarm, episodes_metrics, tune_cusum_h, bootstrap)

ALPHA = 0.01
T0 = time.time()


def hrs():
    return (time.time() - T0) / 3600


def log(*a):
    print(f"[{hrs():5.2f} h]", *a, flush=True)


# --------------------------------------------------------------------------------------------- helpers
def counts(r, alarm, target=(1, 2), ev_labels=(0, 1, 2), sinus_only=True):
    lab = r["label"]
    ev = r["eval_mask"] & np.isin(lab, ev_labels)
    tgt = ev & np.isin(lab, target)
    nor = ev & (lab == 0) & (r["sinus"] if sinus_only else True)
    return {"tp": int((tgt & alarm).sum()), "n_tgt": int(tgt.sum()),
            "fp": int((nor & alarm).sum()), "n_nor": int(nor.sum())}


def summarise(rows, hours=None):
    tp = sum(x["tp"] for x in rows); nt = sum(x["n_tgt"] for x in rows)
    fp = sum(x["fp"] for x in rows); nn = sum(x["n_nor"] for x in rows)
    far = np.array([x["fp"] / x["n_nor"] for x in rows if x["n_nor"] > 0]) * 100
    out = {"se": 100 * tp / nt if nt else float("nan"), "far": 100 * fp / nn if nn else float("nan"),
           "ppv": 100 * tp / (tp + fp) if tp + fp else float("nan"),
           "far_p90": float(np.percentile(far, 90)) if len(far) else float("nan"),
           "far_max": float(far.max()) if len(far) else float("nan"),
           "pct_above_2a": float(100 * np.mean(far > 2)) if len(far) else float("nan"),
           "n_tgt": nt, "n_nor": nn, "n_records": len(rows)}
    return out


def quantile_thr(vals, alpha=ALPHA):
    v = np.concatenate(vals)
    return float(np.quantile(v, 1 - alpha))


def val_normal_scores(val, fn):
    return [fn(r)[r["eval_mask"] & (r["label"] == 0) & r["sinus"]] for r in val]


def sel_score(models, r, fusion):
    return fused_scores(models.raw_scores(r), r)[fusion]


def single(score):
    return score if not isinstance(score, tuple) else None


def patient_of(rid):
    rid = str(rid)
    return rid.split("_d")[0] if rid.startswith("vil") else rid


class ECGOnlyRF:
    """RF on ECG features only, for a given target (AF) or with no-lookahead features."""

    def __init__(self, seed, target_labels=(1, 2), neg_labels=(0,), zero_cols_e=(), zero_cols_p=(),
                 use_ppg=False):
        self.seed, self.tl, self.nl = seed, target_labels, neg_labels
        self.ze, self.zp, self.use_ppg = list(zero_cols_e), list(zero_cols_p), use_ppg
        self._cache = {}

    def _X(self, r):
        Xe = np.array(r["Xe"], float)
        if self.ze:
            Xe[:, self.ze] = 0
        if not self.use_ppg:
            return Xe
        Xp = np.array(r["Xp"], float)
        if self.zp:
            Xp[:, self.zp] = 0
        return np.hstack([Xe, Xp])

    def fit(self, records, max_per_record=4000):
        rng = np.random.default_rng(self.seed)
        X, y = [], []
        for r in records:
            lab = r["label"]
            idx = np.where(np.isin(lab, self.tl + self.nl))[0]
            if len(idx) > max_per_record:
                pos = idx[np.isin(lab[idx], self.tl)]
                neg = idx[np.isin(lab[idx], self.nl)]
                neg = rng.choice(neg, size=min(len(neg), max(0, max_per_record - len(pos))), replace=False)
                idx = np.sort(np.r_[pos, neg])
            if len(idx):
                X.append(self._X(r)[idx]); y.append(np.isin(lab[idx], self.tl).astype(int))
        self.m = _rf(self.seed).fit(np.vstack(X), np.concatenate(y))
        return self

    def logit(self, r):
        k = str(r["id"])
        if k not in self._cache:
            p = np.clip(self.m.predict_proba(self._X(r))[:, 1], 1e-4, 1 - 1e-4)
            self._cache[k] = np.log(p / (1 - p)).astype(np.float32)
        return self._cache[k].astype(float)


# --------------------------------------------------------------------------------------------- one record
def run_record(r, ctx, full, dataset):
    """Evaluate every method on one record; returns dict method -> counts (+ extras)."""
    sel_fu, sel_mode, sel_q = ctx["sel"]
    out, extra = {}, {}
    target = (1, 2)
    raw = ctx["models"].raw_scores(r)
    s_sel = sel_score(ctx["models"], r, sel_fu)
    pf = sel_fu if sel_fu in P_FUSIONS else None
    res = run_monitor_ext(s_sel, r, ALPHA, sel_mode, sel_q, pfusion=pf)
    out["selected"] = counts(r, res["alarm"], target)
    out["selected"]["bound_sum"] = float(np.nansum(res["bound"]))
    out["selected"]["bound_n"] = int(np.isfinite(res["bound"]).sum())
    out["selected"]["reviews"] = res["reviews"]; out["selected"]["reviews_grouped"] = res["reviews_grouped"]
    out["selected"]["startup"] = res["startup_reviews"]
    if sel_fu != "ecg":
        out["ecg_samecal"] = counts(r, run_monitor_ext(raw["ecg"], r, ALPHA, sel_mode, sel_q)["alarm"], target)
    out["labelfree"] = counts(r, run_monitor_ext(s_sel, r, ALPHA, "labelfree", 0.0, pfusion=pf)["alarm"], target)
    ev = r["eval_mask"]
    for key in ("early", "ecg"):
        out[f"pop_default_{key}"] = counts(r, (raw[key] > 0) & ev, target)
        out[f"pop_global_{key}"] = counts(r, (raw[key] > ctx["thr"][key]) & ev, target)
    if not full:
        return out, extra
    # ---------- full analyses (main split) ----------
    out["static"] = counts(r, run_monitor_ext(s_sel, r, ALPHA, sel_mode, sel_q, pfusion=pf, static=True)["alarm"], target)
    pkey = "early" if sel_fu not in ("ecg",) else "ecg"
    _, al = per_patient_normalised(raw[pkey], r, ctx["thr_norm"][pkey])
    out[f"pernorm_{pkey}"] = counts(r, al, target)
    out["aci_early"] = counts(r, aci_monitor(raw["early"], r, ctx["aci_cal"], ALPHA)["alarm"], target)
    if ctx.get("cnn") is not None:
        c = ctx["cnn"].logit(r)
        out["cnn_conformal"] = counts(r, run_monitor_ext(c, r, ALPHA, sel_mode, sel_q)["alarm"], target)
        out["cnn_default"] = counts(r, (c > 0) & ev, target)
        out["cnn_global"] = counts(r, (c > ctx["thr"]["cnn"]) & ev, target)
    nl = ctx["nolook"].logit(r)
    out["nolook_conformal"] = counts(r, run_monitor_ext(nl, r, ALPHA, sel_mode, sel_q)["alarm"], target)
    # AF-trained population model (target: AF beats; label 3 in VitalDB, label 1 in Vilnius)
    af = ctx["afrf"].logit(r)
    af_t = (1,) if dataset == "vilnius" else (3,)
    af_ev = (0, 1) if dataset == "vilnius" else (0, 3)
    out["af_pop_default"] = counts(r, (af > 0) & ev, af_t, af_ev)
    out["af_pop_global"] = counts(r, (af > ctx["thr"]["af"]) & ev, af_t, af_ev)
    out["af_conformal"] = counts(r, run_monitor_ext(af, r, ALPHA, sel_mode, sel_q)["alarm"], af_t, af_ev)
    if dataset == "vitaldb":   # AF detection by the selected (ectopy) configuration, for reference
        out["selected_on_af"] = counts(r, res["alarm"], (3,), (0, 3))
    # reviewer scenarios
    for name, kw in ctx["scenarios"]:
        o = run_monitor_ext(s_sel, r, ALPHA, sel_mode, sel_q, pfusion=pf, seed=zlib.crc32(f"{name}|{r['id']}".encode()), **kw)
        row = counts(r, o["alarm"], target)
        row["bound_sum"] = float(np.nansum(o["bound"])); row["bound_n"] = int(np.isfinite(o["bound"]).sum())
        row["excluded_normal"] = o.get("excluded_normal", 0); row["reviews_grouped"] = o["reviews_grouped"]
        out["rev_" + name] = row
    # notifications (alarm events presented to staff, 60-s refractory)
    h = eval_hours(r)
    for m, al in [("selected", res["alarm"]), ("pop_default_early", (raw["early"] > 0) & ev),
                  ("pop_global_early", (raw["early"] > ctx["thr"]["early"]) & ev)]:
        nt, nf = notifications(al, r["t"], r["label"], ev, 60.0, target)
        extra[f"notif_{m}"] = (nt, nf, h)
    # episodes: CUSUM with threshold tuned on validation, 60-beat refractory
    pos = np.where(ev)[0]
    abn = np.isin(r["label"][pos], target)
    for tag, hh, refr in [("tuned", ctx["cusum_h"], 60), ("default", float(np.log(4000)), 0)]:
        al = cusum_alarm(res["p"][pos], 0.2, hh, refr)
        for ml in ctx["ep_lens"]:
            ne, nd, dl, nfa = episodes_metrics(abn, al, r["t"][pos], min_len=ml)
            extra[f"ep_{tag}_{ml}"] = (ne, nd, dl, nfa, h)
    # conditional FAR: by time since start (hours), PPG quality, ECG quality, for two methods
    nor = ev & (r["label"] == 0) & r["sinus"]
    tsince = (r["t"] - r["t"][0]) / 3600.0
    tb = np.minimum((tsince // ctx["tbin_h"]).astype(int), ctx["n_tbins"] - 1)
    sq = np.asarray(r["sqi"], float)
    qb = np.digitize(sq, [0.01, 0.5, 0.8])     # 0: no PPG, 1: <0.5, 2: 0.5-0.8, 3: >0.8
    bad = np.asarray(r["ecg_bad"], bool).astype(int)
    for m, al in [("selected", res["alarm"]), ("pop_default_early", (raw["early"] > 0) & ev)]:
        extra[f"cond_time_{m}"] = [(int((nor & (tb == b) & al).sum()), int((nor & (tb == b)).sum())) for b in range(ctx["n_tbins"])]
        extra[f"cond_sqi_{m}"] = [(int((nor & (qb == b) & al).sum()), int((nor & (qb == b)).sum())) for b in range(4)]
        extra[f"cond_ecgq_{m}"] = [(int((nor & (bad == b) & al).sum()), int((nor & (bad == b)).sum())) for b in range(2)]
    extra["burden"] = float(np.isin(r["label"][ev], target).mean()) if ev.any() else 0.0
    # decision latency after the R peak: the next R peak is needed (masking, pause features), and
    # the ECG (PPG) window ends 0.45 s (0.80 s) after the R peak
    rr_next = np.r_[np.diff(r["t"]), np.nan][ev]
    rr_next = rr_next[np.isfinite(rr_next) & (rr_next < 3)]
    if len(rr_next):
        extra["lat_ecg"] = [float(np.median(np.maximum(rr_next, .45))), float(np.percentile(np.maximum(rr_next, .45), 95))]
        extra["lat_ppg"] = [float(np.median(np.maximum(rr_next, .80))), float(np.percentile(np.maximum(rr_next, .80), 95))]
    extra["hours"] = h
    return out, extra


# --------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="results_rev")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--main_seed", type=int, default=42)
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--no_cnn", action="store_true")
    ap.add_argument("--vilnius_seeds", type=int, default=4,
                    help="number of non-main splits also applied to Vilnius")
    ap.add_argument("--vilnius_every", type=int, default=3,
                    help="for non-main splits, use every k-th Vilnius day")
    ap.add_argument("--ann", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.cache, "vitaldb", "*.npz")))
    allrecs = [load_record(p) for p in paths]
    recs = {r["id"]: r for r in allrecs
            if r["eval_mask"].sum() > 0 and (r["label"][r["fit_mask"]] == 0).sum() >= 50}
    vil_paths = sorted(glob.glob(os.path.join(args.cache, "vilnius", "*.npz")))
    vil = LazyRecords(vil_paths)
    log(f"VitalDB {len(recs)} records ({len(allrecs)} cached); Vilnius {len(vil_paths)} days")
    # ---------------- inclusion flow ----------------
    flow = {"vitaldb_cached": len(allrecs), "vitaldb_included": len(recs)}
    if args.ann:
        flow["vitaldb_annotated"] = len(glob.glob(os.path.join(args.ann, "Annotation_file_*.csv")))
    exf = os.path.join(args.cache, "vitaldb", "excluded.txt")
    if os.path.exists(exf):
        reasons = [ln.split("\t")[1].split(" ")[0] if "\t" in ln else "?" for ln in open(exf).read().splitlines() if ln.strip()]
        ids = {ln.split("\t")[0] for ln in open(exf).read().splitlines() if ln.strip()}
        flow["vitaldb_excluded_lines"] = len(reasons)
        flow["vitaldb_excluded_unique"] = len(ids)
        flow["vitaldb_excluded_reasons"] = {k: reasons.count(k) for k in set(reasons)}
    nb_vil = beats_vil = af_vil = ppg_beats = 0
    pats = set()
    for r in vil:
        nb_vil += 1; beats_vil += len(r["t"]); af_vil += int((r["label"] == 1).sum())
        ppg_beats += int((np.asarray(r["sqi"]) != 0).sum()); pats.add(patient_of(r["id"]))
    flow.update({"vilnius_days": nb_vil, "vilnius_patients": len(pats), "vilnius_beats": beats_vil,
                 "vilnius_af_beats": af_vil, "vilnius_ppg_coverage_pct": 100 * ppg_beats / max(beats_vil, 1)})
    json.dump(flow, open(os.path.join(args.out, "flow.json"), "w"), indent=1)
    log("flow", flow)

    seeds = [args.main_seed] + [int(s) for s in args.seeds.split(",") if s.strip() and int(s) != args.main_seed]
    for si, seed in enumerate(seeds):
        fn = os.path.join(args.out, f"seed_{seed}.json")
        if os.path.exists(fn):
            log(f"seed {seed}: already done"); continue
        if hrs() > args.hours:
            log("time budget reached; rerun to continue"); break
        full = seed == args.main_seed
        log(f"=== seed {seed} ({'main, full analyses' if full else 'stability'}) ===")
        tr, va, te = split_patients(list(recs), seed=seed)
        train, val, test = [recs[i] for i in tr], [recs[i] for i in va], [recs[i] for i in te]
        models = PopulationModels(seed).fit(train)
        dev = evaluate(val, models, FUSIONS, CALIBRATIONS, alpha=ALPHA, with_cusum=False)
        sk = select_configuration(dev, ALPHA)
        cal = next(c for c in CALIBRATIONS if c[0] == sk[0])
        log(f"selected {sk}")
        ctx = {"models": models, "sel": (sk[1], cal[1], cal[2]),
               "thr": {k: quantile_thr(val_normal_scores(val, lambda r, k=k: models.raw_scores(r)[k])) for k in ("early", "ecg")}}
        res = {"seed": seed, "selected": list(sk), "split": {"train": tr, "val": va, "test": te},
               "dev_selected": dev[sk], "dev_ecg_samecal": dev[(sk[0], "ecg")], "dev_early_samecal": dev[(sk[0], "early")]}
        if full:
            t1 = time.time()
            # extra models
            ctx["nolook"] = ECGOnlyRF(seed, (1, 2), (0,), zero_cols_e=[65], zero_cols_p=[4],
                                      use_ppg=(sk[1] not in ("ecg",))).fit(train)
            ctx["afrf"] = ECGOnlyRF(seed, (3,), (0,)).fit(train)
            ctx["thr"]["af"] = quantile_thr([ctx["afrf"].logit(r)[r["eval_mask"] & (r["label"] == 0) & r["sinus"]] for r in val])
            ctx["cnn"] = None
            if not args.no_cnn:
                try:
                    from ecgppg.deep import CNNModel
                    Xe, Xp, y = stack_training(train, seed=seed)
                    ctx["cnn"] = CNNModel(seed).fit(Xe, Xp, y)
                    ctx["thr"]["cnn"] = quantile_thr([ctx["cnn"].logit(r)[r["eval_mask"] & (r["label"] == 0) & r["sinus"]] for r in val])
                except Exception as e:  # noqa: BLE001
                    log("CNN skipped:", repr(e))
            # per-patient normalisation threshold on validation (z-scores of normal sinus beats)
            ctx["thr_norm"] = {}
            for k in ("early", "ecg"):
                zs = []
                for r in val:
                    z, _ = per_patient_normalised(models.raw_scores(r)[k], r, 0.0)
                    zs.append(z[r["eval_mask"] & (r["label"] == 0) & r["sinus"]])
                ctx["thr_norm"][k] = quantile_thr(zs)
            ctx["aci_cal"] = np.concatenate(val_normal_scores(val, lambda r: models.raw_scores(r)["early"]))
            # CUSUM threshold tuned on validation (<= 1 false episode alarm per hour, 60-beat refractory)
            pl, al_, tl = [], [], []
            for r in val:
                o = run_monitor_ext(sel_score(models, r, sk[1]), r, ALPHA, cal[1], cal[2],
                                    pfusion=sk[1] if sk[1] in P_FUSIONS else None)
                pos = np.where(r["eval_mask"])[0]
                pl.append(o["p"][pos]); al_.append(np.isin(r["label"][pos], (1, 2))); tl.append(r["t"][pos])
            ctx["cusum_h"] = tune_cusum_h(pl, al_, tl, 1.0)
            res["cusum_h"] = ctx["cusum_h"]
            ctx["scenarios"] = [
                ("oracle", {}),
                ("err5", {"fp_rate": .05, "fn_rate": .05}), ("err10", {"fp_rate": .10, "fn_rate": .10}),
                ("err20", {"fp_rate": .20, "fn_rate": .20}),
                ("fp10", {"fp_rate": .10}), ("fn10", {"fn_rate": .10}),
                ("delay60", {"delay_s": 60.0}), ("delay300", {"delay_s": 300.0}),
                ("avail50", {"avail": 0.5}), ("auto", {"auto_exclude": True})]
            res["models_time_s"] = time.time() - t1
            # cost: per-beat inference and monitor time on one test record
            r0 = max(test, key=lambda r: len(r["t"]))
            ta = time.time(); models._cache.clear(); models.raw_scores(r0); tb_ = time.time()
            run_monitor_ext(sel_score(models, r0, sk[1]), r0, ALPHA, cal[1], cal[2]); tc = time.time()
            res["cost_us_per_beat"] = {"rf_scoring": 1e6 * (tb_ - ta) / len(r0["t"]),
                                       "conformal_update": 1e6 * (tc - tb_) / len(r0["t"])}
        vil_seed = vil if full else LazyRecords(vil_paths[::args.vilnius_every])
        res["vilnius_every"] = 1 if full else args.vilnius_every
        for dataset, records, tbin, nbins, ep_lens in [("vitaldb", test, 5 / 60, 4, [3]),
                                                        ("vilnius", vil_seed, 4.0, 6, [30, 120, 300])]:
            if dataset == "vilnius" and not full and si > args.vilnius_seeds:
                continue
            if dataset == "vilnius" and len(vil_paths) == 0:
                continue
            ctx.update({"tbin_h": tbin, "n_tbins": nbins, "ep_lens": ep_lens})
            rows, extras, ids = {}, [], []
            for r in records:
                o, e = run_record(r, ctx, full, dataset)
                for m, row in o.items():
                    rows.setdefault(m, []).append(row)
                extras.append(e); ids.append(str(r["id"]))
                if full and dataset == "vilnius" and len(ids) % 25 == 0:
                    log(f"  vilnius {len(ids)} days")
            groups = [patient_of(i) for i in ids]
            d = {"summary": {m: summarise(rr) for m, rr in rows.items()}, "ids": ids, "rows": rows}
            if full:
                d["extras"] = extras
                d["bootstrap"] = bootstrap(groups, rows, n_boot=2000, seed=seed, paired_with="selected")
            res[dataset] = d
            log(f"  {dataset}: selected {d['summary']['selected']['se']:.1f}/{d['summary']['selected']['far']:.2f}"
                f"  pop_default_early {d['summary']['pop_default_early']['se']:.1f}/{d['summary']['pop_default_early']['far']:.2f}")
            # free RF score cache of large external sets between datasets
            if dataset == "vilnius":
                models._cache.clear()
        if full:
            res.update(shift_analysis(models, test, vil, seed))
            res["case"] = case_series(ctx, vil)
        json.dump(res, open(fn, "w"), default=float)
        log(f"seed {seed} saved")
    log("done")


def shift_analysis(models, test, vil, seed, n=20000):
    """Domain classifier AUROC (VitalDB test vs Vilnius, normal beats) and score shift."""
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(seed)

    def sample(records, k, every=1):
        Xe, Xp, s = [], [], []
        for j, r in enumerate(records):
            if j % every:
                continue
            m = np.where(r["eval_mask"] & (r["label"] == 0) & (np.asarray(r["sqi"]) != 0))[0]
            if len(m) == 0:
                continue
            m = rng.choice(m, size=min(len(m), k), replace=False)
            sub = {"id": "tmp", "Xe": r["Xe"][m], "Xp": r["Xp"][m]}
            Xe.append(r["Xe"][m]); Xp.append(r["Xp"][m]); s.append(models._raw_scores(sub)["early"])
        return np.vstack(Xe), np.vstack(Xp), np.concatenate(s)
    a = sample(test, 500); b = sample(vil, 300, every=3)
    ia = rng.choice(len(a[0]), size=min(n, len(a[0])), replace=False)
    ib = rng.choice(len(b[0]), size=min(n, len(b[0])), replace=False)
    out = {}
    for name, k in [("ecg", 0), ("ppg", 1)]:
        X = np.vstack([a[k][ia], b[k][ib]]); y = np.r_[np.zeros(len(ia)), np.ones(len(ib))]
        clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=seed)
        out[f"domain_auroc_{name}"] = float(np.mean(cross_val_score(clf, X, y, cv=5, scoring="roc_auc")))
    out["score_normal_vitaldb_q"] = [float(np.percentile(a[2], q)) for q in (50, 90, 99)]
    out["score_normal_vilnius_q"] = [float(np.percentile(b[2], q)) for q in (50, 90, 99)]
    out["frac_normal_above0_vitaldb"] = float(np.mean(a[2] > 0))
    out["frac_normal_above0_vilnius"] = float(np.mean(b[2] > 0))
    return {"shift": out}


def case_series(ctx, vil, span_h=3.0):
    """Representative Vilnius day: the first day with an AF episode starting after the patient
    window; arrays saved around its onset for the case figure."""
    sel_fu, mode, q = ctx["sel"]
    from ecgppg.metrics import episodes
    for r in vil:
        ev = np.where(r["eval_mask"])[0]
        eps = episodes(r["label"][ev] == 1, min_len=120)
        if not eps:
            continue
        onset = r["t"][ev[eps[0][0]]]
        if onset - r["t"][0] < 3600:
            continue
        s = sel_score(ctx["models"], r, sel_fu)
        o = run_monitor_ext(s, r, ALPHA, mode, q)
        pop = ctx["models"].raw_scores(r)["early"]
        w = (r["t"] > onset - span_h * 3600 / 2) & (r["t"] < onset + span_h * 3600 / 2)
        rr = np.r_[np.nan, np.diff(r["t"])]
        return {"id": str(r["id"]), "onset": float(onset), "t": r["t"][w].tolist(), "rr": rr[w].tolist(),
                "label": r["label"][w].tolist(), "score": np.asarray(s)[w].tolist() if not isinstance(s, tuple) else [],
                "p": o["p"][w].tolist(), "alarm": o["alarm"][w].astype(int).tolist(),
                "bound": o["bound"][w].tolist(), "pop": pop[w].tolist(), "thr_pop_default": 0.0}
    return {}


if __name__ == "__main__":
    main()
