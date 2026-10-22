"""End-to-end protocol: development and selection on the VitalDB validation patients, single
evaluation on the held-out VitalDB test patients and on external databases. Writes JSON results,
LaTeX table bodies (results/tab_*.tex) and LaTeX macros (results/macros.tex) that the manuscript
inputs directly, so that no number is copied by hand."""
import json
import os
import numpy as np

from .scores import PopulationModels, FUSIONS, P_FUSIONS, fused_scores
from .pipeline import CALIBRATIONS, evaluate, select_configuration, auc_score_of
from .conformal import run_monitor, conformal_cusum
from .metrics import record_metrics, pool

FUSION_NAMES = {"ecg": "ECG-RF", "ppg": "PPG-RF", "early": "Joint-RF (early)",
                "late": "Late (min tail)", "gated": "SQI-gated late",
                "pbonf": "p-Bonferroni", "pmean": "p-mean ($2\\bar p$)"}
CALIB_NAMES = {"labelfree": "Label-free, no screen", "screen6": "Label-free, screen $\\tau=6$",
               "adj0": "Adjudicated, start-up 0\\%", "adj5": "Adjudicated, start-up 5\\%",
               "adj10": "Adjudicated, start-up 10\\%"}


def split_patients(ids, seed=42, frac=(0.70, 0.15, 0.15)):
    ids = sorted(ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n1 = int(round(frac[0] * len(ids)))
    n2 = int(round((frac[0] + frac[1]) * len(ids)))
    return ([ids[i] for i in perm[:n1]], [ids[i] for i in perm[n1:n2]],
            [ids[i] for i in perm[n2:]])


def threshold_rows(records, models, key, thr, alpha):
    rows = []
    for r in records:
        s = models.raw_scores(r)[key]
        res = {"p": np.where(r["eval_mask"], 0.5, np.nan), "alarm": s > thr,
               "bound": np.full(len(s), np.nan), "reviews": 0, "reviews_grouped": 0,
               "startup_reviews": 0}
        res["alarm"] &= r["eval_mask"]
        rows.append(record_metrics(r, res, s, None))
    return pool(rows, alpha), rows


def population_threshold(records, models, key, alpha):
    """Threshold giving a FAR of alpha on normal sinus beats of the given (validation) records."""
    vals = []
    for r in records:
        s = models.raw_scores(r)[key]
        m = r["eval_mask"] & (r["label"] == 0) & r["sinus"]
        vals.append(s[m])
    v = np.concatenate(vals)
    if len(v) == 0:  # no sinus-rhythm label available: fall back on all normal beats
        v = np.concatenate([models.raw_scores(r)[key][r["eval_mask"] & (r["label"] == 0)]
                            for r in records])
    return float(np.quantile(v, 1 - alpha))


def static_rows(records, models, fusion, cal, alpha):
    cname, mode, q, tau = cal
    rows = []
    for r in records:
        sc = fused_scores(models.raw_scores(r), r)[fusion]
        res = run_monitor(sc, r, alpha=alpha, mode=mode, q=q,
                          pfusion=fusion if fusion in P_FUSIONS else None, static=True)
        rows.append(record_metrics(r, res, auc_score_of(sc, r), conformal_cusum(res["p"])[0]))
    return pool(rows, alpha)


def _f(x, nd=1):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _macro_name(*parts):
    name = "".join(p[:1].upper() + p[1:] for p in parts)
    return "res" + "".join(c for c in name if c.isalpha())


class Out:
    def __init__(self, outdir):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.macros = {}
        self.json = {}

    def macro(self, value, *parts, nd=1):
        self.macros[_macro_name(*parts)] = value if isinstance(value, str) else _f(value, nd)

    def table(self, name, lines):
        with open(os.path.join(self.outdir, f"tab_{name}.tex"), "w") as fh:
            fh.write("\n".join(lines) + "\n")

    def save(self, tag):
        path = os.path.join(self.outdir, f"macros_{tag}.tex")
        with open(path, "w") as fh:
            for k, v in sorted(self.macros.items()):
                fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
        with open(os.path.join(self.outdir, f"results_{tag}.json"), "w") as fh:
            json.dump(self.json, fh, indent=1, default=float)


def _key(k):
    return f"{k[0]}|{k[1]}"


def run_vitaldb_protocol(train, val, test, outdir, alpha=0.01, seed=42, tag="vdb",
                         externals=None):
    """externals: optional dict name -> (records, ep_min_len, extended) evaluated once with the
    frozen models and configuration; records may be a LazyRecords object (loaded from disk)."""
    out = Out(outdir)
    models = PopulationModels(seed).fit(train)

    # 1. development on validation patients -------------------------------------------------
    dev = evaluate(val, models, FUSIONS, CALIBRATIONS, alpha=alpha)
    sel = select_configuration(dev, alpha)
    sel_cal = next(c for c in CALIBRATIONS if c[0] == sel[0])
    out.json["dev"] = {_key(k): v for k, v in dev.items()}
    out.json["selected"] = list(sel)
    lines = []
    for c in CALIBRATIONS:
        for fu in FUSIONS:
            m = dev[(c[0], fu)]
            tick = "\\checkmark" if (c[0], fu) == sel else ""
            lines.append(f"{CALIB_NAMES[c[0]]} & {FUSION_NAMES[fu]} & {_f(m['auroc'],3)} & "
                         f"{_f(m['se'])} & {_f(m['far_sinus'],2)} & {_f(m['rev_h'],0)} & {tick} \\\\")
        lines.append("\\midrule")
    out.table(f"{tag}_dev", lines[:-1])
    out.macro(CALIB_NAMES[sel[0]], tag, "selcal")
    out.macro(FUSION_NAMES[sel[1]], tag, "selfusion")
    out.macro(dev[sel]["se"], tag, "dev", "sel", "se")
    out.macro(dev[sel]["far_sinus"], tag, "dev", "sel", "far", nd=2)
    out.macro(str(len(train)), tag, "ntrain")
    out.macro(str(len(val)), tag, "nval")
    out.macro(str(len(test)), tag, "ntest")

    # 2. single evaluation on test patients ---------------------------------------------------
    def evaluate_once(records, name, ep_min_len=3, extended=True, light=False):
        o = {}
        # light: fewer configurations for very large external sets
        fus_list = list(dict.fromkeys([sel[1], "ecg", "ppg"])) if light else FUSIONS
        cal_list = ([c for c in CALIBRATIONS if c[0] in ("labelfree", "screen6")] + [sel_cal]
                    if light else CALIBRATIONS)
        a, rows_a = evaluate(records, models, fus_list, [sel_cal], alpha=alpha,
                             ep_min_len=ep_min_len, keep_rows=True)
        b = evaluate(records, models, [sel[1]], cal_list, alpha=alpha, ep_min_len=ep_min_len)
        o.update({_key(k): v for k, v in a.items()})
        o.update({_key(k): v for k, v in b.items()})
        o["static"] = static_rows(records, models, sel[1], sel_cal, alpha)
        perrec = {"selected": [row["far_sinus"] for row in rows_a[sel]],
                  "ecg_only": [row["far_sinus"] for row in rows_a[(sel[0], "ecg")]]}
        for key in ["ecg", "early"]:
            thr = population_threshold(val, models, key, alpha)
            o[f"global_{key}"], rg = threshold_rows(records, models, key, thr, alpha)
            o[f"default_{key}"], rd = threshold_rows(records, models, key, 0.0, alpha)
            perrec[f"global_{key}"] = [row["far_sinus"] for row in rg]
            perrec[f"default_{key}"] = [row["far_sinus"] for row in rd]
        o["perrecord_far"] = perrec
        # beat-level table
        L = []
        for fu in fus_list:
            m = a[(sel[0], fu)]
            L.append(f"{FUSION_NAMES[fu]}{' (selected)' if fu == sel[1] else ''} & {_f(m['auroc'],3)} & "
                     f"{_f(m['se'])} & {_f(m['se_v'])} & {_f(m['se_s'])} & {_f(m['ppv'])} & "
                     f"{_f(m['far_sinus'],2)} & {_f(m['se_badecg'])} & {_f(m['rev_grouped_h'],0)} \\\\")
        L.append("\\midrule")
        for c in cal_list:
            if c[0] == sel[0]:
                continue
            m = b[(c[0], sel[1])]
            L.append(f"{CALIB_NAMES[c[0]]} & -- & {_f(m['se'])} & {_f(m['se_v'])} & {_f(m['se_s'])} & "
                     f"{_f(m['ppv'])} & {_f(m['far_sinus'],2)} & {_f(m['se_badecg'])} & "
                     f"{_f(m['rev_grouped_h'],0)} \\\\")
        m = o["static"]
        L.append(f"Selected, static calibration & -- & {_f(m['se'])} & {_f(m['se_v'])} & {_f(m['se_s'])} & "
                 f"{_f(m['ppv'])} & {_f(m['far_sinus'],2)} & {_f(m['se_badecg'])} & -- \\\\")
        for key, lab in [("ecg", "ECG-RF"), ("early", "Joint-RF")]:
            for kind, kl in [("global", "global threshold"), ("default", "$P>0.5$")]:
                m = o[f"{kind}_{key}"]
                L.append(f"{lab}, {kl} & {_f(m['auroc'],3)} & {_f(m['se'])} & {_f(m['se_v'])} & "
                         f"{_f(m['se_s'])} & {_f(m['ppv'])} & {_f(m['far_sinus'],2)} & "
                         f"{_f(m['se_badecg'])} & -- \\\\")
        out.table(f"{name}_beat", L)
        # per-record FAR distribution
        L = []
        rows_cfg = [("Selected configuration", a[sel]),
                    ("Same calibration, ECG-RF only", a[(sel[0], 'ecg')]),
                    ("Label-free screen, selected fusion", b[('screen6', sel[1])]),
                    ("Joint-RF, global threshold", o['global_early']),
                    ("Joint-RF, $P>0.5$", o['default_early'])]
        for lab, m in rows_cfg:
            L.append(f"{lab} & {_f(m['se'])} & {_f(m['far_sinus'],2)} & {_f(m['far_median'],2)} & "
                     f"{_f(m['far_p90'],2)} & {_f(m['far_max'],2)} & {_f(m['pct_above_2a'],0)} & "
                     f"{_f(m['pct_below_a2'],0)} \\\\")
        out.table(f"{name}_far", L)
        # episodes
        L = []
        for fu in fus_list:
            m = a[(sel[0], fu)]
            L.append(f"{FUSION_NAMES[fu]} & {_f(m.get('ep_se'))} & {_f(m.get('ep_delay_beats'),0)} & "
                     f"{_f(m.get('ep_delay_s'))} & {_f(m.get('ep_false_h'),2)} \\\\")
        out.table(f"{name}_ep", L)
        # macros for the selected configuration and key comparators
        s = a[sel]
        for k in ["auroc", "se", "se_v", "se_s", "ppv", "far_sinus", "bound", "rev_h", "rev_grouped_h",
                  "startup_per_patient", "far_median", "far_p90", "far_max", "pct_above_2a",
                  "se_badecg", "far_badecg", "se_goodecg", "ep_se", "ep_delay_s", "ep_false_h", "n_ep"]:
            nd = 3 if k == "auroc" else (2 if k.startswith("far") or k == "bound" else
                                         (0 if k in ("rev_h", "rev_grouped_h", "n_ep",
                                                     "startup_per_patient", "pct_above_2a") else 1))
            out.macro(s.get(k, float("nan")), name, "sel", k.replace("_", ""), nd=nd)
        e = a[(sel[0], "ecg")]
        for k in ["se", "far_sinus", "se_badecg", "far_badecg", "ep_se"]:
            out.macro(e.get(k, float("nan")), name, "ecgonly", k.replace("_", ""),
                      nd=2 if k.startswith("far") else 1)
        lf = b[("labelfree", sel[1])]
        out.macro(lf["se"], name, "labelfree", "se")
        out.macro(lf["far_sinus"], name, "labelfree", "far", nd=2)
        sc = b[("screen6", sel[1])]
        out.macro(sc["se"], name, "screen", "se")
        out.macro(sc["far_sinus"], name, "screen", "far", nd=2)
        g = o["global_early"]
        out.macro(g["se"], name, "global", "se")
        out.macro(g["far_sinus"], name, "global", "far", nd=2)
        out.macro(g["far_max"], name, "global", "farmax", nd=2)
        d = o["default_early"]
        out.macro(d["se"], name, "default", "se")
        out.macro(d["far_sinus"], name, "default", "far", nd=2)
        out.macro(d["far_max"], name, "default", "farmax", nd=2)
        out.macro(str(len(records)), name, "nrecords")
        # contamination of the initial calibration buffer (abnormal fraction), per record
        cf = []
        for r in records:
            c = r["cal_mask"] & (r["label"] >= 0)
            if c.sum():
                cf.append(100.0 * np.mean(r["label"][c] > 0))
        cf = np.array(cf)
        if len(cf):
            out.macro(float(np.median(cf)), name, "calcontam", "median")
            out.macro(float(100 * np.mean(cf > 100 * alpha)), name, "calcontam", "pctabove", nd=0)
            o["calibration_contamination_pct"] = cf.tolist()
        out.macro(s["hours"], name, "hours", nd=1)
        out.json[name] = {k: v for k, v in o.items()}
        if not extended:
            return o
        # alpha sweep
        sweep = {}
        for al in [0.005, 0.01, 0.02, 0.03, 0.05]:
            r1 = evaluate(records, models, [sel[1]], [sel_cal, CALIBRATIONS[0], CALIBRATIONS[1]],
                          alpha=al, with_cusum=False)
            sweep[al] = {_key(k): v for k, v in r1.items()}
        o["alpha_sweep"] = sweep
        # buffer-size ablation
        abl = {}
        for M in [100, 300, 600]:
            r1 = evaluate(records, models, [sel[1]], [sel_cal], alpha=alpha, with_cusum=False, M=M)
            abl[M] = r1[sel]
        o["buffer_ablation"] = abl
        L = [f"{M} & {_f(m['se'])} & {_f(m['far_sinus'],2)} & {_f(m['far_p90'],2)} \\\\"
             for M, m in abl.items()]
        out.table(f"{name}_buffer", L)
        out.json[name] = {k: v for k, v in o.items()}
        return o

    evaluate_once(test, f"{tag}test")
    if externals:
        for name, spec in externals.items():
            recs, ep_min_len, extended = spec[:3]
            light = spec[3] if len(spec) > 3 else False
            evaluate_once(recs, name, ep_min_len, extended, light)
    out.save(tag)
    return out, models, sel
