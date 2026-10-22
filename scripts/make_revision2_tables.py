"""Macros, tables and figures of the second-revision analyses.

    python scripts/make_revision2_tables.py results_rev2 results_rev manuscript/results
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC2, SRC1, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
M = {}
ALPHA = 0.01
MIN_NOR = {"vitaldb": 100, "vilnius": 1000, "val": 100}   # per-record FAR percentiles: records with enough normal beats


def mac(name, v, nd=1):
    if isinstance(v, str):
        M[name] = v
    elif v is None or not np.isfinite(v):
        M[name] = "--"
    else:
        M[name] = f"{v:.{nd}f}"


def f(v, nd=1):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def tokname(s):
    """Macro-safe token from a method key."""
    out, up = [], True
    for ch in s:
        if ch.isalpha():
            out.append(ch.upper() if up else ch); up = False
        elif ch.isdigit():
            out.append(["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"][int(ch)]); up = True
        else:
            up = True
    return "".join(out)


# ------------------------------------------------------------------------------- load
meta = json.load(open(os.path.join(SRC2, "meta.json")))
VDB = json.load(open(os.path.join(SRC2, "vitaldb_rows.json")))
VIL = {}
fn = os.path.join(SRC2, "vilnius_rows.jsonl")
if os.path.exists(fn):
    for ln in open(fn):
        try:
            d = json.loads(ln); VIL[d["id"]] = d
        except Exception:  # noqa: BLE001
            pass
# complement (run_revision2b.py): extra methods on the same records
fb = os.path.join(SRC2, "vitaldb_rows_b.json")
if os.path.exists(fb):
    for rid, d in json.load(open(fb)).items():
        if rid in VDB:
            VDB[rid]["rows"].update(d["rows"])
fb = os.path.join(SRC2, "vilnius_rows_b.jsonl")
HAVE_B = os.path.exists(fb)
if HAVE_B:
    for ln in open(fb):
        try:
            d = json.loads(ln)
        except Exception:  # noqa: BLE001
            continue
        if d["id"] in VIL:
            VIL[d["id"]]["rows"].update(d["rows"])
VAL = json.load(open(os.path.join(SRC2, "vitaldb_val_aci.json"))) if os.path.exists(os.path.join(SRC2, "vitaldb_val_aci.json")) else {}
DATA = {"vitaldb": VDB, "vilnius": VIL, "val": VAL}
mac("rwVilDaysDone", str(len(VIL)))


def patient_of(rid):
    return rid.split("_d")[0] if rid.startswith("vil") else rid


def summ(ds, m, min_nor=None):
    recs = DATA[ds]
    min_nor = MIN_NOR[ds] if min_nor is None else min_nor
    rows = [(rid, d["rows"][m], d["desc"]) for rid, d in recs.items() if m in d["rows"] and "tp" in d["rows"][m]]
    if not rows:
        return None
    S = lambda k: sum(x[1].get(k, 0) for x in rows)  # noqa: E731
    tp, nt, fp, nn = S("tp"), S("n_tgt"), S("fp"), S("n_nor")
    hours = sum(x[2]["hours"] for x in rows)
    far_rec = np.array([100 * x[1]["fp"] / x[1]["n_nor"] for x in rows if x[1]["n_nor"] >= min_nor])
    far_all = np.array([100 * x[1]["fp"] / x[1]["n_nor"] for x in rows if x[1]["n_nor"] > 0])
    pp = {}
    for rid, r, _ in rows:
        a = pp.setdefault(patient_of(rid), [0, 0]); a[0] += r["fp"]; a[1] += r["n_nor"]
    far_pat = np.array([100 * a / b for a, b in pp.values() if b > 0])
    tasks = sum(x[1].get("tasks", x[1].get("tasks_all", 0)) for x in rows)
    out = {"se": 100 * tp / nt if nt else np.nan, "far": 100 * fp / nn if nn else np.nan,
           "ppv": 100 * tp / (tp + fp) if tp + fp else np.nan,
           "p90": np.percentile(far_rec, 90) if len(far_rec) else np.nan,
           "p95": np.percentile(far_rec, 95) if len(far_rec) else np.nan,
           "max": far_rec.max() if len(far_rec) else np.nan,
           "max_all": far_all.max() if len(far_all) else np.nan,
           "above2": 100 * np.mean(far_rec > 2.0) if len(far_rec) else np.nan,
           "above5": 100 * np.mean(far_rec > 5.0) if len(far_rec) else np.nan,
           "n_rec_ranked": len(far_rec), "n_rec": len(rows),
           "pat_max": far_pat.max() if len(far_pat) else np.nan,
           "pat_above2": 100 * np.mean(far_pat > 2.0) if len(far_pat) else np.nan,
           "nf_h": S("nf") / hours if hours else np.nan, "nt_h": S("nt") / hours if hours else np.nan,
           "tasks_h": tasks / hours if hours else np.nan, "tasks_day": 24 * tasks / hours if hours else np.nan,
           "fp_irr": 100 * S("fp_irr") / fp if fp else np.nan, "fp_near": 100 * S("fp_near") / fp if fp else np.nan,
           "miss_real": 100 * S("miss_ab") / S("rev_ab") if S("rev_ab") else np.nan,
           "fpconf_real": 100 * S("fpconf_nor") / S("rev_nor") if S("rev_nor") else np.nan,
           "startup": S("startup_reviews") / len(rows), "hours": hours, "n_tgt": nt, "n_nor": nn}
    return out


def multi(ds, base, seeds=(0, 1, 2)):
    xs = [summ(ds, f"{base}_s{s}") for s in seeds]
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    out = {}
    for k in xs[0]:
        v = np.array([x[k] for x in xs], float)
        out[k] = float(np.nanmean(v)); out[k + "_lo"] = float(np.nanmin(v)); out[k + "_hi"] = float(np.nanmax(v))
    out["nseeds"] = len(xs)
    return out


def get(ds, key):
    if key.endswith("*"):
        return multi(ds, key[:-1])
    return summ(ds, key)


def cell(s, k, nd=1, rng=False):
    if s is None:
        return "--"
    v = f(s[k], nd)
    if rng and k + "_lo" in s and s.get("nseeds", 1) > 1:
        return f"{v} ({f(s[k + '_lo'], nd)}--{f(s[k + '_hi'], nd)})"
    return v


# ------------------------------------------------------------------------------- reference reproduction
for ds, P in [("vitaldb", "rwV"), ("vilnius", "rwL")]:
    s = summ(ds, "ps_a0.01_M300")
    if s is None:
        continue
    for k in ("se", "far", "ppv", "p90", "p95", "max", "max_all", "above2", "above5", "pat_max", "pat_above2",
              "nf_h", "nt_h", "tasks_h", "tasks_day", "fp_irr", "fp_near", "startup"):
        mac(f"{P}Sel{tokname(k)}", s[k], 2 if k in ("far", "p90", "p95", "max", "pat_max") else 1)
    mac(f"{P}SelNRanked", str(s["n_rec_ranked"])); mac(f"{P}SelNRec", str(s["n_rec"]))
    mac(f"{P}Hours", s["hours"], 0)
    a = summ(ds, "aci_a0.01_g0.005")
    if a is not None:
        for k in ("se", "far", "p90", "p95", "max", "max_all", "above2", "above5", "pat_max", "pat_above2", "nf_h", "tasks_h", "fp_irr"):
            mac(f"{P}Aci{tokname(k)}", a[k], 2 if k in ("far", "p90", "p95", "max", "pat_max") else 1)

# ------------------------------------------------------------------------------- day descriptors (Vilnius)
if VIL:
    desc = {rid: d["desc"] for rid, d in VIL.items()}
    sel = {rid: d["rows"]["ps_a0.01_M300"] for rid, d in VIL.items()}
    far = {rid: 100 * r["fp"] / r["n_nor"] for rid, r in sel.items() if r["n_nor"] > 0}
    hi = [rid for rid, v in far.items() if v > 2]
    mac("rwLDaysAboveTwo", str(len(hi))); mac("rwLDaysRanked", str(len(far)))
    mac("rwLDaysAboveTwoSmall", str(sum(1 for rid in hi if sel[rid]["n_nor"] < 5000)))
    worst = max(far, key=far.get)
    mac("rwLWorstId", worst.replace("_", "\\_")); mac("rwLWorstFar", far[worst], 1)
    mac("rwLWorstFp", str(sel[worst]["fp"])); mac("rwLWorstNnor", str(sel[worst]["n_nor"]))
    mac("rwLWorstBurden", 100 * desc[worst]["burden"], 1)
    big = {rid: v for rid, v in far.items() if sel[rid]["n_nor"] >= 1000}
    mac("rwLMaxBig", max(big.values()), 2)
    fp_tot = sum(r["fp"] for r in sel.values())
    mac("rwLFpShareHi", 100 * sum(sel[rid]["fp"] for rid in hi) / fp_tot, 1)
    mac("rwLMedBurdenHi", 100 * np.median([desc[rid]["burden"] for rid in hi]), 1)
    mac("rwLMedBurdenLo", 100 * np.median([desc[rid]["burden"] for rid in far if rid not in hi]), 1)
    # irregular non-AF beats: base rate among non-AF beats
    nn = sum(d["desc"]["n_nor"] for d in VIL.values())
    irr = sum(d["desc"]["irr_frac_nor"] * d["desc"]["n_nor"] for d in VIL.values() if np.isfinite(d["desc"]["irr_frac_nor"]))
    near = sum(d["desc"]["near_frac_nor"] * d["desc"]["n_nor"] for d in VIL.values() if np.isfinite(d["desc"]["near_frac_nor"]))
    mac("rwLIrrBase", 100 * irr / nn, 1); mac("rwLNearBase", 100 * near / nn, 1)
    # correlation of the day FAR with descriptors (Spearman)
    from scipy.stats import spearmanr
    ids = list(big)
    y = np.array([big[i] for i in ids])
    for k, tok in [("burden", "Burden"), ("ppg_cov", "Ppg"), ("ecg_bad", "Ecg"), ("rr_cv", "Rrcv"),
                   ("irr_frac_nor", "Irr"), ("score_drift", "Drift"), ("hr", "Hr")]:
        x = np.array([desc[i].get(k, np.nan) for i in ids], float)
        ok = np.isfinite(x)
        rho = spearmanr(x[ok], y[ok]).correlation if ok.sum() > 5 else np.nan
        mac(f"rwLRho{tok}", rho, 2)

# ------------------------------------------------------------------------------- alpha sweep table
ALPHA_ROWS = [(0.001, 3000), (0.0025, 1200), (0.005, 300), (0.005, 600), (0.01, 300), (0.02, 300)]
lines = []
for a, Mb in ALPHA_ROWS:
    key = f"ps_a{a}_M{Mb}"
    v = summ("vitaldb", key); l_ = summ("vilnius", key)
    pg = summ("vilnius", f"pop_early_a{a}")
    ac = summ("vilnius", f"aci_a{a}_g0.005") if a in meta["aci_alphas"] else None
    lab = f"{100 * a:g} & {Mb}"
    lines.append(f"{lab} & {cell(v, 'se')} & {cell(v, 'far', 2)} & {cell(l_, 'se')} & {cell(l_, 'far', 2)} & "
                 f"{cell(l_, 'above2')} & {cell(l_, 'nf_h')} & {cell(l_, 'tasks_h')} & "
                 f"{cell(ac, 'se')} & {cell(ac, 'far', 2)} & {cell(pg, 'se')} & {cell(pg, 'far', 2)} \\\\")
    tk = {0.001: "PtOne", 0.0025: "PtTwoFive", 0.005: "PtFive", 0.01: "One", 0.02: "Two"}[a] + \
        {300: "Mthree", 600: "Msix", 1200: "Mtwelve", 3000: "Mthirty"}[Mb]
    for ds, P in [("vitaldb", "rwV"), ("vilnius", "rwL")]:
        s = summ(ds, key)
        if s:
            for k in ("se", "far", "nf_h", "tasks_h", "above2", "nt_h"):
                mac(f"{P}Alpha{tk}{tokname(k)}", s[k], 2 if k == "far" else 1)
open(os.path.join(OUT, "tab_rev2_alpha.tex"), "w").write("\n".join(lines) + "\n")
# impossible levels with M = 300
mac("rwImpossibleAlpha", "0.33")

# ------------------------------------------------------------------------------- frontier (ACI vs ours vs population)
front = {"ps": [], "aci": {}, "pop": []}
for a, Mb in ALPHA_ROWS:
    if (a, Mb) == (0.005, 300):
        continue
    s = summ("vilnius", f"ps_a{a}_M{Mb}")
    if s:
        front["ps"].append((a, s))
for g in sorted(set(meta["aci_gammas"]) | ({0.0005, 0.002} if HAVE_B else set())):
    for a in meta["aci_alphas"]:
        s = summ("vilnius", f"aci_a{a}_g{g}")
        if s:
            front["aci"].setdefault(g, []).append((a, s))
for a in meta["alphas"]:
    s = summ("vilnius", f"pop_early_a{a}")
    if s:
        front["pop"].append((a, s))
lines = []
for a, s in front["ps"]:
    lines.append(f"Per-patient conformal & {100 * a:g} & -- & {cell(s, 'se')} & {cell(s, 'far', 2)} & {cell(s, 'p95', 2)} & "
                 f"{cell(s, 'above2')} & {cell(s, 'pat_max', 2)} & {cell(s, 'nf_h')} & {cell(s, 'tasks_h')} \\\\")
lines.append("\\midrule")
for g, lst in front["aci"].items():
    for a, s in lst:
        lines.append(f"ACI & {100 * a:g} & {g:g} & {cell(s, 'se')} & {cell(s, 'far', 2)} & {cell(s, 'p95', 2)} & "
                     f"{cell(s, 'above2')} & {cell(s, 'pat_max', 2)} & {cell(s, 'nf_h')} & {cell(s, 'tasks_h')} \\\\")
lines.append("\\midrule")
for a, s in front["pop"]:
    lines.append(f"Population Joint-RF, global thr. & {100 * a:g} & -- & {cell(s, 'se')} & {cell(s, 'far', 2)} & {cell(s, 'p95', 2)} & "
                 f"{cell(s, 'above2')} & {cell(s, 'pat_max', 2)} & {cell(s, 'nf_h')} & -- \\\\")
open(os.path.join(OUT, "tab_rev2_frontier.tex"), "w").write("\n".join(lines) + "\n")
# matched comparison: most sensitive ACI setting whose budget exceedance is at most that of ours at 1 %
ours = summ("vilnius", "ps_a0.01_M300")
if ours:
    cand = [(g, a, s) for g, lst in front["aci"].items() for a, s in lst if s["above2"] <= ours["above2"] + 1e-9]
    mac("rwLAciMatchN", str(len(cand)))
    if cand:
        g, a, s = max(cand, key=lambda x: x[2]["se"])
        mac("rwLAciMatchSe", s["se"]); mac("rwLAciMatchFar", s["far"], 2); mac("rwLAciMatchAlpha", 100 * a, 2)
        mac("rwLAciMatchGamma", f"{g:g}"); mac("rwLAciMatchAbove", s["above2"])
    cand = [(g, a, s) for g, lst in front["aci"].items() for a, s in lst if s["far"] <= ours["far"] + 1e-9]
    if cand:
        g, a, s = max(cand, key=lambda x: x[2]["se"])
        mac("rwLAciFarMatchSe", s["se"]); mac("rwLAciFarMatchFar", s["far"], 2); mac("rwLAciFarMatchAlpha", 100 * a, 2)
        mac("rwLAciFarMatchGamma", f"{g:g}"); mac("rwLAciFarMatchAbove", s["above2"]); mac("rwLAciFarMatchPNF", s["p95"], 2)
    # ours at 2 %: what sensitivity does a larger alpha buy?
    for a, s in front["ps"]:
        if a == 0.02:
            mac("rwLPsTwoSe", s["se"]); mac("rwLPsTwoFar", s["far"], 2); mac("rwLPsTwoAbove", s["above2"])

fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
for ax, xk, xl in [(axs[0], "above2", "Days with FAR above 2% (%)"), (axs[1], "nf_h", "False notifications per hour")]:
    xs = [s[xk] for a, s in front["ps"]]; ys = [s["se"] for a, s in front["ps"]]
    ax.plot(xs, ys, "-o", color=BLUE, ms=3.5, lw=1.2, label="Per-patient conformal ($\\alpha$ grid)")
    for a, s in front["ps"]:
        ax.annotate(f"{100 * a:g}%", (s[xk], s["se"]), textcoords="offset points", xytext=(5, -3), fontsize=6, color=BLUE)
    marks = {0.0005: "o", 0.001: "s", 0.002: "P", 0.005: "^", 0.02: "D"}
    for g, lst in front["aci"].items():
        ax.plot([s[xk] for a, s in lst], [s["se"] for a, s in lst], "-" + marks.get(g, "o"), color=ORANGE, ms=3.5,
                lw=0.9, alpha=0.6 + 0.2 * (g == 0.005), label=f"ACI, $\\gamma={g:g}$")
    ax.plot([s[xk] for a, s in front["pop"]], [s["se"] for a, s in front["pop"]], "-v", color=AQUA, ms=3.5, lw=1.0,
            label="Population Joint-RF, global thr.")
    ax.set_xlabel(xl); ax.set_ylabel("Sensitivity, AF beats (%)")
    if xk == "nf_h":
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 40]); ax.set_xticklabels(["2", "5", "10", "20", "40"])
        ax.minorticks_off()
    ax.set_ylim(0, 100)
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, fontsize=6.5, frameon=False, loc="lower center", ncol=4)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(os.path.join(OUT, "fig_rev2_frontier.pdf")); plt.close(fig)

# ------------------------------------------------------------------------------- reviewer table
REV = [("Exact reviewer (reference)", "rev_oracle", "aci_a0.01_g0.005"),
       ("5\\% + 5\\% errors, independent per beat", "rev_err5_beat*", "aci_err5_beat*"),
       ("5\\% + 5\\% errors, one decision per review task", "rev_err5_task*", "aci_err5_task*"),
       ("10\\% + 10\\% errors, one decision per review task", "rev_err10_task*", None),
       ("5\\% + 5\\% errors, concentrated on borderline alarms", "rev_err5_score*", None),
       ("Delay, exponential, mean 1 min", "rev_expdelay60", None),
       ("Delay, exponential, mean 5 min", "rev_expdelay300*", "aci_expdelay300*"),
       ("Delay 5 min, beats awaiting review admitted provisionally", "rev_delay300_prov", None),
       ("Queue checked every 15 min", "rev_batch15", "aci_batch15"),
       ("Queue checked every 15 min, provisional admission", "rev_batch15_prov", None),
       ("Half of the alarms reviewed, the others kept out", "rev_avail50_withhold", None),
       ("Strip adjudication (exact reviewer)", "rev_epiexcl_oracle", None),
       ("Strip adjudication, 5\\% + 5\\% task errors", "rev_epiexcl_err5_task*", None)]
lines = []
for lab, k, ka in REV:
    v, l_ = get("vitaldb", k), get("vilnius", k)
    lines.append(f"{lab} & {cell(v, 'se', 1, True)} & {cell(v, 'far', 2)} & {cell(l_, 'se', 1, True)} & {cell(l_, 'far', 2)} & "
                 f"{cell(l_, 'above2')} & {cell(l_, 'tasks_h')} & {cell(l_, 'miss_real')} & {cell(l_, 'fpconf_real')} \\\\")
    tk = tokname(k.replace("*", "").replace("rev_", ""))
    for ds, P, s_ in [("vitaldb", "rwV", v), ("vilnius", "rwL", l_)]:
        if s_:
            for kk in ("se", "far", "above2", "tasks_h", "tasks_day", "miss_real", "fpconf_real", "p95"):
                mac(f"{P}Rev{tk}{tokname(kk)}", s_[kk], 2 if kk in ("far", "p95") else 1)
            if "se_lo" in s_:
                mac(f"{P}Rev{tk}SeLo", s_["se_lo"]); mac(f"{P}Rev{tk}SeHi", s_["se_hi"])
    a = get("vilnius", ka) if ka else None
    if a:
        for kk in ("se", "far", "above2"):
            mac(f"rwLAciRev{tk}{tokname(kk)}", a[kk], 2 if kk == "far" else 1)
open(os.path.join(OUT, "tab_rev2_reviewer.tex"), "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------------------------------- ACI under the same reviewers
CMP = [("Exact reviewer", "rev_oracle", "aci_a0.01_g0.005", "aci_a0.0025_g0.001"),
       ("5\\% + 5\\% errors per beat", "rev_err5_beat*", "aci_err5_beat*", "aci_a0.0025_g0.001_err5_beat*"),
       ("5\\% + 5\\% errors per task", "rev_err5_task*", "aci_err5_task*", "aci_a0.0025_g0.001_err5_task*"),
       ("10\\% + 10\\% errors per task", "rev_err10_task*", None, "aci_a0.0025_g0.001_err10_task*"),
       ("10\\% false confirmations per task", "rev_fp10_task*", None, "aci_a0.0025_g0.001_fp10_task*"),
       ("10\\% missed confirmations per task", "rev_fn10_task*", None, "aci_a0.0025_g0.001_fn10_task*"),
       ("Delay, exponential, mean 1 min", "rev_expdelay60", None, "aci_a0.0025_g0.001_expdelay60"),
       ("Delay, exponential, mean 5 min", "rev_expdelay300*", "aci_expdelay300*", "aci_a0.0025_g0.001_expdelay300*"),
       ("Queue checked every 15 min", "rev_batch15", "aci_batch15", "aci_a0.0025_g0.001_batch15")]
lines = []
for lab, kp, ka1, ka2 in CMP:
    cells = []
    for k in (kp, ka1, ka2):
        s_ = get("vilnius", k) if k else None
        cells += [cell(s_, "se"), cell(s_, "far", 2), cell(s_, "above2")]
        v_ = get("vitaldb", k) if k else None
        if s_ is not None:
            tk = tokname(k.replace("*", ""))
            mac(f"rwLCmp{tk}Se", s_["se"]); mac(f"rwLCmp{tk}Far", s_["far"], 2); mac(f"rwLCmp{tk}Above", s_["above2"])
        if v_ is not None:
            tk = tokname(k.replace("*", ""))
            mac(f"rwVCmp{tk}Se", v_["se"]); mac(f"rwVCmp{tk}Far", v_["far"], 2); mac(f"rwVCmp{tk}Above", v_["above2"])
    lines.append(lab + " & " + " & ".join(cells) + " \\\\")
open(os.path.join(OUT, "tab_rev2_acicmp.tex"), "w").write("\n".join(lines) + "\n")

# ACI setting selected on VALIDATION patients with the rule used for our configuration
if VAL:
    grid = sorted({k for d in VAL.values() for k in d["rows"] if k.startswith("aci_a")})
    res = {k: summ("val", k) for k in grid}
    ok = [k for k in grid if res[k]["far"] <= 1.5]
    if ok:
        best = max(res[k]["se"] for k in ok)
        near = [k for k in ok if res[k]["se"] >= best - 1.0]
        ksel = min(near, key=lambda k: (res[k]["tasks_h"], -res[k]["se"]))
        a_ = float(ksel.split("_a")[1].split("_g")[0]); g_ = float(ksel.split("_g")[1])
        mac("rwAciValAlpha", 100 * a_, 2); mac("rwAciValGamma", f"{g_:g}")
        mac("rwAciValSe", res[ksel]["se"]); mac("rwAciValFar", res[ksel]["far"], 2)
        mac("rwAciValN", str(len(grid)))
        for ds, P in [("vitaldb", "rwV"), ("vilnius", "rwL")]:
            s_ = summ(ds, ksel)
            if s_:
                for kk in ("se", "far", "above2", "p95", "pat_max", "nf_h", "tasks_h"):
                    mac(f"{P}AciVal{tokname(kk)}", s_[kk], 2 if kk in ("far", "p95", "pat_max") else 1)
    # best budget-respecting setting on validation (FAR of every patient <= 2 % ... reported for information)
    mac("rwAciGridN", str(len(grid)))

# ------------------------------------------------------------------------------- decomposition
lines = []
for ds, P in [("vitaldb", "rwV"), ("vilnius", "rwL")]:
    recs = DATA[ds]
    if not recs:
        continue
    for lab, k in [("ECG-RF, global threshold", "pop_ecg_a0.01"),
                   ("Per-patient, oracle-clean buffer", "oracle_clean"),
                   ("Per-patient, adjudicated (selected)", "ps_a0.01_M300")]:
        s = summ(ds, k)
        # burden strata (VitalDB: ectopic burden of the record; Vilnius: AF burden of the day)
        def strat(lo, hi):
            rr = [d["rows"][k] for d in recs.values() if lo <= d["desc"]["burden"] < hi]
            nt = sum(x["n_tgt"] for x in rr)
            return 100 * sum(x["tp"] for x in rr) / nt if nt else np.nan
        cut = 0.10
        se_lo, se_hi = strat(0, cut), strat(cut, 1.01)
        name = "VitalDB" if ds == "vitaldb" else "Vilnius"
        lines.append(f"{name} & {lab} & {cell(s, 'se')} & {f(se_lo)} & {f(se_hi)} & {cell(s, 'far', 2)} \\\\")
        tk = tokname(k)
        mac(f"{P}Dec{tk}Se", s["se"] if s else np.nan); mac(f"{P}Dec{tk}Far", s["far"] if s else np.nan, 2)
        mac(f"{P}Dec{tk}SeLow", se_lo); mac(f"{P}Dec{tk}SeHigh", se_hi)
        if s:
            DEC = globals().setdefault("DEC", {}); DEC[(ds, k)] = s["se"]
    if ds == "vitaldb":
        lines.append("\\midrule")
    nt = [d["rows"]["ps_a0.01_M300"]["n_tgt"] for d in recs.values()]
    hi = [d["rows"]["ps_a0.01_M300"]["n_tgt"] for d in recs.values() if d["desc"]["burden"] >= 0.10]
    mac(f"{P}HighBurdenShare", 100 * sum(hi) / max(1, sum(nt)), 0)
    mac(f"{P}HighBurdenN", str(sum(1 for d in recs.values() if d["desc"]["burden"] >= 0.10)))
open(os.path.join(OUT, "tab_rev2_decomp.tex"), "w").write("\n".join(lines) + "\n")

if "DEC" in globals() and ("vitaldb", "pop_ecg_a0.01") in DEC:
    mac("rwVDecLoss", DEC[("vitaldb", "pop_ecg_a0.01")] - DEC[("vitaldb", "ps_a0.01_M300")], 0)
    mac("rwVDecRecovered", DEC[("vitaldb", "oracle_clean")] - DEC[("vitaldb", "ps_a0.01_M300")], 0)

# ------------------------------------------------------------------------------- splits: burden strata (first revision)
seeds = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(SRC1, "seed_*.json")))]
gap_rows = []
for d in seeds:
    v = d["vitaldb"]
    sel, pop = v["rows"]["selected"], v["rows"]["pop_default_early"]
    nt = np.array([x["n_tgt"] for x in sel]); nn = np.array([x["n_nor"] for x in sel])
    hb = nt / np.maximum(nt + nn, 1) >= 0.10
    se = lambda rows, m: 100 * sum(x["tp"] for x, h in zip(rows, m) if h) / max(1, nt[m].sum())  # noqa: E731
    gap_rows.append({"seed": d["seed"], "cal": d["selected"][0], "share": 100 * nt[hb].sum() / nt.sum(),
                     "sel_hi": se(sel, hb), "sel_lo": se(sel, ~hb), "pop_hi": se(pop, hb), "pop_lo": se(pop, ~hb),
                     "sel": v["summary"]["selected"]["se"], "pop": v["summary"]["pop_default_early"]["se"]})
G = {k: np.array([g[k] for g in gap_rows], float) for k in ("share", "sel_hi", "sel_lo", "pop_hi", "pop_lo", "sel", "pop")}
for k, tok in [("share", "Share"), ("sel_hi", "SelHi"), ("sel_lo", "SelLo"), ("pop_hi", "PopHi"), ("pop_lo", "PopLo")]:
    mac(f"rwSplit{tok}Med", float(np.median(G[k]))); mac(f"rwSplit{tok}Min", float(np.min(G[k]))); mac(f"rwSplit{tok}Max", float(np.max(G[k])))
prim = next(g for g in gap_rows if g["seed"] == 42)
for k, tok in [("share", "Share"), ("sel_hi", "SelHi"), ("sel_lo", "SelLo"), ("pop_hi", "PopHi"), ("pop_lo", "PopLo")]:
    mac(f"rwPrim{tok}", prim[k])
adj10 = [g["sel"] for g in gap_rows if g["cal"] == "adj10"]; adj5 = [g["sel"] for g in gap_rows if g["cal"] == "adj5"]
mac("rwSplitAdjTenN", str(len(adj10))); mac("rwSplitAdjFiveN", str(len(adj5)))
mac("rwSplitAdjTenSeMed", float(np.median(adj10)) if adj10 else np.nan); mac("rwSplitAdjFiveSeMed", float(np.median(adj5)) if adj5 else np.nan)
rank = sorted(g["sel"] for g in gap_rows).index(prim["sel"]) + 1
mac("rwPrimRank", str(rank)); mac("rwNSplits", str(len(gap_rows)))
gap = G["pop"] - G["sel"]
from scipy.stats import spearmanr  # noqa: E402
mac("rwSplitGapRhoShare", spearmanr(gap, G["share"]).correlation, 2)

# notifications versus beat-level false alarms (first revision, Vilnius, selected configuration)
prim_d = next(d for d in seeds if d["seed"] == 42)
vv = prim_d["vilnius"]
fpv = sum(x["fp"] for x in vv["rows"]["selected"]); hv = sum(e["hours"] for e in vv["extras"])
nfv = sum(e["notif_selected"][1] for e in vv["extras"]); nnv = sum(x["n_nor"] for x in vv["rows"]["selected"])
mac("rwLNorPerHour", nnv / hv, 0); mac("rwLFpPerHour", fpv / hv, 1); mac("rwLFpPerNotif", fpv / nfv, 1)

# ------------------------------------------------------------------------------- unified appendix tables
app_fn = os.path.join(SRC2, "appendix.json")
FUS = [("ecg", "ECG-RF"), ("ppg", "PPG-RF"), ("early", "Joint-RF (early)"), ("late", "Late (min tail)"),
       ("gated", "SQI-gated late"), ("pbonf", "Bonferroni, $2\\min p$"), ("pmean", "Twice the mean, $2\\bar p$")]
CAL = [("labelfree", "Label-free, no screen"), ("screen6", "Label-free, screen $\\tau=6$"),
       ("adj0", "Adjudicated, start-up 0\\%"), ("adj5", "Adjudicated, start-up 5\\%"), ("adj10", "Adjudicated, start-up 10\\%")]
if os.path.exists(app_fn):
    APP = json.load(open(app_fn))

    def pooled(part, key):
        rows = [v[key] for v in APP[part].values() if key in v]
        S = lambda k: sum(r.get(k, 0) for r in rows)  # noqa: E731
        au = [r["auroc"] for r in rows if "auroc" in r]
        h = S("hours")
        return {"auroc": np.mean(au) if au else np.nan,
                "se": 100 * S("tp") / S("n_tgt") if S("n_tgt") else np.nan,
                "se_v": 100 * S("tp_v") / S("n_v") if S("n_v") else np.nan,
                "se_s": 100 * S("tp_s") / S("n_s") if S("n_s") else np.nan,
                "se_poor": 100 * S("tp_poor") / S("n_poor") if S("n_poor") else np.nan,
                "ppv": 100 * S("tp") / (S("tp") + S("fp")) if S("tp") + S("fp") else np.nan,
                "far": 100 * S("fp") / S("n_nor") if S("n_nor") else np.nan,
                "rev_h": S("reviews_grouped") / h if h and any("reviews_grouped" in r for r in rows) else np.nan}
    sel_key = "adj5|ecg"
    lines = []
    for ci, (c, cl) in enumerate(CAL):
        if ci:
            lines.append("\\midrule")
        for fu, fl in FUS:
            s = pooled("val", f"{c}|{fu}")
            mark = "\\checkmark" if f"{c}|{fu}" == sel_key else ""
            lines.append(f"{cl} & {fl} & {f(s['auroc'], 3)} & {f(s['se'])} & {f(s['far'], 2)} & {f(s['rev_h'], 0)} & {mark} \\\\")
    open(os.path.join(OUT, "tab_vdb_dev.tex"), "w").write("\n".join(lines) + "\n")
    lines = []
    for fu, fl in FUS:
        s = pooled("test", f"adj5|{fu}")
        nm = fl + (" (selected)" if fu == "ecg" else "")
        lines.append(f"{nm} & {f(s['auroc'], 3)} & {f(s['se'])} & {f(s['se_v'])} & {f(s['se_s'])} & {f(s['ppv'])} & "
                     f"{f(s['far'], 2)} & {f(s['se_poor'])} & {f(s['rev_h'], 0)} \\\\")
    lines.append("\\midrule")
    for key, nm in [("labelfree|ecg", "Label-free, no screen"), ("screen6|ecg", "Label-free, screen $\\tau=6$"),
                    ("adj0|ecg", "Adjudicated, start-up 0\\%"), ("adj10|ecg", "Adjudicated, start-up 10\\%"),
                    ("static", "Selected, frozen calibration"), ("global_ecg", "ECG-RF, global threshold"),
                    ("default_ecg", "ECG-RF, $P>0.5$"), ("global_early", "Joint-RF, global threshold"),
                    ("default_early", "Joint-RF, $P>0.5$")]:
        s = pooled("test", key)
        au = "--" if ("|" in key or key == "static") else f(s["auroc"], 3)
        rv = f(s["rev_h"], 0) if "|" in key else "--"
        lines.append(f"{nm} & {au} & {f(s['se'])} & {f(s['se_v'])} & {f(s['se_s'])} & {f(s['ppv'])} & "
                     f"{f(s['far'], 2)} & {f(s['se_poor'])} & {rv} \\\\")
    open(os.path.join(OUT, "tab_vdbtest_beat.tex"), "w").write("\n".join(lines) + "\n")
    s = pooled("test", sel_key)
    mac("rwAppSelPpv", s["ppv"]); mac("rwAppSelSe", s["se"]); mac("rwAppSelFar", s["far"], 2)
    for key, tok in [("global_ecg", "GlobEcg"), ("default_ecg", "DefEcg"), ("global_early", "GlobEarly"), ("default_early", "DefEarly")]:
        s = pooled("test", key); mac(f"rwApp{tok}Ppv", s["ppv"]); mac(f"rwApp{tok}Se", s["se"]); mac(f"rwApp{tok}Far", s["far"], 2)
    s = pooled("test", "adj5|ecg"); mac("rwAppSelAuroc", s["auroc"], 3)
    au = [v["adj5|ecg"]["auroc"] for v in APP["test"].values() if "auroc" in v["adj5|ecg"]]
    mac("rwAppSelAurocMed", float(np.median(au)), 3); mac("rwAppSelAurocN", str(len(au)))

# ------------------------------------------------------------------------------- causal check
cfn = os.path.join(SRC2, "causal.json")
if os.path.exists(cfn):
    C = {k: v for k, v in json.load(open(cfn)).items() if "error" not in v}
    mac("rwCausalN", str(len(C)))
    if C:
        mac("rwCausalRepro", float(np.median([v["repro_maxabs"] for v in C.values()])), 3)
        mac("rwCausalReproMax", float(np.max([v["repro_maxabs"] for v in C.values()])), 3)
        for tag in ["ref", "full", "causal", "bounded1", "bounded2", "bounded3"]:
            rows = [v[tag] for v in C.values() if tag in v]
            if not rows:
                continue
            tp = sum(r["tp"] for r in rows); nt = sum(r["n_tgt"] for r in rows)
            fp = sum(r["fp"] for r in rows); nn = sum(r["n_nor"] for r in rows)
            tk = tokname(tag)
            mac(f"rwCausal{tk}Se", 100 * tp / nt if nt else np.nan); mac(f"rwCausal{tk}Far", 100 * fp / nn if nn else np.nan, 2)
            if tag not in ("ref", "full"):
                mac(f"rwCausal{tk}Agree", 100 * np.mean([v[f"agree_{tag}"] for v in C.values()]), 2)
                mac(f"rwCausal{tk}AgreeFull", 100 * np.mean([v[f"agree_full_{tag}"] for v in C.values()]), 2)
                mac(f"rwCausal{tk}Rel", 100 * np.median([v[f"{tag}_rel"][0] for v in C.values()]), 1)
            if tag.startswith("bounded"):
                mac(f"rwCausal{tk}Lat", float(np.median([v[f"lat_{tag}"][0] for v in C.values()])), 2)
                mac(f"rwCausal{tk}LatP", float(np.median([v[f"lat_{tag}"][1] for v in C.values()])), 2)

# ------------------------------------------------------------------------------- episode alarms under a budget (Vilnius)
efn = os.path.join(SRC2, "vilnius_episodes.jsonl")
if os.path.exists(efn):
    import math as _m
    em = json.load(open(os.path.join(SRC2, "episodes_meta.json")))
    ER = [json.loads(ln) for ln in open(efn)]
    EH = sum(r["hours"] for r in ER)

    def eagg(key, j):
        fa = sum(r[key][j]["false"] for r in ER)
        out = {"fday": 24 * fa / EH}
        for ml in (30, 120):
            ne = sum(r[key][j][f"ep{ml}"][0] for r in ER); nd = sum(r[key][j][f"ep{ml}"][1] for r in ER)
            dl = [d for r in ER for d in r[key][j][f"ep{ml}"][2]]
            out[f"se{ml}"] = 100 * nd / ne if ne else np.nan
            out[f"delay{ml}"] = float(np.median(dl)) if dl else np.nan
            out[f"n{ml}"] = ne
        return out
    METH = [("ps_cusum", "Per-patient conformal, CUSUM", [f"$h=\\ln{_m.exp(h):.0e}$".replace("e+0", "e").replace("e+", "e") for h in em["h_grid"]]),
            ("ps_run", "Per-patient conformal, $k$ of 60", [f"$k={k}$" for k in em["k_grid"]]),
            ("aci_prim_run", "ACI (1\\%, $\\gamma=0.005$), $k$ of 60", [f"$k={k}$" for k in em["k_grid"]]),
            ("aci_cons_run", "ACI (0.25\\%, $\\gamma=0.001$), $k$ of 60", [f"$k={k}$" for k in em["k_grid"]])]
    BUD = [(24.0, "1 per hour", "Hour"), (4.0, "1 per 6 h", "Six"), (1.0, "1 per day", "Day")]
    lines = []
    for bi, (b, blab, btok) in enumerate(BUD):
        if bi:
            lines.append("\\midrule")
        for key, lab, glabs in METH:
            pts = [(j, eagg(key, j)) for j in range(len(glabs))]
            ok = [(j, p) for j, p in pts if p["fday"] <= b]
            mtok = {"ps_cusum": "PsCusum", "ps_run": "PsRun", "aci_prim_run": "AciPrimRun", "aci_cons_run": "AciConsRun"}[key]
            if ok:
                j, p = max(ok, key=lambda x: x[1]["se120"])
                lines.append(f"{blab} & {lab} & {glabs[j]} & {f(p['fday'], 2)} & {f(p['se120'])} & {f(p['delay120'], 0)} & {f(p['se30'])} \\\\")
                mac(f"rwEp{btok}{mtok}Se", p["se120"]); mac(f"rwEp{btok}{mtok}SeThirty", p["se30"])
                mac(f"rwEp{btok}{mtok}Fday", p["fday"], 2); mac(f"rwEp{btok}{mtok}Delay", p["delay120"], 0)
            else:
                mn = min(p["fday"] for _, p in pts)
                lines.append(f"{blab} & {lab} & not reached & ({f(mn, 1)}) & -- & -- & -- \\\\")
                mac(f"rwEp{btok}{mtok}Se", "--")
    open(os.path.join(OUT, "tab_rev2_epbudget.tex"), "w").write("\n".join(lines) + "\n")
    p0 = eagg("ps_cusum", 0)
    mac("rwEpNOneTwenty", str(p0["n120"])); mac("rwEpNThirty", str(p0["n30"])); mac("rwEpHours", EH, 0)
    # minimum false-alarm rate reachable by the CUSUM on the grid
    mac("rwEpCusumMinFday", min(eagg("ps_cusum", j)["fday"] for j in range(len(em["h_grid"]))), 1)

# ------------------------------------------------------------------------------- short aliases
for pre_long, pre_short in [("CmpAciAZeroZeroZeroTwoFiveGZeroZeroZeroOne", "Acic"), ("CmpAciAZeroZeroOneGZeroZeroZeroFive", "Acip"),
                            ("CmpRev", "Ps"), ("CmpAci", "AcipRev")]:
    for k in list(M):
        for P in ("rwL", "rwV"):
            if k.startswith(P + pre_long):
                M[P + pre_short + k[len(P + pre_long):]] = M[k]
for g, a, tok in [(0.001, 0.0025, "AcicGrid"), (0.001, 0.01, "AciOneSlow"), (0.005, 0.005, "AciHalf")]:
    for ds, P in [("vitaldb", "rwV"), ("vilnius", "rwL")]:
        s_ = summ(ds, f"aci_a{a}_g{g}")
        if s_:
            for kk in ("se", "far", "above2", "p95", "pat_max", "pat_above2", "nf_h"):
                mac(f"{P}{tok}{tokname(kk)}", s_[kk], 2 if kk in ("far", "p95", "pat_max") else 1)

# ------------------------------------------------------------------------------- write
with open(os.path.join(OUT, "macros_rev2.tex"), "w") as fh:
    for k, v in sorted(M.items()):
        fh.write(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}\n")
print(len(M), "macros")
