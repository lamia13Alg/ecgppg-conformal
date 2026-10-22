"""LaTeX macros, tables and figures of the revision analyses (results_rev/*.json).

    python scripts/make_revision_tables.py results_rev manuscript/results
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6})

M = {}


def mac(name, v, nd=1):
    if isinstance(v, str):
        M[name] = v
    elif v is None or not np.isfinite(v):
        M[name] = "--"
    else:
        M[name] = f"{v:.{nd}f}"


def f(v, nd=1):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


main = json.load(open(os.path.join(SRC, "seed_42.json")))
flow = json.load(open(os.path.join(SRC, "flow.json")))
seeds = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(SRC, "seed_*.json")))]
seeds = sorted(seeds, key=lambda d: (d["seed"] != 42, d["seed"]))

# ------------------------------------------------------------------------------- flow
mac("rvFlowAnnot", str(flow.get("vitaldb_annotated", 482)))
mac("rvFlowNoPPG", str(flow.get("vitaldb_annotated", 482) - flow["vitaldb_cached"]))
mac("rvFlowCached", str(flow["vitaldb_cached"]))
mac("rvFlowNoBaseline", str(flow["vitaldb_cached"] - flow["vitaldb_included"]))
mac("rvFlowIncluded", str(flow["vitaldb_included"]))
mac("rvVilDays", str(flow["vilnius_days"]))
mac("rvVilPatients", str(flow["vilnius_patients"]))
mac("rvVilBeatsM", flow["vilnius_beats"] / 1e6, 1)
mac("rvVilAFBeatsM", flow["vilnius_af_beats"] / 1e6, 2)
mac("rvVilAFpct", 100 * flow["vilnius_af_beats"] / flow["vilnius_beats"], 1)
mac("rvVilPPGcov", flow["vilnius_ppg_coverage_pct"], 1)

# ------------------------------------------------------------------------------- main split
TOK = {"selected": "Sel", "pop_default_early": "PopDef", "pop_global_early": "PopGlob",
       "pop_default_ecg": "PopDefEcg", "pop_global_ecg": "PopGlobEcg", "aci_early": "Aci",
       "cnn_default": "CnnDef", "cnn_global": "CnnGlob", "cnn_conformal": "CnnConf",
       "af_pop_default": "AfDef", "af_pop_global": "AfGlob", "af_conformal": "AfConf",
       "pernorm_ecg": "Norm", "pernorm_early": "Norm", "static": "Static", "labelfree": "Lf",
       "nolook_conformal": "Nolook", "ecg_samecal": "EcgSame"}
NAMES = [("selected", "Per-patient conformal (selected)"), ("static", "\\quad same, frozen calibration"),
         ("labelfree", "\\quad same, label-free buffer"), ("nolook_conformal", "\\quad same, no look-ahead"),
         ("pernorm_ecg", "Per-patient normalisation (no conformal)"),
         ("pernorm_early", "Per-patient normalisation (no conformal)"),
         ("aci_early", "ACI on population calibration set"),
         ("pop_default_early", "Population Joint-RF, $P>0.5$"), ("pop_global_early", "Population Joint-RF, global thr."),
         ("pop_default_ecg", "Population ECG-RF, $P>0.5$"), ("pop_global_ecg", "Population ECG-RF, global thr."),
         ("cnn_default", "Population 1-D CNN, $P>0.5$"), ("cnn_global", "Population 1-D CNN, global thr."),
         ("cnn_conformal", "1-D CNN + per-patient conformal"),
         ("af_pop_default", "AF-trained RF, $P>0.5$"), ("af_pop_global", "AF-trained RF, global thr."),
         ("af_conformal", "AF-trained RF + per-patient conformal")]
for ds, P in [("vitaldb", "rvV"), ("vilnius", "rvL")]:
    S, B = main[ds]["summary"], main[ds]["bootstrap"]
    for m, tok in TOK.items():
        if m not in S:
            continue
        s, b = S[m], B.get(m, {})
        mac(f"{P}{tok}Se", s["se"]); mac(f"{P}{tok}Far", s["far"], 2); mac(f"{P}{tok}Ppv", s["ppv"])
        mac(f"{P}{tok}Pninety", s["far_p90"], 2); mac(f"{P}{tok}Max", s["far_max"], 1)
        mac(f"{P}{tok}Above", s["pct_above_2a"], 0)
        if "se" in b:
            mac(f"{P}{tok}SeLo", b["se"][1]); mac(f"{P}{tok}SeHi", b["se"][2])
            mac(f"{P}{tok}FarLo", b["far"][1], 2); mac(f"{P}{tok}FarHi", b["far"][2], 2)
        if "d_se" in b:
            mac(f"{P}{tok}DSe", b["d_se"][0]); mac(f"{P}{tok}DSeLo", b["d_se"][1]); mac(f"{P}{tok}DSeHi", b["d_se"][2])
            mac(f"{P}{tok}DFar", b["d_far"][0], 2); mac(f"{P}{tok}DFarLo", b["d_far"][1], 2); mac(f"{P}{tok}DFarHi", b["d_far"][2], 2)
    mac(f"{P}NTgt", f"{S['selected']['n_tgt']:,}".replace(",", "\\,"))
    mac(f"{P}NNor", f"{S['selected']['n_nor']:,}".replace(",", "\\,"))
    L = []
    for m, nm in NAMES:
        if m not in S or (ds == "vitaldb" and m.startswith("af_")):
            continue
        s, b = S[m], B.get(m, {})
        ci = lambda k, nd: f"[{b[k][1]:.{nd}f}, {b[k][2]:.{nd}f}]" if k in b else ""  # noqa: E731
        L.append(f"{nm} & {f(s['se'])} {ci('se', 1)} & {f(s['far'], 2)} {ci('far', 2)} & {f(s['ppv'])} & "
                 f"{f(s['far_p90'], 2)} & {f(s['far_max'], 1)} & {f(s['pct_above_2a'], 0)} \\\\")
        if m in ("nolook_conformal", "pernorm_ecg", "pernorm_early", "aci_early", "pop_global_ecg",
                 "cnn_conformal"):
            L.append("\\midrule")
    if L and L[-1] == "\\midrule":
        L.pop()
    open(os.path.join(OUT, f"tab_rev_comp_{ds}.tex"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------------------- reviewer scenarios
SC = [("oracle", "Exact reviewer (reference)", "Oracle"), ("err5", "5\\% false + 5\\% missed confirmations", "ErrFive"),
      ("err10", "10\\% + 10\\%", "ErrTen"), ("err20", "20\\% + 20\\%", "ErrTwenty"),
      ("fp10", "10\\% false confirmations only", "FpTen"), ("fn10", "10\\% missed confirmations only", "FnTen"),
      ("delay60", "Confirmation delayed by 1 min", "DelaySixty"), ("delay300", "Confirmation delayed by 5 min", "DelayFive"),
      ("avail50", "Half of the alarms reviewed", "AvailFifty"), ("auto", "No reviewer: alarms auto-excluded", "Auto")]
L = []
for key, nm, tok in SC:
    row = [nm]
    for ds, P in [("vitaldb", "rvV"), ("vilnius", "rvL")]:
        s = main[ds]["summary"]["rev_" + key]
        row += [f(s["se"]), f(s["far"], 2), f(s["far_max"], 1)]
        mac(f"{P}Rev{tok}Se", s["se"]); mac(f"{P}Rev{tok}Far", s["far"], 2); mac(f"{P}Rev{tok}Max", s["far_max"], 1)
    L.append(" & ".join(row) + " \\\\")
open(os.path.join(OUT, "tab_rev_reviewer.tex"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------------------- burden and episodes
for ds, P in [("vitaldb", "rvV"), ("vilnius", "rvL")]:
    E = main[ds]["extras"]
    H = sum(e["hours"] for e in E)
    mac(f"{P}Hours", H, 1)
    for m, tok in [("selected", "Sel"), ("pop_default_early", "PopDef"), ("pop_global_early", "PopGlob")]:
        nt = sum(e[f"notif_{m}"][0] for e in E); nf = sum(e[f"notif_{m}"][1] for e in E)
        mac(f"{P}Notif{tok}True", nt / H, 1); mac(f"{P}Notif{tok}False", nf / H, 1)
        mac(f"{P}Notif{tok}FalseDay", 24 * nf / H, 0)
    for k in sorted(E[0]):
        if not k.startswith("ep_"):
            continue
        _, tag, ml = k.split("_")
        ne = sum(e[k][0] for e in E); nd = sum(e[k][1] for e in E)
        dl = [x for e in E for x in e[k][2]]; fa = sum(e[k][3] for e in E)
        tok = f"{P}Ep{tag.capitalize()}{ {'3': 'Three', '30': 'Thirty', '120': 'OneTwenty', '300': 'ThreeHundred'}[ml]}"
        mac(tok + "N", str(ne)); mac(tok + "Se", 100 * nd / max(ne, 1)); mac(tok + "Delay", float(np.median(dl)) if dl else np.nan)
        mac(tok + "False", fa / H, 2)
    lat = np.array([e["lat_ecg"] for e in E if "lat_ecg" in e])
    mac(f"{P}LatMed", float(np.median(lat[:, 0])), 2); mac(f"{P}LatNinetyFive", float(np.median(lat[:, 1])), 2)
    bur = 100 * np.array([e["burden"] for e in E])
    mac(f"{P}BurdenMed", float(np.median(bur)), 1); mac(f"{P}BurdenQThree", float(np.percentile(bur, 75)), 1)
    for kind, n in [("time", None), ("sqi", 4)]:
        for m, tok in [("selected", "Sel"), ("pop_default_early", "PopDef")]:
            arr = np.array([e[f"cond_{kind}_{m}"] for e in E]).sum(0)
            rates = [100 * a / b for a, b in arr if b > 1000]
            mac(f"{P}Cond{kind.capitalize()}{tok}Min", min(rates), 2); mac(f"{P}Cond{kind.capitalize()}{tok}Max", max(rates), 2)
mac("rvCusumH", main["cusum_h"], 1)
mac("rvCostRf", main["cost_us_per_beat"]["rf_scoring"], 0)
mac("rvCostConf", main["cost_us_per_beat"]["conformal_update"], 0)
sh = main["shift"]
mac("rvShiftEcg", sh["domain_auroc_ecg"], 3); mac("rvShiftPpg", sh["domain_auroc_ppg"], 3)
mac("rvShiftFracV", 100 * sh["frac_normal_above0_vitaldb"], 1); mac("rvShiftFracL", 100 * sh["frac_normal_above0_vilnius"], 1)
mac("rvAfVdbN", str(main["vitaldb"]["summary"]["af_conformal"]["n_tgt"]))

L = []
for ds, P, eps in [("vitaldb", "rvV", ["3"]), ("vilnius", "rvL", ["30", "120", "300"])]:
    E = main[ds]["extras"]; H = sum(e["hours"] for e in E)
    for tag in ("default", "tuned"):
        for ml in eps:
            k = f"ep_{tag}_{ml}"
            ne = sum(e[k][0] for e in E); nd = sum(e[k][1] for e in E)
            dl = [x for e in E for x in e[k][2]]; fa = sum(e[k][3] for e in E)
            L.append(f"{'VitalDB' if ds == 'vitaldb' else 'Vilnius'} & $\\geq${ml} beats & "
                     f"{'$h=\\ln 4000$' if tag == 'default' else '$h$ tuned'} & {ne} & {f(100 * nd / max(ne, 1))} & "
                     f"{f(float(np.median(dl)) if dl else np.nan)} & {f(fa / H, 2)} \\\\")
open(os.path.join(OUT, "tab_rev_episodes.tex"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------------------- stability across splits
CAL = {"adj0": "adj. 0\\%", "adj5": "adj. 5\\%", "adj10": "adj. 10\\%", "labelfree": "label-free", "screen6": "screen"}
FU = {"ecg": "ECG", "early": "Joint", "ppg": "PPG", "late": "late", "gated": "gated", "pbonf": "p-Bonf", "pmean": "p-mean"}
L, sv, pv, lv, lpv, fus = [], [], [], [], [], []
for d in seeds:
    v = d["vitaldb"]["summary"]; vi = d.get("vilnius", {}).get("summary")
    fus.append(d["selected"][1])
    sv.append([v["selected"]["se"], v["selected"]["far"], v["selected"]["far_max"]])
    pv.append([v["pop_default_early"]["se"], v["pop_default_early"]["far"], v["pop_default_early"]["far_max"]])
    row = [str(d["seed"]) + ("$^\\ast$" if d["seed"] == 42 else ""), f"{CAL[d['selected'][0]]}, {FU[d['selected'][1]]}",
           f(v["selected"]["se"]), f(v["selected"]["far"], 2), f(v["selected"]["far_max"], 1),
           f(v["pop_default_early"]["se"]), f(v["pop_default_early"]["far"], 2), f(v["pop_default_early"]["far_max"], 1)]
    if vi:
        lv.append([vi["selected"]["se"], vi["selected"]["far"], vi["selected"]["far_max"]])
        lpv.append([vi["pop_default_early"]["se"], vi["pop_default_early"]["far"], vi["pop_default_early"]["far_max"]])
        row += [f(vi["selected"]["se"]), f(vi["selected"]["far"], 2), f(vi["pop_default_early"]["se"]), f(vi["pop_default_early"]["far"], 2)]
    else:
        row += ["--", "--", "--", "--"]
    L.append(" & ".join(row) + " \\\\")
open(os.path.join(OUT, "tab_rev_seeds.tex"), "w").write("\n".join(L) + "\n")
sv, pv, lv, lpv = map(np.array, (sv, pv, lv, lpv))
mac("rvNSplits", str(len(seeds))); mac("rvNSplitsVil", str(len(lv)))
mac("rvNSplitsJoint", str(sum(x == "early" for x in fus)))
for arr, P in [(sv, "rvSplitSel"), (pv, "rvSplitPop"), (lv, "rvSplitVilSel"), (lpv, "rvSplitVilPop")]:
    for j, k in enumerate(["Se", "Far", "Max"]):
        nd = 2 if k == "Far" else 1
        mac(f"{P}{k}Med", float(np.median(arr[:, j])), nd); mac(f"{P}{k}Min", float(arr[:, j].min()), nd)
        mac(f"{P}{k}MaxV", float(arr[:, j].max()), nd)
mac("rvPopSeBetterN", str(int((pv[:, 0] > sv[:, 0]).sum())))
mac("rvPopFarBetterN", str(int((pv[:, 1] < sv[:, 1]).sum())))
mac("rvPopMaxWorseN", str(int((pv[:, 2] > sv[:, 2]).sum())))
dev_gain = [d["dev_early_samecal"]["se"] - d["dev_ecg_samecal"]["se"] for d in seeds]
mac("rvDevJointGainMed", float(np.median(dev_gain)), 1)
mac("rvDevJointBetter", str(sum(g > 0 for g in dev_gain)))

# ------------------------------------------------------------------------------- figures
# (1) operating points with 95% CIs
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
pts = [("selected", "Per-patient conformal", BLUE, "o"), ("aci_early", "ACI", AQUA, "D"),
       ("pop_default_early", "Population, $P>0.5$", ORANGE, "s"), ("pop_global_early", "Population, global thr.", ORANGE, "^")]
for a, (ds, title) in zip(ax, [("vitaldb", "VitalDB test (in distribution)"), ("vilnius", "Vilnius (distribution shift)")]):
    S, B = main[ds]["summary"], main[ds]["bootstrap"]
    for m, lab, col, mk in pts:
        s, b = S[m], B[m]
        a.errorbar(s["far"], s["se"], xerr=[[s["far"] - b["far"][1]], [b["far"][2] - s["far"]]],
                   yerr=[[s["se"] - b["se"][1]], [b["se"][2] - s["se"]]], fmt=mk, color=col, ms=6,
                   mec="white", mew=1.0, elinewidth=1.2, capsize=0, label=lab, zorder=3)
    a.axvline(1.0, ls="--", color=MUTED, lw=0.8)
    a.set_xscale("log"); a.set_xlabel("False-alarm rate (%, log scale)"); a.set_title(title, fontsize=8, color=INK)
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    tk = [0.3, 0.5, 1, 2, 5, 10, 20, 50]
    lo_, hi_ = a.get_xlim(); tk = [x for x in tk if lo_ <= x <= hi_]
    a.xaxis.set_major_locator(FixedLocator(tk)); a.xaxis.set_major_formatter(FixedFormatter([f"{x:g}" for x in tk]))
    a.xaxis.set_minor_locator(NullLocator())
    a.set_ylim(0, 100)
ax[0].set_ylabel("Sensitivity (%)")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=7)
fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(os.path.join(OUT, "fig_rev_operating.pdf")); plt.close(fig)

# (2) stability across splits: worst-patient FAR and sensitivity, selected vs population
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
rng = np.random.default_rng(0)
for a, j, yl in [(ax[0], 0, "Sensitivity (%)"), (ax[1], 2, "Worst-patient FAR (%)")]:
    for x, arr, col, lab in [(0, sv, BLUE, "Per-patient\nconformal"), (1, pv, ORANGE, "Population\n$P>0.5$")]:
        a.scatter(x + rng.uniform(-.12, .12, len(arr)), arr[:, j], s=22, color=col, edgecolor="white", lw=0.8, zorder=3)
        a.plot([x - .22, x + .22], [np.median(arr[:, j])] * 2, color=INK, lw=1.6)
    a.set_xticks([0, 1]); a.set_xticklabels(["Per-patient\nconformal", "Population\n$P>0.5$"]); a.set_ylabel(yl)
    a.set_xlim(-.5, 1.5)
ax[1].set_yscale("log")
fig.suptitle(f"VitalDB, {len(sv)} patient splits (one dot per split; bar: median)", fontsize=8, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(os.path.join(OUT, "fig_rev_splits.pdf")); plt.close(fig)

# (3) reviewer scenarios
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
labels = ["exact", "5%+5%", "10%+10%", "20%+20%", "10% false", "10% missed", "delay 1 min", "delay 5 min", "50% reviewed", "automatic"]
y = np.arange(len(SC))[::-1]
for a, k, xl in [(ax[0], "se", "Sensitivity (%)"), (ax[1], "far", "False-alarm rate (%, log scale)")]:
    for off, ds, col, lab in [(0.18, "vitaldb", BLUE, "VitalDB"), (-0.18, "vilnius", ORANGE, "Vilnius")]:
        vals = [main[ds]["summary"]["rev_" + key][k] for key, _, _ in SC]
        a.barh(y + off, vals, height=0.34, color=col, label=lab, edgecolor="white", linewidth=1.0)
    if k == "far":
        a.set_xscale("log"); a.axvline(1.0, ls="--", color=MUTED, lw=0.8)
    a.set_xlabel(xl); a.grid(axis="y", visible=False)
ax[0].set_yticks(y); ax[0].set_yticklabels(labels)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=7)
fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(os.path.join(OUT, "fig_rev_reviewer.pdf")); plt.close(fig)

# (4) conditional FAR on Vilnius
E = main["vilnius"]["extras"]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
for a, kind, xt, xl in [(ax[0], "time", ["0-4", "4-8", "8-12", "12-16", "16-20", "20-24"], "Hours since start of the monitoring day"),
                        (ax[1], "sqi", ["absent/\n<0.01", "<0.5", "0.5-0.8", ">0.8"], "PPG quality index of the beat")]:
    for m, col, mk, lab in [("selected", BLUE, "o", "Per-patient conformal"), ("pop_default_early", ORANGE, "s", "Population, $P>0.5$")]:
        arr = np.array([e[f"cond_{kind}_{m}"] for e in E]).sum(0)
        r = [100 * a_ / b if b else np.nan for a_, b in arr]
        a.plot(range(len(r)), r, marker=mk, color=col, lw=2, ms=6, mec="white", label=lab)
    a.axhline(1.0, ls="--", color=MUTED, lw=0.8)
    a.set_xticks(range(len(xt))); a.set_xticklabels(xt); a.set_xlabel(xl)
ax[0].set_ylabel("False-alarm rate (%)"); ax[0].set_ylim(0, None)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=7)
fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(os.path.join(OUT, "fig_rev_conditional.pdf")); plt.close(fig)

# (5) representative case
c = main.get("case", {})
if c.get("t"):
    t = (np.array(c["t"]) - c["onset"]) / 60.0
    lab = np.array(c["label"]); rr = np.array(c["rr"], float); p = np.array(c["p"], float)
    al = np.array(c["alarm"], bool); pop = np.array(c["pop"], float); bound = np.array(c["bound"], float)
    fig, ax = plt.subplots(4, 1, figsize=(7.0, 5.0), sharex=True)
    for a in ax:
        a.fill_between(t, 0, 1, where=lab == 1, color="#f2d7cc", transform=a.get_xaxis_transform(), lw=0, zorder=0)
    ax[0].plot(t, rr, ".", ms=1.2, color=INK); ax[0].set_ylabel("RR (s)"); ax[0].set_ylim(0.2, 2.0)
    ax[1].plot(t, -np.log10(np.clip(p, 1e-4, 1)), ".", ms=1.2, color=BLUE)
    ax[1].axhline(2, ls="--", color=MUTED, lw=0.8); ax[1].set_ylabel("$-\\log_{10} p$")
    ax[2].plot(t, pop, ".", ms=1.2, color=ORANGE); ax[2].axhline(0, ls="--", color=MUTED, lw=0.8)
    ax[2].set_ylabel("Population\nlogit")
    w = 60  # rolling alarm rate over 60 beats
    ra = np.convolve(al.astype(float), np.ones(w) / w, mode="same") * 100
    rp = np.convolve(((pop > 0) & np.isfinite(pop)).astype(float), np.ones(w) / w, mode="same") * 100
    ax[3].plot(t, ra, color=BLUE, lw=1.5, label="Per-patient conformal")
    ax[3].plot(t, rp, color=ORANGE, lw=1.5, label="Population, $P>0.5$")
    ax[3].set_ylabel("Alarms (%,\n60-beat window)"); ax[3].set_xlabel("Minutes from AF onset (shaded: annotated AF)")
    ax[3].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_rev_case.pdf")); plt.close(fig)
    mac("rvCaseId", c["id"].replace("_", "\\_"))

with open(os.path.join(OUT, "macros_rev.tex"), "w") as fh:
    for k, v in sorted(M.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(M), "macros")
