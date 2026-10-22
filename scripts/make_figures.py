"""Figures for the manuscript, read from results/results_vdb.json.

    python scripts/make_figures.py --res results/results_vdb.json --out results
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = {"sel": "#2a78d6", "lf": "#eb6834", "sc": "#1baf7a", "ecg": "#52514e", "glob": "#0b0b0b"}
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False})


def alpha_sweep(res, name, out):
    sel = res["selected"]
    sw = res[name]["alpha_sweep"]
    al = sorted(float(a) for a in sw)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for cal, lab, col, mk in [(sel[0], "Selected (adjudicated)", C["sel"], "*"),
                              ("labelfree", "Label-free, no screen", C["lf"], "o"),
                              ("screen6", "Label-free, screen", C["sc"], "s")]:
        k = f"{cal}|{sel[1]}"
        far = [sw[str(a)][k]["far_sinus"] for a in al]
        se = [sw[str(a)][k]["se"] for a in al]
        ax[0].plot(np.array(al) * 100, far, marker=mk, color=col, label=lab)
        ax[1].plot(np.array(al) * 100, se, marker=mk, color=col)
        if cal == sel[0]:
            bd = [sw[str(a)][k]["bound"] for a in al]
            ax[0].plot(np.array(al) * 100, bd, ":", color="k", lw=1, label="Bound (Prop. 1)")
    ax[0].plot(np.array(al) * 100, np.array(al) * 100, "--", color="grey", lw=1, label=r"Nominal $\alpha$")
    ax[0].set_xlabel(r"Nominal level $\alpha$ (%)"); ax[0].set_ylabel("False-alarm rate (%)")
    ax[1].set_xlabel(r"Nominal level $\alpha$ (%)"); ax[1].set_ylabel("Sensitivity (%)")
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"fig_{name}_alpha.pdf"))
    plt.close(fig)


def perrecord(res, name, out, alpha):
    pr = res[name]["perrecord_far"]
    keys = [("selected", "Selected\n(ECG+PPG conformal)", C["sel"]),
            ("ecg_only", "ECG only\n(same calibration)", C["ecg"]),
            ("global_early", "Joint-RF\nglobal threshold", C["glob"]),
            ("default_early", "Joint-RF\n$P>0.5$", C["lf"])]
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    rng = np.random.default_rng(0)
    for j, (k, lab, col) in enumerate(keys):
        v = np.array([x for x in pr[k] if x is not None and np.isfinite(x)]) * 100
        ax.scatter(j + rng.uniform(-0.18, 0.18, len(v)), v, s=8, color=col, alpha=0.7)
        if len(v):
            q1, md, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([j - 0.25, j + 0.25], [md, md], color="k", lw=2)
            ax.plot([j, j], [q1, q3], color="k", lw=1)
    ax.axhline(alpha * 100, ls="--", color="grey", lw=1)
    ax.axhline(2 * alpha * 100, ls=":", color="grey", lw=1)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([k[1] for k in keys])
    ax.set_ylabel("Per-record false-alarm rate (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"fig_{name}_perrecord.pdf"))
    plt.close(fig)


def quality(res, name, out):
    sel = res["selected"]
    m_sel = res[name][f"{sel[0]}|{sel[1]}"]
    m_ecg = res[name][f"{sel[0]}|ecg"]
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.5))
    x = np.arange(2)
    for j, (m, lab, col) in enumerate([(m_ecg, "ECG only", C["ecg"]), (m_sel, "ECG+PPG (selected)", C["sel"])]):
        ax[0].bar(x + (j - 0.5) * 0.36, [m["se_goodecg"], m["se_badecg"]], 0.36, color=col, label=lab)
        ax[1].bar(x + (j - 0.5) * 0.36, [m["far_goodecg"], m["far_badecg"]], 0.36, color=col)
    for a, yl in zip(ax, ["Sensitivity (%)", "False-alarm rate (%)"]):
        a.set_xticks(x); a.set_xticklabels(["Good ECG", "Poor ECG"]); a.set_ylabel(yl)
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"fig_{name}_quality.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/results_vdb.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--name", default="vdbtest")
    args = ap.parse_args()
    res = json.load(open(args.res))
    name = args.name
    alpha_sweep(res, name, args.out)
    perrecord(res, name, args.out, args.alpha)
    quality(res, name, args.out)
    print("figures written to", args.out)


if __name__ == "__main__":
    main()
