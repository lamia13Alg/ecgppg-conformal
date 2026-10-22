"""Figure and macros of the simulation study (results/simulation.json)."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

res = json.load(open(sys.argv[1]))
out = sys.argv[2]
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False})
deps = [("iid", "Independent"), ("ar5", "AR(1), $\\rho=0.5$"), ("ar9", "AR(1), $\\rho=0.9$"), ("drift", "Linear drift")]
modes = [("labelfree", "Label-free", "#2e86ab", "o"), ("adj0", "Adjudicated, $q=0$", "#edae49", "s"),
         ("adj10", "Adjudicated, $q=10\\%$", "#d1495b", "*")]
fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.6), sharex=True)
for j, (dep, title) in enumerate(deps):
    for mode, lab, col, mk in modes:
        xs = sorted({x["eps"] for x in res if x["dep"] == dep})
        far = [next(x["far"] for x in res if x["dep"] == dep and x["eps"] == e and x["mode"] == mode and x["M"] == 300 and not x["cluster"]) for e in xs]
        se = [next(x["se"] for x in res if x["dep"] == dep and x["eps"] == e and x["mode"] == mode and x["M"] == 300 and not x["cluster"]) for e in xs]
        ax[0, j].plot(np.array(xs) * 100, far, marker=mk, color=col, label=lab, ms=4)
        ax[1, j].plot(np.array(xs[1:]) * 100, se[1:], marker=mk, color=col, ms=4)
    bd = [np.mean([x["bound"] for x in res if x["dep"] == dep and x["eps"] == e and x["M"] == 300 and not x["cluster"] and x["mode"] == "adj10"]) for e in xs]
    ax[0, j].plot(np.array(xs) * 100, bd, ":", color="k", lw=1, label="Bound (Prop. 1)")
    ax[0, j].axhline(1.0, ls="--", color="grey", lw=0.8)
    ax[0, j].set_title(title, fontsize=8)
    ax[0, j].set_ylim(0, 2.4)
    ax[1, j].set_ylim(0, 100)
    ax[1, j].set_xlabel("Contamination $\\varepsilon$ (%)")
ax[0, 0].set_ylabel("False-alarm rate (%)")
ax[1, 0].set_ylabel("Sensitivity (%)")
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=7)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(out, "fig_simulation.pdf"))


def get(dep, eps, mode, M=300, cl=False, k="far"):
    return next(x[k] for x in res if x["dep"] == dep and x["eps"] == eps and x["mode"] == mode and x["M"] == M and x["cluster"] == cl)


mac = {
    "resSimIidFarMax": max(x["far"] for x in res if x["dep"] == "iid" and x["mode"] != "labelfree"),
    "resSimIidBoundMax": max(x["bound"] for x in res if x["dep"] == "iid" and x["mode"] != "labelfree"),
    "resSimArFiveFar": get("ar5", 0.0, "adj10"),
    "resSimArNineFar": get("ar9", 0.0, "adj10"),
    "resSimDriftFar": get("drift", 0.0, "adj10"),
    "resSimLfSeTen": get("iid", 0.10, "labelfree", k="se"),
    "resSimLfSeOne": get("iid", 0.01, "labelfree", k="se"),
    "resSimAdjTenSeTen": get("iid", 0.10, "adj10", k="se"),
    "resSimAdjZeroSeTen": get("iid", 0.10, "adj0", k="se"),
    "resSimAdjZeroFarTen": get("iid", 0.10, "adj0"),
}
with open(os.path.join(out, "macros_sim.tex"), "w") as fh:
    for k, v in mac.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v:.{1 if 'Se' in k else 2}f}}}\n")
print(mac)
