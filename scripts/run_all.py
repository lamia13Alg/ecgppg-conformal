"""Main protocol: VitalDB development/selection/test, then the Vilnius external test.

    python scripts/run_all.py --cache cache --out results
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ecgppg.pipeline import load_record, LazyRecords  # noqa: E402
from ecgppg.experiments import run_vitaldb_protocol, split_patients  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="results")
    ap.add_argument("--alpha", type=float, default=0.01)
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.cache, "vitaldb", "*.npz")))
    recs = {}
    for p in paths:
        r = load_record(p)
        if r["eval_mask"].sum() > 0 and (r["label"][r["fit_mask"]] == 0).sum() >= 50:
            recs[r["id"]] = r
    tr, va, te = split_patients(list(recs), seed=42)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "split_vitaldb.json"), "w") as fh:
        json.dump({"train": tr, "val": va, "test": te}, fh, indent=1)
    print(f"VitalDB records: {len(recs)} (train {len(tr)}, val {len(va)}, test {len(te)})")
    externals = {}
    vil = sorted(glob.glob(os.path.join(args.cache, "vilnius", "*.npz")))
    if vil:
        # AF episodes: >= 30 consecutive AF beats (gaps <= 10 beats)
        externals["vilnius"] = (LazyRecords(vil), 30, False, True)
    run_vitaldb_protocol([recs[i] for i in tr], [recs[i] for i in va], [recs[i] for i in te],
                         args.out, alpha=args.alpha, tag="vdb", externals=externals)
    print("Results, LaTeX tables and macros written to", args.out)


if __name__ == "__main__":
    main()
