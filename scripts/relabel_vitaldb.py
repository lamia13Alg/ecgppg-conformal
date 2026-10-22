"""Recompute beat labels, sinus-rhythm flags and ECG-quality flags of cached VitalDB records from
the annotation CSV files (no waveform download), and print an inventory of the annotation strings.

    python scripts/relabel_vitaldb.py --ann physionet/Annotation_Files --cache cache
"""
import argparse
import collections
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ecgppg.vitaldb_labels import read_annotations, labels_from_annotations  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--cache", default="cache")
    args = ap.parse_args()
    beats = collections.Counter()
    rhythms = collections.Counter()
    quality = collections.Counter()
    tot = collections.Counter()
    fixed = mismatched = 0
    for f in sorted(glob.glob(os.path.join(args.cache, "vitaldb", "vdb*.npz"))):
        cid = os.path.basename(f)[3:-4]
        csv = os.path.join(args.ann, f"Annotation_file_{cid}.csv")
        if not os.path.exists(csv):
            mismatched += 1
            continue
        ann = read_annotations(csv)
        beats.update(ann["beat_type"].astype(str))
        rhythms.update(ann["rhythm_label"].astype(str))
        if "bad_signal_quality" in ann:
            quality.update(ann["bad_signal_quality"].astype(str))
        z = dict(np.load(f, allow_pickle=True))
        t = z["t"]
        ta = ann["time_second"].to_numpy(float)
        if len(ta) != len(t) or np.max(np.abs(ta - t)) > 1e-6:
            # align by nearest time if the order or length differs
            idx = np.clip(np.searchsorted(ta, t), 1, len(ta) - 1)
            idx = np.where(np.abs(ta[idx - 1] - t) < np.abs(ta[idx] - t), idx - 1, idx)
            ann = ann.iloc[idx].reset_index(drop=True)
            mismatched += 1
        label, sinus, bad = labels_from_annotations(ann)
        z["label"], z["sinus"], z["ecg_bad"] = label, sinus, bad
        np.savez_compressed(f, **z)
        fixed += 1
        ev = z["eval_mask"]
        tot["beats"] += len(t)
        tot["normal"] += int((label == 0).sum())
        tot["supra"] += int((label == 1).sum())
        tot["ventr"] += int((label == 2).sum())
        tot["unclass"] += int((label < 0).sum())
        tot["af_conducted"] += int((label == 3).sum())
        tot["sinus"] += int(sinus.sum())
        tot["normal_sinus_eval"] += int((ev & (label == 0) & sinus).sum())
        tot["bad_ecg"] += int(bad.sum())
    print(f"records relabelled: {fixed} (re-aligned or missing CSV: {mismatched})")
    print("beat_type values      :", beats.most_common(12))
    print("rhythm_label values   :", rhythms.most_common(15))
    print("bad_signal_quality    :", quality.most_common(5))
    print("totals                :", dict(tot))
    if tot["normal_sinus_eval"] == 0:
        print("WARNING: no normal beat in sinus rhythm found -- send this output to update the mapping")


if __name__ == "__main__":
    main()
