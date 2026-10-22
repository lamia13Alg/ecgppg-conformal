"""Robust mapping of VitalDB Arrhythmia Database annotation strings to beat classes and rhythm flags.

Beat classes: 0 normal, 1 supraventricular, 2 ventricular, 3 conducted during AF/flutter,
-1 unclassifiable/unknown.
Accepts full words ("Normal", "Supraventricular", "Ventricular", "Unclassifiable"), common
abbreviations (N, S, A, APC, PAC, SVPB, V, PVC, VPC, U, Q) and their lower-case variants.
Rhythm labels may be given only at segment starts: they are forward-filled within a record.
"""
import numpy as np
import pandas as pd


def beat_code(s):
    s = str(s).strip().lower()
    if not s or s in ("nan", "none"):
        return -1
    if s.startswith("supra") or s in ("s", "a", "apc", "pac", "sveb", "svpb", "j", "e"):
        return 1
    if s.startswith("ventric") or s in ("v", "pvc", "vpc", "veb", "f"):
        return 2
    if s.startswith("normal") or s in ("n", "l", "r", "nor"):
        return 0
    if "supraventricular" in s or "atrial premature" in s:
        return 1
    if "ventricular" in s:
        return 2
    return -1


def is_sinus(r):
    r = str(r).strip().lower()
    if not r or r in ("nan", "none"):
        return False
    if "dysfunction" in r or "node" in r:
        return False
    return ("sinus" in r) or r in ("nsr", "n", "(n", "sr")


def is_af(r):
    r = str(r).strip().lower()
    return ("afib" in r) or ("afl" in r) or ("atrial fib" in r) or ("atrial flutter" in r)


def labels_from_annotations(ann):
    """ann: DataFrame already filtered to finite time_second and sorted.
    Returns (label, sinus, bad) arrays aligned with ann rows."""
    label = ann["beat_type"].map(beat_code).to_numpy(int)
    rh = ann["rhythm_label"].replace({"": np.nan}).ffill()
    sinus = rh.map(is_sinus).to_numpy(bool)
    # Beats conducted during atrial fibrillation/flutter are labelled S in VitalDB. They are not
    # ectopic beats: code them 3 (non-normal for the reviewer, excluded from beat-level metrics).
    af = rh.map(is_af).to_numpy(bool)
    label = np.where(af & (label == 1), 3, label)
    bad = ann.get("bad_signal_quality", pd.Series(False, index=ann.index))
    bad = bad.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"]).to_numpy()
    return label, sinus, bad


def read_annotations(path):
    ann = pd.read_csv(path)
    ann = ann[np.isfinite(pd.to_numeric(ann["time_second"], errors="coerce"))]
    ann = ann.assign(time_second=pd.to_numeric(ann["time_second"])).sort_values("time_second")
    return ann.reset_index(drop=True)
