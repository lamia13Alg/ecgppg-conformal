"""Dataset loaders.

VitalDB Arrhythmia Database (development): annotations from PhysioNet
  (https://physionet.org/content/vitaldb-arrhythmia/1.0.0/), waveforms from VitalDB through the
  `vitaldb` package (tracks SNUADC/ECG_II and SNUADC/PLETH, 500 Hz).
Vilnius long-term ECG + wrist PPG (external test 1): MAT files from the open version v3 on Zenodo
  (10.5281/zenodo.23187681, non-commercial licence). Field names must be checked with scripts/inspect_vilnius.py and set in
  VILNIUS_KEYS below.
PhysioNet/CinC Challenge 2015 (external test 2): WFDB records with ECG, PLETH and an alarm
  label in the header comments.
"""
import glob
import os
import numpy as np
import pandas as pd

from .pipeline import build_record

FS_VITALDB = 500.0


from .vitaldb_labels import beat_code as _beat_code, read_annotations, labels_from_annotations  # noqa: E402,F401


def load_vitaldb_case(case_id, ann_dir, window_min=5.0, pad=2.0):
    import vitaldb  # pip install vitaldb
    ann = read_annotations(os.path.join(ann_dir, f"Annotation_file_{case_id}.csv"))
    t = ann["time_second"].to_numpy(float)
    if len(t) < 600:
        return None
    t0, t1 = max(0.0, t[0] - pad), t[-1] + pad
    vals = vitaldb.load_case(int(case_id), ["SNUADC/ECG_II", "SNUADC/PLETH"], 1 / FS_VITALDB)
    if vals is None or len(vals) == 0:
        return None
    i0, i1 = int(t0 * FS_VITALDB), min(len(vals), int(t1 * FS_VITALDB))
    ecg = vals[i0:i1, 0]
    ppg = vals[i0:i1, 1]
    if not np.isfinite(ecg).any() or np.isfinite(ppg).mean() < 0.5:
        return None  # PPG missing: case excluded (reported in the flow diagram)
    label, sinus, bad = labels_from_annotations(ann)
    off = i0 / FS_VITALDB
    return build_record(f"vdb{case_id}", ecg, FS_VITALDB, ppg, FS_VITALDB, t, label,
                        sinus=sinus, ecg_bad=bad, window_min=window_min,
                        offset_e=off, offset_p=off)


# ---------------------------------------------------------------------------------------------
# Vilnius dataset. Set the field names after running scripts/inspect_vilnius.py on one file.
# Open version v3 (Zenodo 10.5281/zenodo.23187681, 39 patients, 261 days, non-commercial licence).
# Files are MATLAB v7.3 (HDF5). Structure checked on patient 001:
#   XX_ECG.mat : ECG (N,1) 500 Hz; QRSindex (B,1) 1-based sample indices; AF_annotation (B-1,1)
#                and rr (B-1,1), one value per RR interval; recording_startday / recording_starttime
#                (MATLAB char arrays, "HH:MM:SS"); Accelerometer_X/Y/Z 25 Hz.
#   XX_PPG.mat : cell arrays (one cell per uninterrupted piece) PPG_GREEN, PPG_AMBIENT,
#                Accelerometer_X/Y/Z, recording_startday, recording_starttime.
FS_VIL_ECG = 500.0
FS_VIL_PPG = 100.0


def _h5_value(f, obj):
    """Return a numpy array, or a str for MATLAB char arrays (stored as uint16)."""
    import h5py
    if isinstance(obj, h5py.Reference):
        obj = f[obj]
    a = obj[()]
    if obj.attrs.get("MATLAB_class", b"") in (b"char", "char") or a.dtype == np.uint16:
        return "".join(chr(int(c)) for c in np.asarray(a).ravel() if int(c) != 0)
    return np.asarray(a)


def _h5_cell(f, name):
    refs = np.asarray(f[name][()]).ravel()
    return [_h5_value(f, f[r]) for r in refs]


def _day_number(v):
    if isinstance(v, str):
        v = v.strip()
        return float(v) if v else 0.0
    v = np.asarray(v).ravel()
    return float(v[0]) if v.size else 0.0


def _tod_seconds(v):
    if isinstance(v, str):
        parts = [float(x) for x in v.strip().replace(".", ":").split(":") if x != ""]
        parts = (parts + [0, 0, 0])[:3]
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    v = np.asarray(v, float).ravel()
    if v.size >= 3:
        return v[0] * 3600 + v[1] * 60 + v[2]
    x = float(v[0]) if v.size else 0.0
    return x * 86400.0 if x < 1.0 else x


def read_vilnius_patient(ecg_path, ppg_path):
    """Return dict with ECG h5 dataset handle (kept open), beat times, AF labels and PPG pieces."""
    import h5py
    fe = h5py.File(ecg_path, "r")
    qrs = np.asarray(fe["QRSindex"][()], float).ravel() - 1.0     # MATLAB 1-based -> 0-based
    afi = np.asarray(fe["AF_annotation"][()], float).ravel()
    # one AF value per RR interval: assign it to the beat that ends the interval
    af = np.zeros(len(qrs), int)
    m = min(len(afi), len(qrs) - 1)
    af[1:m + 1] = (np.nan_to_num(afi[:m]) > 0).astype(int)
    af[0] = af[1] if len(af) > 1 else 0
    ecg_start = (_day_number(_h5_value(fe, fe["recording_startday"])) * 86400.0
                 + _tod_seconds(_h5_value(fe, fe["recording_starttime"])))
    pieces = []
    if ppg_path is not None and os.path.exists(ppg_path):
        with h5py.File(ppg_path, "r") as fp:
            sigs = _h5_cell(fp, "PPG_GREEN")
            days = _h5_cell(fp, "recording_startday")
            tods = _h5_cell(fp, "recording_starttime")
            for x, d, h in zip(sigs, days, tods):
                st = _day_number(d) * 86400.0 + _tod_seconds(h) - ecg_start
                pieces.append((st, np.asarray(x, np.float32).ravel()))
    return {"ecg": fe["ECG"], "file": fe, "t": qrs / FS_VIL_ECG, "af": af, "pieces": pieces}


def load_vilnius_days(ecg_path, ppg_path, window_min=5.0, verbose=True):
    """Yield one record per monitoring day (24 h) of one patient: ECG beats with AF labels and PPG
    pieces placed on the ECG time axis. Each day restarts the patient window. AF beats are coded 1
    (target), others 0."""
    P = read_vilnius_patient(ecg_path, ppg_path)
    ds, t_all, af = P["ecg"], P["t"], P["af"]
    n_ecg = ds.shape[0] if ds.shape[0] > ds.shape[-1] else ds.shape[-1]
    rid = os.path.basename(ecg_path).split("_")[0]
    if verbose:
        cover = sum(len(x) for _, x in P["pieces"]) / FS_VIL_PPG / 3600
        print(f"patient {rid}: {len(t_all)} beats, {n_ecg / FS_VIL_ECG / 3600:.1f} h ECG, "
              f"{len(P['pieces'])} PPG pieces ({cover:.1f} h), AF beats {int(af.sum())}, "
              f"first PPG offset {P['pieces'][0][0] if P['pieces'] else float('nan'):.0f} s")
    n_days = int(np.ceil(n_ecg / FS_VIL_ECG / 86400.0))
    for day in range(n_days):
        a, b = day * 86400.0, (day + 1) * 86400.0
        keep = (t_all >= a) & (t_all < b)
        if keep.sum() < 2000:
            continue
        i0, i1 = int(a * FS_VIL_ECG), min(n_ecg, int(b * FS_VIL_ECG))
        ecg = (ds[i0:i1, 0] if ds.shape[0] > ds.shape[-1] else ds[0, i0:i1]).astype(float)
        ppg = np.full(int(86400 * FS_VIL_PPG) + 1, np.nan)
        for (st, x) in P["pieces"]:
            s0 = int(round((st - a) * FS_VIL_PPG))
            lo, hi = max(0, s0), min(len(ppg), s0 + len(x))
            if hi > lo:
                ppg[lo:hi] = x[lo - s0: hi - s0]
        # beats without PPG get zero PPG features and SQI 0
        yield build_record(f"vil{rid}_d{day}", ecg, FS_VIL_ECG, ppg, FS_VIL_PPG, t_all[keep],
                           af[keep], window_min=window_min, offset_e=a, offset_p=a)
    P["file"].close()


def list_vilnius(root):
    ecgs = sorted(glob.glob(os.path.join(root, "*_ECG*.mat")))
    out = []
    for e in ecgs:
        pid = os.path.basename(e).split("_")[0]
        out.append((e, sorted(glob.glob(os.path.join(root, f"{pid}_PPG*.mat")))))
    return out


# ---------------------------------------------------------------------------------------------
def load_cinc2015_record(path_noext):
    """Return dict with ECG, PPG, fs, alarm type, true/false label, alarm time (300 s)."""
    import wfdb
    rec = wfdb.rdrecord(path_noext)
    names = [n.upper() for n in rec.sig_name]
    fs = float(rec.fs)
    ecg_idx = next((names.index(n) for n in ["II", "I", "III", "V", "AVF", "MCL"] if n in names), 0)
    ppg = rec.p_signal[:, names.index("PLETH")] if "PLETH" in names else None
    comments = [c.strip() for c in (rec.comments or [])]
    atype = comments[0] if comments else ""
    truth = None
    for c in comments:
        if "true" in c.lower():
            truth = 1
        elif "false" in c.lower():
            truth = 0
    return {"id": os.path.basename(path_noext), "ecg": rec.p_signal[:, ecg_idx], "ppg": ppg,
            "fs": fs, "type": atype, "truth": truth, "alarm_t": 300.0}
