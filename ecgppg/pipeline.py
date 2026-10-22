"""Record construction and experiment loops."""
import numpy as np

from .features import ecg_features, ppg_features, make_blocks
from .scores import fused_scores, P_FUSIONS, upper_tail
from .conformal import run_monitor, conformal_cusum, morphology_screen
from .metrics import record_metrics, pool

# calibration configurations: (name, mode, q, tau)
CALIBRATIONS = [
    ("labelfree", "labelfree", 0.0, None),
    ("screen6", "screen", 0.0, 6.0),
    ("adj0", "adjudicated", 0.00, None),
    ("adj5", "adjudicated", 0.05, None),
    ("adj10", "adjudicated", 0.10, None),
]


def build_record(rid, ecg, fs_e, ppg, fs_p, t, label, sinus=None, ecg_bad=None,
                 window_min=5.0, offset_e=0.0, offset_p=0.0):
    """Build a record dict.

    t      : R-peak times in seconds on the common time axis
    ecg    : ECG samples; sample k is at time offset_e + k/fs_e (same for ppg/offset_p)
    label  : 0 normal, 1 supraventricular, 2 ventricular, -1 unclassifiable
    The first `window_min` minutes of beats form the patient window (40% fitting, 60%
    calibration initialisation); evaluation starts at the end of the window.
    """
    t = np.asarray(t, float)
    block = make_blocks(t)
    rel = t - t[0]
    W = window_min * 60.0
    fit = rel < 0.4 * W
    cal = (rel >= 0.4 * W) & (rel < W)
    ev = rel >= W
    Xe = ecg_features(ecg, fs_e, t - offset_e, block, fit)
    if ppg is not None:
        Xp, sqi = ppg_features(ppg, fs_p, t - offset_p, block, fit)
    else:
        Xp, sqi = ppg_features(None, 1.0, t, block, fit)
    n = len(t)
    return {
        "id": rid, "t": t, "block": block, "label": np.asarray(label, int),
        "sinus": np.ones(n, bool) if sinus is None else np.asarray(sinus, bool),
        "ecg_bad": np.zeros(n, bool) if ecg_bad is None else np.asarray(ecg_bad, bool),
        "Xe": Xe.astype(np.float32), "Xp": Xp.astype(np.float32), "sqi": sqi.astype(np.float32),
        "fit_mask": fit, "cal_mask": cal, "eval_mask": ev,
    }


def save_record(r, path):
    np.savez_compressed(path, **{k: v for k, v in r.items()})


def load_record(path):
    z = np.load(path, allow_pickle=True)
    r = {k: z[k] for k in z.files}
    r["id"] = str(r["id"])
    return r


def auc_score_of(score, r):
    if isinstance(score, tuple):
        fit = r["fit_mask"]
        te = upper_tail(score[0], score[0][fit])
        tp = upper_tail(score[1], score[1][fit])
        return -np.log(np.minimum(te, tp))
    return score


def evaluate(records, models, fusions, calibrations, alpha=0.01, with_cusum=True,
             ep_min_len=3, keep_rows=False, M=300, G=20):
    """Run every (calibration, fusion) configuration on records. Returns dict -> pooled metrics."""
    results = {}
    rows = {}
    for r in records:
        raw = models.raw_scores(r)
        fs = fused_scores(raw, r)
        screens = {}
        for (cname, mode, q, tau) in calibrations:
            if mode == "screen" and tau not in screens:
                screens[tau] = morphology_screen(r, tau)
            for fu in fusions:
                sc = fs[fu]
                res = run_monitor(sc, r, alpha=alpha, mode=mode, q=q, M=M, G=G,
                                  pfusion=fu if fu in P_FUSIONS else None,
                                  screen=screens.get(tau))
                ep = conformal_cusum(res["p"])[0] if with_cusum else None
                row = record_metrics(r, res, auc_score_of(sc, r), ep, ep_min_len)
                rows.setdefault((cname, fu), []).append(row)
    for key, rr in rows.items():
        results[key] = pool(rr, alpha)
    if keep_rows:
        return results, rows
    return results


def select_configuration(results, alpha, calib_allowed=None):
    """Pre-specified rule: among configurations with pooled FAR <= 1.5 alpha, keep those within one
    percentage point of the best sensitivity and choose the fewest reviews per hour;
    if none qualifies, choose the lowest FAR."""
    keys = [k for k in results if calib_allowed is None or k[0] in calib_allowed]
    ok = [k for k in keys if results[k]["far_sinus"] <= 150.0 * alpha]
    if not ok:
        return min(keys, key=lambda k: results[k]["far_sinus"])
    best = max(results[k]["se"] for k in ok)
    near = [k for k in ok if results[k]["se"] >= best - 1.0]
    return min(near, key=lambda k: (np.nan_to_num(results[k]["rev_h"], nan=0.0), -results[k]["se"]))


class LazyRecords:
    """Iterable over cached records that loads one record at a time (for very large test sets)."""

    def __init__(self, paths):
        self.paths = list(paths)

    def __iter__(self):
        for pth in self.paths:
            yield load_record(pth)

    def __len__(self):
        return len(self.paths)
