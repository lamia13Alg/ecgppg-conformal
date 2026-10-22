"""PPG witness for ECG alarms (PhysioNet/CinC 2015, external test 2).

An ECG alarm is suppressed only if the PPG in the seconds before the alarm is (i) of good quality
and (ii) typical of the same patient's PPG earlier in the record, as measured by a conformal
p-value. Pulse windows are summarised by: pulse rate, pulse-interval irregularity, longest
pulse gap, median amplitude relative to the patient reference, and template correlation.
The rule is fixed a priori (beta = 0.10, SQI >= 0.5); a sweep over beta is reported as sensitivity
analysis only.
"""
import numpy as np

from .signal import ppg_pulses

WIN = 8.0
STEP = 2.0
FEATS = ["rate", "irreg", "maxgap", "amp", "sqi"]
SD_FLOOR = np.array([2.0, 0.02, 0.2, 0.05, 0.02])  # bpm, -, s, -, -


def _window_feats(pk, amp, corr, a, b, amp_ref):
    m = (pk >= a) & (pk < b)
    p = pk[m]
    if len(p) < 2:
        gap = b - a
        return np.array([60.0 * len(p) / (b - a), 0.0, gap, 0.0, 0.0])
    ppi = np.diff(p)
    edges = np.r_[p[0] - a, ppi, b - p[-1]]
    return np.array([60.0 / np.median(ppi), np.std(ppi) / (np.mean(ppi) + 1e-9), edges.max(),
                     np.median(amp[m]) / (amp_ref + 1e-12), np.median(corr[m])])


def ppg_witness(ppg, fs, alarm_t, ref_end=120.0, guard=20.0):
    """Return (p_value, sqi_of_alarm_window). p small = PPG atypical (supports a true alarm)."""
    if ppg is None or np.isfinite(ppg).mean() < 0.5:
        return 0.0, 0.0
    P = ppg_pulses(ppg, fs)
    pk, amp, f = P["peak"], P["amp"], P["filtered"]
    if len(pk) < 10:
        return 0.0, 0.0
    # pulse template from the reference period
    half = int(0.4 * fs)
    def seg(t):
        k = int(t * fs)
        return f[max(0, k - half): k + half] if k - half >= 0 and k + half <= len(f) else None
    ref = [seg(t) for t in pk[pk < ref_end]]
    ref = [s for s in ref if s is not None]
    if len(ref) < 5:
        return 0.0, 0.0
    tmpl = np.median(np.vstack(ref), axis=0)
    tc = tmpl - tmpl.mean()
    corr = np.zeros(len(pk))
    for j, t in enumerate(pk):
        s = seg(t)
        if s is not None:
            sc = s - s.mean()
            corr[j] = sc @ tc / (np.linalg.norm(sc) * np.linalg.norm(tc) + 1e-12)
    amp_ref = np.median(amp[pk < ref_end])
    cal = []
    a = 0.0
    while a + WIN <= alarm_t - guard:
        cal.append(_window_feats(pk, amp, corr, a, a + WIN, amp_ref))
        a += STEP
    test = _window_feats(pk, amp, corr, alarm_t - WIN, alarm_t, amp_ref)
    cal = np.array(cal)
    mu = np.median(cal, axis=0)
    sd = np.maximum(1.4826 * np.median(np.abs(cal - mu), axis=0), SD_FLOOR)
    # robust diagonal distance (windows overlap and features are few: no full covariance)
    s_cal = (((cal - mu) / sd) ** 2).sum(1)
    s_test = (((test - mu) / sd) ** 2).sum()
    p = (1.0 + np.count_nonzero(s_cal >= s_test)) / (len(s_cal) + 1.0)
    return float(p), float(test[4])


def decide(p, sqi, beta=0.10, sqi_min=0.5):
    """True = keep the alarm. Suppress only when the PPG is reliable and typical."""
    return not (sqi >= sqi_min and p > beta)


def challenge_score(tp, tn, fp, fn):
    d = tp + tn + fp + 5 * fn
    return 100.0 * (tp + tn) / d if d else float("nan")
