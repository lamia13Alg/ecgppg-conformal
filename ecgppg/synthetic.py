"""Synthetic ECG + PPG recordings, used only to test that the code runs end to end.
They are NOT used for any reported result."""
import numpy as np

from .pipeline import build_record


def _gauss(tt, mu, sd, a):
    return a * np.exp(-0.5 * ((tt - mu) / sd) ** 2)


def synth_record(rid, minutes=25, fs=250.0, p_v=0.03, p_s=0.03, noise_frac=0.05, seed=0,
                 window_min=5.0):
    rng = np.random.default_rng(seed)
    T = minutes * 60.0
    rr0 = rng.uniform(0.7, 1.0)
    t, lab = [], []
    cur = 1.0
    prev_ab = False
    while cur < T - 2:
        u = rng.random()
        if not prev_ab and u < p_v:
            cur_t = cur - 0.35 * rr0
            t.append(cur_t); lab.append(2)
            cur = cur_t + 1.6 * rr0 * rng.normal(1, 0.02)
            prev_ab = True
            continue
        if not prev_ab and u < p_v + p_s:
            cur_t = cur - 0.25 * rr0
            t.append(cur_t); lab.append(1)
            cur = cur_t + 1.1 * rr0 * rng.normal(1, 0.02)
            prev_ab = True
            continue
        t.append(cur); lab.append(0)
        cur += rr0 * rng.normal(1, 0.03)
        prev_ab = False
    t = np.array(t); lab = np.array(lab)
    n = int(T * fs)
    tt = np.arange(n) / fs
    ecg = rng.normal(0, 0.02, n)
    ppg = rng.normal(0, 0.01, n)
    pat = 0.25
    for ti, li in zip(t, lab):
        i0, i1 = int((ti - 0.4) * fs), int((ti + 0.9) * fs)
        i0, i1 = max(0, i0), min(n, i1)
        w = tt[i0:i1]
        if li == 2:
            ecg[i0:i1] += _gauss(w, ti, 0.04, -1.2) + _gauss(w, ti + 0.3, 0.08, 0.4)
            amp = 0.25
        else:
            ecg[i0:i1] += (_gauss(w, ti - 0.16, 0.025, 0.12) + _gauss(w, ti, 0.012, 1.0)
                           + _gauss(w, ti + 0.25, 0.05, 0.25))
            amp = 0.55 if li == 1 else 1.0
        ppg[i0:i1] += _gauss(w, ti + pat + 0.12, 0.09, amp) + _gauss(w, ti + pat + 0.38, 0.08, 0.3 * amp)
    bad = np.zeros(len(t), bool)
    k = int(noise_frac * T / 20)
    for _ in range(k):
        s = rng.uniform(6 * 60, T - 30)
        a, b = int(s * fs), int((s + 20) * fs)
        ecg[a:b] += rng.normal(0, 0.6, b - a)
        bad |= (t >= s) & (t < s + 20)
    return build_record(rid, ecg, fs, ppg, fs, t, lab, ecg_bad=bad, window_min=window_min)
