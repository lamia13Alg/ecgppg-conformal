"""Signal processing: filtering, QRS detection, PPG pulse detection, ECG-PPG pulse alignment."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks


def bandpass(x, fs, lo, hi, order=3):
    """Zero-phase Butterworth band-pass filter; NaNs are linearly interpolated first."""
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if bad.all():
        return np.zeros_like(x)
    if bad.any():
        idx = np.arange(len(x))
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    hi = min(hi, 0.45 * fs)
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def detect_qrs(ecg, fs):
    """Simple energy-based QRS detector (used only when no reference beat annotations exist).

    Returns R-peak sample indices refined to the local absolute maximum of the band-passed signal.
    """
    f = bandpass(ecg, fs, 5.0, 15.0, order=2)
    d = np.gradient(f) ** 2
    w = max(1, int(0.15 * fs))
    e = np.convolve(d, np.ones(w) / w, mode="same")
    thr = 0.3 * np.percentile(e, 98)
    peaks, _ = find_peaks(e, height=thr, distance=int(0.25 * fs))
    r = []
    half = int(0.08 * fs)
    for p in peaks:
        lo, hi = max(0, p - half), min(len(f), p + half)
        r.append(lo + int(np.argmax(np.abs(f[lo:hi]))))
    return np.unique(np.array(r, dtype=int))


def _pulses_in(f, fs, scale, offset=0):
    peaks, _ = find_peaks(f, distance=max(1, int(0.27 * fs)), prominence=0.15 * scale)
    df = np.gradient(f)
    feet, slopes, amps, keep = [], [], [], []
    look = int(0.5 * fs)
    for k, p in enumerate(peaks):
        lo = peaks[k - 1] if k > 0 else max(0, p - look)
        lo = max(lo, p - look)
        if p - lo < 3:
            continue
        foot = lo + int(np.argmin(f[lo:p]))
        if p - foot < 2:
            continue
        sl = foot + int(np.argmax(df[foot:p]))
        feet.append(foot + offset)
        slopes.append(sl + offset)
        amps.append(f[p] - f[foot])
        keep.append(p + offset)
    return keep, feet, slopes, amps


def _finite_runs(x, min_len):
    ok = np.isfinite(x)
    d = np.diff(np.r_[0, ok.astype(np.int8), 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(a, b) for a, b in zip(starts, ends) if b - a >= min_len]


def ppg_pulses(ppg, fs):
    """Detect PPG pulses on a band-passed (0.5-8 Hz) signal.

    Returns a dict of arrays (one entry per pulse): peak, foot and max-slope times in seconds,
    and pulse amplitude (peak minus foot). When the PPG has gaps longer than 2 s (wearable
    recordings), each uninterrupted piece is filtered and analysed separately and the filtered
    signal is set to 0 inside the gaps.
    """
    ppg = np.asarray(ppg, float)
    nan = ~np.isfinite(ppg)
    long_gaps = False
    if nan.any():
        d = np.diff(np.r_[0, nan.astype(np.int8), 0])
        long_gaps = np.max(np.where(d == -1)[0] - np.where(d == 1)[0]) > 2 * fs
    if not long_gaps:
        f = bandpass(ppg, fs, 0.5, 8.0, order=2)
        scale = np.percentile(f, 95) - np.percentile(f, 5) + 1e-12
        keep, feet, slopes, amps = _pulses_in(f, fs, scale)
    else:
        f = np.zeros(len(ppg))
        runs = _finite_runs(ppg, int(10 * fs))
        for a, b in runs:
            f[a:b] = bandpass(ppg[a:b], fs, 0.5, 8.0, order=2)
        vals = np.concatenate([f[a:b] for a, b in runs]) if runs else np.zeros(1)
        scale = np.percentile(vals, 95) - np.percentile(vals, 5) + 1e-12
        keep, feet, slopes, amps = [], [], [], []
        for a, b in runs:
            k1, f1, s1, a1 = _pulses_in(f[a:b], fs, scale, offset=a)
            keep += k1; feet += f1; slopes += s1; amps += a1
    return {
        "peak": np.array(keep, dtype=float) / fs,
        "foot": np.array(feet, dtype=float) / fs,
        "slope": np.array(slopes, dtype=float) / fs,
        "amp": np.array(amps, dtype=float),
        "filtered": f,
    }


def align_pulses(r_times, pulses, pat_min=0.08, pat_max=0.60):
    """Associate each R peak with the pulse it drove.

    A pulse is assigned to beat i if its max-slope point lies after t_i + pat_min and before
    min(t_i + pat_max, t_{i+1} + pat_min). Beats without a pulse (e.g. a premature ventricular beat
    with insufficient stroke volume, or a PPG artefact) get NaN.
    Returns arrays PAT (s), amplitude, and pulse peak time, one per beat.
    """
    n = len(r_times)
    pat = np.full(n, np.nan)
    amp = np.full(n, np.nan)
    ptime = np.full(n, np.nan)
    s = pulses["slope"]
    if len(s) == 0 or n == 0:
        return pat, amp, ptime
    r_times = np.asarray(r_times, float)
    upper = r_times + pat_max
    upper[:-1] = np.minimum(upper[:-1], r_times[1:] + pat_min)
    j = np.searchsorted(s, r_times + pat_min)
    jj = np.minimum(j, len(s) - 1)
    ok = (j < len(s)) & (s[jj] < upper)
    pat[ok] = s[jj[ok]] - r_times[ok]
    amp[ok] = pulses["amp"][jj[ok]]
    ptime[ok] = pulses["peak"][jj[ok]]
    return pat, amp, ptime


def resample_window(sig, fs, center_t, pre, post, n_out, mask_lo_t=None, mask_hi_t=None):
    """Linearly resample sig on [center_t - pre, center_t + post] to n_out samples.

    Samples before mask_lo_t or after mask_hi_t are set to the median of the unmasked samples
    (neighbour masking).
    """
    tt = center_t + np.linspace(-pre, post, n_out)
    idx = tt * fs
    lo = max(0, int(np.floor(idx[0])) - 1)
    hi = min(len(sig), int(np.ceil(idx[-1])) + 2)
    if hi - lo < 2:
        w = np.full(n_out, sig[min(max(lo, 0), len(sig) - 1)], dtype=float)
    else:
        seg = sig[lo:hi]
        w = np.interp(idx, np.arange(lo, hi), seg, left=seg[0], right=seg[-1])
    m = np.ones(n_out, dtype=bool)
    if mask_lo_t is not None:
        m &= tt >= mask_lo_t
    if mask_hi_t is not None:
        m &= tt <= mask_hi_t
    if m.any() and (~m).any():
        w[~m] = np.median(w[m])
    return w
