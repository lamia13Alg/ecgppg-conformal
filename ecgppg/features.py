"""Neighbour-isolated beat representation for ECG and PPG.

ECG (as in the single-modality version of the method):
  * 64-sample morphology window [-250, +450] ms around the R peak, masked before t_{i-1}+350 ms
    and after t_{i+1}-100 ms, divided by the patient's median |R| amplitude on fitting beats;
  * one-sided rhythm features: prematurity = max(0, 1 - RRpre/RRref), pause = max(0, RRpost/RRref - 1).

PPG (new), all one-sided or referenced to the patient's own fitting beats:
  * pulse deficit (1 if no pulse could be associated with the beat);
  * pulse amplitude deficit = max(0, -log(A_i / A_ref)), A_ref = median amplitude of the
    previous 10 associated pulses (a premature beat ejects less blood);
  * PAT deviation = |PAT_i - median PAT of fitting beats|;
  * pulse-interval prematurity and pause, computed on pulse peak times like the ECG features
    (they remain informative when the ECG is noisy);
  * 32-sample PPG window [0, 800] ms after the R peak, masked after t_{i+1}+100 ms, divided by
    the median pulse amplitude of fitting beats;
  * PPG signal-quality index: correlation of the beat's PPG window with the patient's median
    PPG template (fitting beats). It is used for gating, never as evidence of abnormality.
All patient-specific references depend only on the fitting beats or on past beats, so every
feature is a fixed function of the beat and of data preceding calibration -> conformal validity
is preserved.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .signal import bandpass, ppg_pulses, align_pulses

N_ECG_MORPH = 64
N_PPG_MORPH = 32


def _past_median(x, k=10, min_count=3):
    """out[i] = median of the finite values of x[i-k:i] if there are >= min_count, else NaN."""
    n = len(x)
    pad = np.r_[np.full(k, np.nan), np.asarray(x, float)]
    W = sliding_window_view(pad, k)[:n]          # row i = pad[i:i+k] = x[i-k:i]
    cnt = np.isfinite(W).sum(1)
    out = np.full(n, np.nan)
    ok = cnt >= min_count
    if ok.any():
        out[ok] = np.nanmedian(W[ok], axis=1)
    return out


def _windows(sig, fs, t, pre, post, n_out, lo_t=None, hi_t=None, chunk=20000):
    """Vectorised equivalent of signal.resample_window for all beats (rows)."""
    grid = np.linspace(-pre, post, n_out)
    xp = np.arange(len(sig))
    out = np.empty((len(t), n_out))
    for a in range(0, len(t), chunk):
        b = min(len(t), a + chunk)
        tt = t[a:b, None] + grid[None, :]
        w = np.interp((tt * fs).ravel(), xp, sig, left=sig[0], right=sig[-1]).reshape(tt.shape)
        m = np.ones_like(w, dtype=bool)
        if lo_t is not None:
            m &= tt >= lo_t[a:b, None]
        if hi_t is not None:
            m &= tt <= hi_t[a:b, None]
        part = (~m).any(1) & m.any(1)
        if part.any():
            wm = np.where(m[part], w[part], np.nan)
            med = np.nanmedian(wm, axis=1)
            w[part] = np.where(m[part], w[part], med[:, None])
        out[a:b] = w
    return out


def _rr_features(t, block, nref=10):
    n = len(t)
    rr_pre = np.full(n, np.nan)
    rr_post = np.full(n, np.nan)
    same_prev = np.r_[False, block[1:] == block[:-1]]
    same_next = np.r_[block[1:] == block[:-1], False]
    d = np.diff(t)
    rr_pre[1:][same_prev[1:]] = d[same_prev[1:]]
    rr_post[:-1][same_next[:-1]] = d[same_next[:-1]]
    ref = _past_median(rr_pre, nref, 3)
    glob = np.nanmedian(rr_pre) if np.isfinite(rr_pre).any() else 0.8
    ref = np.where(np.isfinite(ref), ref, glob)
    prem = np.maximum(0.0, 1.0 - rr_pre / ref)
    pause = np.maximum(0.0, rr_post / ref - 1.0)
    return np.nan_to_num(prem), np.nan_to_num(pause), rr_pre, rr_post


def ecg_features(ecg, fs, t, block, fit_mask):
    """Return (n, 66) ECG features: 64 masked morphology samples + prematurity + pause."""
    f = bandpass(ecg, fs, 0.5, 40.0)
    n = len(t)
    same_prev = np.r_[False, block[1:] == block[:-1]]
    same_next = np.r_[block[1:] == block[:-1], False]
    lo = np.where(same_prev, np.r_[np.nan, t[:-1]] + 0.350, -np.inf)
    hi = np.where(same_next, np.r_[t[1:], np.nan] - 0.100, np.inf)
    M = _windows(f, fs, t, 0.250, 0.450, N_ECG_MORPH, lo, hi)
    M -= np.median(M, axis=1, keepdims=True)
    centre = int(round(0.250 / 0.700 * (N_ECG_MORPH - 1)))
    ramp = np.median(np.abs(M[fit_mask, centre])) if fit_mask.any() else 1.0
    M /= ramp + 1e-9
    prem, pause, _, _ = _rr_features(t, block)
    return np.column_stack([M, prem, pause])


PPG_SCALAR_NAMES = ["deficit", "amp_deficit", "pat_dev", "ppi_prem", "ppi_pause"]


def ppg_features(ppg, fs, t, block, fit_mask):
    """Return (ppg_feats (n, 5 + 32), sqi (n,)). ppg may be None (all features set to 0, sqi 0)."""
    n = len(t)
    if ppg is None or not np.isfinite(ppg).any():
        return np.zeros((n, len(PPG_SCALAR_NAMES) + N_PPG_MORPH)), np.zeros(n)
    pulses = ppg_pulses(ppg, fs)
    f = pulses["filtered"]
    pat, amp, ptime = align_pulses(t, pulses)
    deficit = (~np.isfinite(pat)).astype(float)
    # amplitude referenced to the previous 10 associated pulses
    ref_amp = _past_median(amp, 10, 3)
    ok = np.isfinite(amp) & np.isfinite(ref_amp)
    amp_def = np.zeros(n)
    amp_def[ok] = np.maximum(0.0, -np.log(amp[ok] / (ref_amp[ok] + 1e-12) + 1e-12))
    amp_def = np.minimum(amp_def, 5.0)
    pat_ref = np.nanmedian(pat[fit_mask]) if np.isfinite(pat[fit_mask]).any() else np.nanmedian(pat)
    pat_dev = np.nan_to_num(np.abs(pat - pat_ref), nan=0.0)
    # pulse-interval features on associated pulse peaks (NaN pulses break the chain)
    pt = np.where(np.isfinite(ptime), ptime, np.nan)
    ppi_prem = np.zeros(n)
    ppi_pause = np.zeros(n)
    valid = np.isfinite(pt)
    if valid.sum() > 5:
        idx = np.where(valid)[0]
        tb = pt[idx]
        bb = block[idx]
        pr, pa, _, _ = _rr_features(tb, bb)
        ppi_prem[idx] = pr
        ppi_pause[idx] = pa
    same_next = np.r_[block[1:] == block[:-1], False]
    hi = np.where(same_next, np.r_[t[1:], np.nan] + 0.100, np.inf)
    W = _windows(f, fs, t, 0.0, 0.800, N_PPG_MORPH, None, hi)
    W -= W[:, :1]
    a_ref = np.nanmedian(amp[fit_mask]) if np.isfinite(amp[fit_mask]).any() else np.nanmedian(amp)
    if not np.isfinite(a_ref) or a_ref <= 0:
        a_ref = np.std(f) + 1e-9
    W /= a_ref
    tmpl = np.median(W[fit_mask], axis=0) if fit_mask.any() else np.median(W, axis=0)
    Wc = W - W.mean(1, keepdims=True)
    tc = tmpl - tmpl.mean()
    sqi = (Wc @ tc) / (np.linalg.norm(Wc, axis=1) * np.linalg.norm(tc) + 1e-12)
    feats = np.column_stack([deficit, amp_def, pat_dev, ppi_prem, ppi_pause, W])
    # beats where the PPG was not recorded (NaN at or right after the R peak): no PPG evidence
    k0 = np.clip((t * fs).astype(int), 0, len(ppg) - 1)
    k1 = np.clip(((t + 0.6) * fs).astype(int), 0, len(ppg) - 1)
    avail = np.isfinite(ppg[k0]) & np.isfinite(ppg[k1])
    feats[~avail] = 0.0
    sqi = np.where(avail, sqi, 0.0)
    return feats, sqi


def make_blocks(t, max_gap=3.0):
    """Split beat times into contiguous blocks (gap > max_gap seconds starts a new block)."""
    return np.r_[0, np.cumsum(np.diff(t) > max_gap)].astype(int)
