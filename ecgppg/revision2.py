"""Analyses added for the second revision.

* run_monitor_v2 : the sliding conformal monitor with a richer reviewer model
    - error_model "beat"  : independent errors per reviewed beat (as in revision.run_monitor_ext);
      "group" : one decision per review task (alarms closer than group_s seconds form one task),
                so all beats of a task share the reviewer's error;
      "score" : errors concentrated on borderline alarms (miss probability 2*fn_rate*p/alpha,
                false-confirmation probability 2*fp_rate*(1 - p/alpha)), i.e. correlated with the
                difficulty of the beat;
    - delay_model "fixed" (every decision arrives delay_s after the beat), "exp" (one exponential
      delay with mean delay_s per review task), "batch" (decisions released at the next multiple of
      delay_s seconds: a reviewer who checks the queue periodically);
    - pending "withhold" (a beat awaiting review is kept out of the buffer, as in the first
      revision) or "provisional" (it enters the buffer at once and is removed if confirmed abnormal);
    - avail < 1 with unreviewed "enter" (unreviewed alarm beats enter the buffer) or "withhold";
    - episode_exclusion: a confirmed abnormal alarm also removes from the buffer the unreviewed
      beats of the preceding group_s seconds (the reviewer adjudicates the strip, not the beat);
    - oracle_clean: the buffer admits only beats whose reference label is normal (decomposition of
      the sensitivity loss into contamination and intrinsic within-patient overlap).
  With the default arguments it reproduces revision.run_monitor_ext exactly (checked at run time).
* aci_monitor_v2 : adaptive conformal inference with the same reviewer model (errors, delays).
* review_tasks   : number of review tasks (groups of reviewed alarms separated by > group_s).
* ecg_features_bounded / ecg_features_causal : ECG features computed with a band-pass filter that
  sees at most `lookahead` seconds after each beat's chunk (zero-phase on a bounded segment), or with a
  strictly causal (forward-only) filter, to check what the zero-phase filtering implies for latency.
"""
import math
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from .conformal import _pvals, _combine, _rank_score
from .features import _windows, _rr_features, N_ECG_MORPH


# ---------------------------------------------------------------------------------------------
def run_monitor_v2(score, r, alpha=0.01, mode="adjudicated", q=0.05, M=300, G=20, pfusion=None,
                   fp_rate=0.0, fn_rate=0.0, error_model="beat", delay_s=0.0, delay_model="fixed",
                   pending_policy="withhold", avail=1.0, unreviewed="enter",
                   episode_exclusion=False, oracle_clean=False, seed=0, group_s=30.0):
    rng = np.random.default_rng(seed)
    n = len(r["t"])
    lab = r["label"]
    t = r["t"]
    cal = np.where(r["cal_mask"])[0]
    startup = 0
    if oracle_clean:
        cal = cal[lab[cal] == 0]
    elif mode == "adjudicated" and q > 0 and len(cal):
        k = int(math.ceil(q * len(cal)))
        rs = _rank_score(score, cal)
        top = cal[np.argsort(-rs)[:k]]
        startup = len(top)
        drop = set()
        for c in top:
            ab = lab[c] != 0
            said_ab = (rng.random() >= fn_rate) if ab else (rng.random() < fp_rate)
            if said_ab:
                drop.add(int(c))
        cal = np.array([c for c in cal if c not in drop], dtype=np.int64)
    cap = n + len(cal) + 1
    bufa = np.empty(cap, dtype=np.int64)       # sorted buffer of beat indices (numpy, for speed)
    bufa[:len(cal)] = np.sort(cal)
    nb = [len(cal)]
    pending = []            # (release time, index, said_abnormal, provisional_in_buffer)
    p = np.full(n, np.nan)
    alarm = np.zeros(n, dtype=bool)
    bound = np.full(n, np.nan)
    reviewed_normal = np.zeros(n, dtype=bool)
    st = {"reviews": 0, "tasks": 0, "rev_ab": 0, "miss_ab": 0, "rev_nor": 0, "fpconf_nor": 0,
          "excluded_normal": 0}
    g_last = -np.inf
    g_u_fn = g_u_fp = 0.0
    g_t0 = 0.0
    g_delay = 0.0
    t_start = t[0]
    single = not isinstance(score, tuple)
    lab0 = lab == 0

    def _append(j):
        k = nb[0]
        if k == 0 or bufa[k - 1] < j:
            bufa[k] = j
        else:                                   # out-of-order insertion (delayed decision)
            pos = int(np.searchsorted(bufa[:k], j))
            bufa[pos + 1:k + 1] = bufa[pos:k].copy()
            bufa[pos] = j
        nb[0] = k + 1

    def _remove_range(a, b, keep_mask=None):
        k = nb[0]
        seg = bufa[a:b]
        kept = seg[keep_mask[seg]] if keep_mask is not None else seg[:0]
        tail = bufa[b:k].copy()
        m = len(kept)
        bufa[a:a + m] = kept
        bufa[a + m:a + m + len(tail)] = tail
        nb[0] = a + m + len(tail)

    def _apply(j, said_ab, prov):
        if said_ab:
            if prov:
                k = nb[0]
                pos = int(np.searchsorted(bufa[:k], j))
                if pos < k and bufa[pos] == j:
                    _remove_range(pos, pos + 1)
            if episode_exclusion:
                j0 = int(np.searchsorted(t, t[j] - group_s, side="left"))
                k = nb[0]
                a = int(np.searchsorted(bufa[:k], j0))
                b = int(np.searchsorted(bufa[:k], j))
                if b > a:
                    _remove_range(a, b, reviewed_normal)
        else:
            reviewed_normal[j] = True
            if not prov:
                _append(j)

    for i in np.where(r["eval_mask"])[0]:
        if pending:
            keep = []
            for item in pending:
                if item[0] <= t[i]:
                    _apply(item[1], item[2], item[3])
                else:
                    keep.append(item)
            pending = keep
        k = int(np.searchsorted(bufa[:nb[0]], i - G))
        if k == 0:
            continue
        bidx = bufa[max(0, k - M):k]
        if single:
            pi = (1.0 + np.count_nonzero(score[bidx] >= score[i])) / (len(bidx) + 1.0)
        else:
            pi = _combine(_pvals(score, bidx, i), pfusion)
        p[i] = pi
        if lab0[i]:
            n0 = np.count_nonzero(lab0[bidx])
            bound[i] = alpha * (len(bidx) + 1.0) / (n0 + 1.0)
        if oracle_clean:
            if pi <= alpha:
                alarm[i] = True
            if lab0[i]:
                _append(int(i))
            continue
        if pi > alpha:
            _append(int(i))
            continue
        alarm[i] = True
        if mode != "adjudicated":
            _append(int(i))
            continue
        if not (rng.random() < avail):
            if unreviewed == "enter":
                _append(int(i))
            continue
        # ---- reviewed alarm
        st["reviews"] += 1
        new_task = t[i] - g_last > group_s
        if new_task:
            st["tasks"] += 1
            g_t0 = t[i]
            if error_model == "group":
                g_u_fn, g_u_fp = rng.random(), rng.random()
            if delay_model == "exp" and delay_s > 0:
                g_delay = rng.exponential(delay_s)
        g_last = t[i]
        ab = not lab0[i]
        if error_model == "beat":
            said_ab = (rng.random() >= fn_rate) if ab else (rng.random() < fp_rate)
        elif error_model == "group":
            said_ab = (g_u_fn >= fn_rate) if ab else (g_u_fp < fp_rate)
        elif error_model == "score":
            ratio = min(1.0, max(0.0, pi / alpha))
            if ab:
                said_ab = rng.random() >= min(1.0, 2.0 * fn_rate * ratio)
            else:
                said_ab = rng.random() < min(1.0, 2.0 * fp_rate * (1.0 - ratio))
        else:
            raise ValueError(error_model)
        if ab:
            st["rev_ab"] += 1
            st["miss_ab"] += int(not said_ab)
        else:
            st["rev_nor"] += 1
            st["fpconf_nor"] += int(said_ab)
            st["excluded_normal"] += int(said_ab)
        if delay_s <= 0:
            release = t[i]
        elif delay_model == "fixed":
            release = t[i] + delay_s
        elif delay_model == "exp":
            release = max(t[i], g_t0 + g_delay)
        elif delay_model == "batch":
            release = t_start + math.ceil((t[i] - t_start) / delay_s + 1e-9) * delay_s
        else:
            raise ValueError(delay_model)
        prov = pending_policy == "provisional"
        if prov:
            _append(int(i))
        if release <= t[i]:
            _apply(int(i), said_ab, prov)
        else:
            pending.append((release, int(i), said_ab, prov))
    out = {"p": p, "alarm": alarm, "bound": bound, "startup_reviews": startup}
    out.update(st)
    return out


# ---------------------------------------------------------------------------------------------
def aci_monitor_v2(score, r, cal_scores, alpha=0.01, gamma=0.005, lo=1e-4, hi=0.2,
                   fp_rate=0.0, fn_rate=0.0, error_model="beat", delay_s=0.0, delay_model="fixed",
                   seed=0, group_s=30.0):
    """ACI on a fixed population calibration set. Every beat moves the working level up by
    gamma*alpha; an alarm that the reviewer declares normal moves it down by gamma when the
    decision becomes known. Beats without alarm are never reviewed and give no error signal (the
    same information as the patient-specific monitor). With no error and no delay this is
    revision.aci_monitor exactly."""
    rng = np.random.default_rng(seed)
    cal = np.sort(np.asarray(cal_scores, float))
    m = len(cal)
    n = len(r["t"])
    lab = r["label"]
    t = r["t"]
    p = np.full(n, np.nan)
    alarm = np.zeros(n, dtype=bool)
    a = alpha
    pending = []
    g_last = -np.inf
    g_u_fn = g_u_fp = 0.0
    g_t0 = g_delay = 0.0
    t_start = t[0]
    tasks = 0
    for i in np.where(r["eval_mask"])[0]:
        if pending:
            keep = []
            for (tk, err) in pending:
                if tk <= t[i]:
                    a = min(hi, max(lo, a - gamma * err))
                else:
                    keep.append((tk, err))
            pending = keep
        cnt = m - np.searchsorted(cal, score[i], side="left")
        pi = (1.0 + cnt) / (m + 1.0)
        p[i] = pi
        err = 0.0
        release = t[i]
        if pi <= a:
            alarm[i] = True
            if t[i] - g_last > group_s:
                tasks += 1
                g_t0 = t[i]
                if error_model == "group":
                    g_u_fn, g_u_fp = rng.random(), rng.random()
                if delay_model == "exp" and delay_s > 0:
                    g_delay = rng.exponential(delay_s)
            g_last = t[i]
            ab = lab[i] != 0
            if fp_rate > 0 or fn_rate > 0:
                if error_model == "group":
                    said_ab = (g_u_fn >= fn_rate) if ab else (g_u_fp < fp_rate)
                else:
                    said_ab = (rng.random() >= fn_rate) if ab else (rng.random() < fp_rate)
            else:
                said_ab = ab
            err = 0.0 if said_ab else 1.0
            if delay_s > 0:
                if delay_model == "fixed":
                    release = t[i] + delay_s
                elif delay_model == "exp":
                    release = max(t[i], g_t0 + g_delay)
                else:
                    release = t_start + math.ceil((t[i] - t_start) / delay_s + 1e-9) * delay_s
        if release <= t[i]:
            a = min(hi, max(lo, a + gamma * (alpha - err)))
        else:
            a = min(hi, max(lo, a + gamma * alpha))
            pending.append((release, err))
    return {"p": p, "alarm": alarm, "tasks": tasks, "reviews": int(alarm.sum())}


def review_tasks(alarm, t, eval_mask, group_s=30.0):
    idx = np.where(alarm & eval_mask)[0]
    if len(idx) == 0:
        return 0
    tt = t[idx]
    return int(1 + np.count_nonzero(np.diff(tt) > group_s))


# ---------------------------------------------------------------------------------------------
def _interp_nan(x):
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if bad.all():
        return np.zeros_like(x)
    if bad.any():
        idx = np.arange(len(x))
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def _finish(M, t, block, fit_mask):
    M = M - np.median(M, axis=1, keepdims=True)
    centre = int(round(0.250 / 0.700 * (N_ECG_MORPH - 1)))
    ramp = np.median(np.abs(M[fit_mask, centre])) if fit_mask.any() else 1.0
    M = M / (ramp + 1e-9)
    prem, pause, _, _ = _rr_features(t, block)
    return np.column_stack([M, prem, pause])


def ecg_features_bounded(ecg, fs, t, block, fit_mask, chunk_s=0.5, lookahead_s=1.0, preroll_s=20.0,
                         lo=0.5, hi=40.0, order=3):
    """ECG features where the zero-phase band-pass of the beats of each chunk [a, a+chunk_s) only
    uses samples in [a - preroll_s, a + chunk_s + lookahead_s]: every beat's features depend on at
    most chunk_s + lookahead_s seconds after its R peak (plus the next R peak, as before)."""
    ecg = np.asarray(ecg, float)
    hi_ = min(hi, 0.45 * fs)
    sos = butter(order, [lo, hi_], btype="band", fs=fs, output="sos")
    n = len(t)
    M = np.zeros((n, N_ECG_MORPH))
    same_prev = np.r_[False, block[1:] == block[:-1]]
    same_next = np.r_[block[1:] == block[:-1], False]
    lo_t = np.where(same_prev, np.r_[np.nan, t[:-1]] + 0.350, -np.inf)
    hi_t = np.where(same_next, np.r_[t[1:], np.nan] - 0.100, np.inf)
    starts = np.floor(t / chunk_s).astype(np.int64)
    uniq, first = np.unique(starts, return_index=True)
    bounds = list(first) + [n]
    padlen = 3 * (2 * len(sos) + 1)
    for c in range(len(uniq)):
        a_i, b_i = bounds[c], bounds[c + 1]
        a_t = uniq[c] * chunk_s
        s0 = max(0, int(np.floor((a_t - preroll_s) * fs)))
        s1 = min(len(ecg), int(np.ceil((a_t + chunk_s + lookahead_s) * fs)))
        seg = _interp_nan(ecg[s0:s1])
        if len(seg) <= padlen:
            f = seg - np.mean(seg) if len(seg) else seg
        else:
            f = sosfiltfilt(sos, seg)
        off = s0 / fs
        tt = t[a_i:b_i]
        M[a_i:b_i] = _windows(f, fs, tt - off, 0.250, 0.450, N_ECG_MORPH,
                              lo_t[a_i:b_i] - off, hi_t[a_i:b_i] - off)
    return _finish(M, t, block, fit_mask)


def ecg_features_causal(ecg, fs, t, block, fit_mask, lo=0.5, hi=40.0, order=3):
    """ECG features with a strictly causal (forward-only) Butterworth band-pass of the same order."""
    hi_ = min(hi, 0.45 * fs)
    sos = butter(order, [lo, hi_], btype="band", fs=fs, output="sos")
    f = sosfilt(sos, _interp_nan(ecg))
    same_prev = np.r_[False, block[1:] == block[:-1]]
    same_next = np.r_[block[1:] == block[:-1], False]
    lo_t = np.where(same_prev, np.r_[np.nan, t[:-1]] + 0.350, -np.inf)
    hi_t = np.where(same_next, np.r_[t[1:], np.nan] - 0.100, np.inf)
    M = _windows(f, fs, t, 0.250, 0.450, N_ECG_MORPH, lo_t, hi_t)
    return _finish(M, t, block, fit_mask)
