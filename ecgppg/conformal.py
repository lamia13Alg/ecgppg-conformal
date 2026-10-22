"""Sliding conformal calibration with clinician adjudication, and conformal CUSUM."""
import math
import numpy as np
from bisect import bisect_left

from .scores import upper_tail


def morphology_screen(r, tau):
    """Label-free screen: True if the beat's ECG morphology is within tau robust SDs of the
    patient's median template (template and scale from fitting beats)."""
    M = r["Xe"][:, :64]
    tmpl = np.median(M[r["fit_mask"]], axis=0)
    d = np.linalg.norm(M - tmpl, axis=1)
    df = d[r["fit_mask"]]
    med = np.median(df)
    mad = 1.4826 * np.median(np.abs(df - med)) + 1e-9
    return (d - med) / mad < tau


def _pvals(score, buf_idx, i):
    if isinstance(score, tuple):
        out = []
        for s in score:
            b = s[buf_idx]
            out.append((1.0 + np.count_nonzero(b >= s[i])) / (len(b) + 1.0))
        return out
    b = score[buf_idx]
    return [(1.0 + np.count_nonzero(b >= score[i])) / (len(b) + 1.0)]


def _combine(ps, pfusion):
    if len(ps) == 1:
        return ps[0]
    if pfusion == "pbonf":
        return min(1.0, 2.0 * min(ps))
    if pfusion == "pmean":
        return min(1.0, sum(ps))
    raise ValueError(pfusion)


def _rank_score(score, idx):
    """Single score used to rank start-up beats for review."""
    if isinstance(score, tuple):
        te = upper_tail(score[0][idx], score[0][idx])
        tp = upper_tail(score[1][idx], score[1][idx])
        return -np.log(np.minimum(te, tp))
    return score[idx]


def run_monitor(score, r, alpha=0.01, mode="adjudicated", q=0.10, M=300, G=20,
                pfusion=None, screen=None, group_s=30.0, static=False):
    """Monitor one record.

    score : array of anomaly scores (higher = more abnormal), or a tuple (s_ecg, s_ppg) with
            pfusion in {"pbonf", "pmean"}.
    r     : record dict with t, label (0 normal, >0 abnormal, -1 unclassifiable), cal_mask, eval_mask.
    mode  : "adjudicated" | "labelfree" | "screen" (label-free morphology screen, needs `screen`)
    Labels are read only for beats that are reviewed (start-up review or alarm).
    """
    n = len(r["t"])
    lab = r["label"]
    t = r["t"]
    cap = n + 1
    buf = np.empty(cap, dtype=np.int64)
    nb = 0
    cal = np.where(r["cal_mask"])[0]
    startup_reviews = 0
    if mode == "adjudicated" and q > 0 and len(cal):
        k = int(math.ceil(q * len(cal)))
        rs = _rank_score(score, cal)
        top = cal[np.argsort(-rs)[:k]]
        startup_reviews = len(top)
        drop = set(top[lab[top] != 0].tolist())
        cal = np.array([c for c in cal if c not in drop], dtype=np.int64)
    if mode == "screen" and screen is not None:
        cal = cal[screen[cal]]
    buf[:len(cal)] = cal
    nb = len(cal)

    p = np.full(n, np.nan)
    alarm = np.zeros(n, dtype=bool)
    bound = np.full(n, np.nan)
    reviews = 0
    reviews_grouped = 0
    last_confirmed_t = -np.inf
    ev = np.where(r["eval_mask"])[0]
    for i in ev:
        k = bisect_left(buf[:nb], i - G) if nb else 0
        lo = max(0, k - M)
        bidx = buf[lo:k]
        if len(bidx) == 0:
            continue
        pi = _combine(_pvals(score, bidx, i), pfusion)
        p[i] = pi
        if lab[i] == 0:
            n0 = np.count_nonzero(lab[bidx] == 0)
            bound[i] = alpha * (len(bidx) + 1.0) / (n0 + 1.0)
        enter = True
        if pi <= alpha:
            alarm[i] = True
            if mode == "adjudicated":
                reviews += 1
                abnormal = lab[i] != 0
                if not (abnormal and t[i] - last_confirmed_t <= group_s):
                    reviews_grouped += 1
                if abnormal:
                    last_confirmed_t = t[i]
                    enter = False
        if mode == "screen" and screen is not None and not screen[i]:
            enter = False
        if enter and not static:
            buf[nb] = i
            nb += 1
    return {"p": p, "alarm": alarm, "bound": bound, "reviews": reviews,
            "reviews_grouped": reviews_grouped, "startup_reviews": startup_reviews}


def conformal_cusum(p, eps=0.2, h=math.log(4000)):
    """Page's CUSUM on conformal p-values with Beta(eps,1) alternative. NaN p-values are skipped.
    Returns a boolean array of episode alarms (statistic reset after each alarm) and the statistic."""
    S = 0.0
    out = np.zeros(len(p), dtype=bool)
    stat = np.zeros(len(p))
    for i, pi in enumerate(p):
        if not np.isfinite(pi):
            stat[i] = S
            continue
        S = max(0.0, S + math.log(eps * max(pi, 1e-12) ** (eps - 1.0)))
        if S > h:
            out[i] = True
            S = 0.0
        stat[i] = S
    return out, stat
