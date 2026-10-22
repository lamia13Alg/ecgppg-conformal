"""Analyses added for the major revision.

* run_monitor_ext : the sliding conformal monitor with an imperfect reviewer (false confirmations,
  missed confirmations, delayed confirmations, limited availability) and a fully automatic variant.
  With a perfect reviewer it reproduces conformal.run_monitor exactly.
* aci_monitor     : adaptive conformal inference (Gibbs & Candes 2021) on a population calibration
  set, with the level updated online from the reviewer's feedback on alarms.
* notifications   : alarm events actually presented to staff (refractory grouping).
* tune_cusum_h    : CUSUM threshold chosen on validation patients for a target false-episode rate.
* hierarchical patient bootstrap of pooled metrics and paired differences.
"""
import math
from bisect import bisect_left, insort

import numpy as np

from .conformal import _pvals, _combine, _rank_score


# ---------------------------------------------------------------------------------------------
def run_monitor_ext(score, r, alpha=0.01, mode="adjudicated", q=0.10, M=300, G=20, pfusion=None,
                    fp_rate=0.0, fn_rate=0.0, delay_s=0.0, avail=1.0, auto_exclude=False,
                    static=False, seed=0, group_s=30.0):
    """Sliding conformal monitor with a simulated reviewer.

    fp_rate : probability that a reviewed NORMAL beat is wrongly confirmed abnormal (excluded).
    fn_rate : probability that a reviewed ABNORMAL beat is missed (enters the buffer).
    delay_s : confirmation delay; a reviewed beat enters (or not) the buffer only after delay_s.
    avail   : probability that an alarm is reviewed at all; unreviewed alarm beats enter the buffer.
    auto_exclude : no reviewer; every alarm beat is excluded from the buffer (self-training).
    Start-up review (fraction q of the highest-scoring calibration beats) is subject to the same
    fp/fn errors.
    """
    rng = np.random.default_rng(seed)
    n = len(r["t"])
    lab = r["label"]
    t = r["t"]
    cal = np.where(r["cal_mask"])[0]
    startup = 0
    if mode == "adjudicated" and q > 0 and len(cal) and not auto_exclude:
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
    buf = list(cal.tolist())
    pending = []        # (time when decision is known, index, enters)
    p = np.full(n, np.nan)
    alarm = np.zeros(n, dtype=bool)
    bound = np.full(n, np.nan)
    reviews = reviews_grouped = 0
    last_conf = -np.inf
    excluded_normal = 0
    for i in np.where(r["eval_mask"])[0]:
        # release pending decisions whose confirmation has arrived
        if pending:
            keep = []
            for (tk, j, enters) in pending:
                if tk <= t[i]:
                    if enters and not static:
                        insort(buf, j)
                else:
                    keep.append((tk, j, enters))
            pending = keep
        k = bisect_left(buf, i - G)
        bidx = np.asarray(buf[max(0, k - M):k], dtype=np.int64)
        if len(bidx) == 0:
            continue
        pi = _combine(_pvals(score, bidx, i), pfusion)
        p[i] = pi
        if lab[i] == 0:
            n0 = np.count_nonzero(lab[bidx] == 0)
            bound[i] = alpha * (len(bidx) + 1.0) / (n0 + 1.0)
        enters = True
        delay = 0.0
        if pi <= alpha:
            alarm[i] = True
            if auto_exclude:
                enters = False
            elif mode == "adjudicated" and rng.random() < avail:
                reviews += 1
                ab = lab[i] != 0
                if not (ab and t[i] - last_conf <= group_s):
                    reviews_grouped += 1
                said_ab = (rng.random() >= fn_rate) if ab else (rng.random() < fp_rate)
                if said_ab:
                    last_conf = t[i]
                    enters = False
                    if not ab:
                        excluded_normal += 1
                delay = delay_s
        if static:
            continue
        if delay > 0:
            pending.append((t[i] + delay, int(i), enters))
        elif enters:
            buf.append(int(i))   # i is larger than every buffered index
    return {"p": p, "alarm": alarm, "bound": bound, "reviews": reviews,
            "reviews_grouped": reviews_grouped, "startup_reviews": startup,
            "excluded_normal": excluded_normal}


# ---------------------------------------------------------------------------------------------
def aci_monitor(score, r, cal_scores, alpha=0.01, gamma=0.005, lo=1e-4, hi=0.2):
    """Adaptive conformal inference on a fixed population calibration set (normal beats of the
    validation patients). The working level a_t is updated after each beat:
      a_{t+1} = clip(a_t + gamma * (alpha - err_t)),  err_t = 1 if the beat raised an alarm that
    the reviewer found normal (a confirmed false alarm), else 0.
    Only alarms are reviewed, as in the adjudicated monitor."""
    cal = np.sort(np.asarray(cal_scores, float))
    m = len(cal)
    n = len(r["t"])
    lab = r["label"]
    p = np.full(n, np.nan)
    alarm = np.zeros(n, dtype=bool)
    a = alpha
    level = np.full(n, np.nan)
    for i in np.where(r["eval_mask"])[0]:
        cnt = m - np.searchsorted(cal, score[i], side="left")
        pi = (1.0 + cnt) / (m + 1.0)
        p[i] = pi
        level[i] = a
        err = 0.0
        if pi <= a:
            alarm[i] = True
            err = 1.0 if lab[i] == 0 else 0.0
        a = min(hi, max(lo, a + gamma * (alpha - err)))
    return {"p": p, "alarm": alarm, "bound": np.full(n, np.nan), "reviews": int(alarm.sum()),
            "reviews_grouped": int(alarm.sum()), "startup_reviews": 0, "level": level}


def per_patient_normalised(score, r, thr):
    """Non-conformal personalisation: score standardised by the median/MAD of the patient window
    (fitting + calibration beats), compared with a global threshold thr chosen on validation."""
    w = r["fit_mask"] | r["cal_mask"]
    med = np.median(score[w])
    mad = 1.4826 * np.median(np.abs(score[w] - med)) + 1e-6
    z = (score - med) / mad
    alarm = (z > thr) & r["eval_mask"]
    return z, alarm


# ---------------------------------------------------------------------------------------------
def notifications(alarm, t, label, eval_mask, refractory=60.0, target=(1, 2)):
    """Group beat alarms into notifications presented to staff: an alarm opens a notification
    unless a notification was opened less than `refractory` seconds before. A notification is true
    if a target beat occurs within its refractory window. Returns (n_true, n_false)."""
    idx = np.where(alarm & eval_mask)[0]
    tgt = np.isin(label, target)
    n_true = n_false = 0
    last = -np.inf
    for i in idx:
        if t[i] - last < refractory:
            continue
        last = t[i]
        j1 = np.searchsorted(t, t[i] + refractory)
        if tgt[i:j1].any():
            n_true += 1
        else:
            n_false += 1
    return n_true, n_false


def cusum_alarm(p, eps=0.2, h=math.log(4000), refractory_beats=0):
    S = 0.0
    out = np.zeros(len(p), dtype=bool)
    hold = 0
    for i, pi in enumerate(p):
        if not np.isfinite(pi):
            continue
        if hold > 0:
            hold -= 1
            continue
        S = max(0.0, S + math.log(eps * max(pi, 1e-12) ** (eps - 1.0)))
        if S > h:
            out[i] = True
            S = 0.0
            hold = refractory_beats
    return out


def episodes_metrics(abn_seq, al_seq, t_seq, min_len=3, max_gap=10, tail=5):
    from .metrics import episodes
    eps = episodes(abn_seq, min_len=min_len, max_gap=max_gap)
    det, delays = 0, []
    in_ep = np.zeros(len(abn_seq), bool)
    for (f, l) in eps:
        in_ep[f:min(len(abn_seq), l + tail + 1)] = True
        w = np.where(al_seq[f:min(len(al_seq), l + tail + 1)])[0]
        if len(w):
            det += 1
            delays.append(t_seq[f + w[0]] - t_seq[f])
    return len(eps), det, delays, int((al_seq & ~in_ep).sum())


def tune_cusum_h(p_list, abn_list, t_list, target_false_per_h=1.0, eps=0.2, refractory_beats=60,
                 grid=None):
    """Smallest h on the grid whose false episode alarm rate on validation is <= target."""
    grid = grid or [math.log(x) for x in (1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 1e7)]
    hours = sum((tt[-1] - tt[0]) / 3600 for tt in t_list if len(tt) > 1)
    for h in grid:
        fa = 0
        for p, ab, tt in zip(p_list, abn_list, t_list):
            al = cusum_alarm(p, eps, h, refractory_beats)
            fa += episodes_metrics(ab, al, tt)[3]
        if fa / max(hours, 1e-9) <= target_false_per_h:
            return h
    return grid[-1]


# ---------------------------------------------------------------------------------------------
def counts_row(r, alarm, target=(1, 2), sinus_only=True):
    """Per-record counts used by the bootstrap."""
    ev = r["eval_mask"] & (r["label"] >= 0) & (r["label"] <= 2)
    lab = r["label"]
    tgt = ev & np.isin(lab, target)
    nor = ev & (lab == 0) & (r["sinus"] if sinus_only else True)
    return {"tp": int((tgt & alarm).sum()), "n_tgt": int(tgt.sum()),
            "fp": int((nor & alarm).sum()), "n_nor": int(nor.sum()),
            "alarms_all": int((ev & alarm).sum())}


def pooled(rows):
    tp = sum(x["tp"] for x in rows); nt = sum(x["n_tgt"] for x in rows)
    fp = sum(x["fp"] for x in rows); nn = sum(x["n_nor"] for x in rows)
    return {"se": 100.0 * tp / nt if nt else np.nan, "far": 100.0 * fp / nn if nn else np.nan,
            "ppv": 100.0 * tp / (tp + fp) if tp + fp else np.nan}


def bootstrap(groups, metric_rows, n_boot=2000, seed=0, paired_with=None):
    """Hierarchical bootstrap: resample patients (groups) with replacement and keep all their
    records (days). metric_rows: dict method -> list of rows aligned with `groups` (patient id per
    row). Returns dict method -> {metric: (point, lo, hi)} and, if paired_with is a method name,
    the paired differences (method - paired_with)."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    out = {}
    samples = {m: [] for m in metric_rows}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        for m, rows in metric_rows.items():
            samples[m].append(pooled([rows[i] for i in idx]))
    for m, rows in metric_rows.items():
        pt = pooled(rows)
        res = {}
        for k in ("se", "far", "ppv"):
            v = np.array([s[k] for s in samples[m]], float)
            res[k] = (pt[k], float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
        if paired_with and m != paired_with:
            for k in ("se", "far"):
                d = np.array([a[k] - b[k] for a, b in zip(samples[m], samples[paired_with])], float)
                res["d_" + k] = (pt[k] - pooled(metric_rows[paired_with])[k],
                                 float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5)))
        out[m] = res
    return out
