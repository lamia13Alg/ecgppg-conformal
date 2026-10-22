"""Beat-, record- and episode-level metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score


def eval_hours(r):
    t = r["t"][r["eval_mask"]]
    b = r["block"][r["eval_mask"]]
    h = 0.0
    for blk in np.unique(b):
        tb = t[b == blk]
        if len(tb) > 1:
            h += tb[-1] - tb[0]
    return h / 3600.0


def episodes(abn, min_len=3, max_gap=10):
    """Clusters of >= min_len abnormal beats with gaps <= max_gap beats. abn: bool array on the
    evaluated beat sequence. Returns list of (first, last) positions."""
    idx = np.where(abn)[0]
    eps = []
    if len(idx) == 0:
        return eps
    start = prev = idx[0]
    count = 1
    for j in idx[1:]:
        if j - prev - 1 <= max_gap:
            count += 1
        else:
            if count >= min_len:
                eps.append((start, prev))
            start, count = j, 1
        prev = j
    if count >= min_len:
        eps.append((start, prev))
    return eps


def record_metrics(r, res, auc_score, ep_alarm=None, ep_min_len=3):
    ev = r["eval_mask"] & (r["label"] >= 0) & (r["label"] <= 2) & np.isfinite(res["p"])  # 3 = AF-conducted, excluded
    lab = r["label"]
    a = res["alarm"]
    sinus = r.get("sinus", np.ones(len(lab), bool))
    bad = r.get("ecg_bad", np.zeros(len(lab), bool))
    out = {"record": r["id"], "hours": eval_hours(r)}
    def cnt(mask):
        m = ev & mask
        return int(m.sum()), int((m & a).sum())
    out["n_abn"], out["tp"] = cnt(lab > 0)
    out["n_v"], out["tp_v"] = cnt(lab == 2)
    out["n_s"], out["tp_s"] = cnt(lab == 1)
    out["n_norm"], out["fp"] = cnt(lab == 0)
    out["n_norm_sinus"], out["fp_sinus"] = cnt((lab == 0) & sinus)
    out["n_abn_bad"], out["tp_bad"] = cnt((lab > 0) & bad)
    out["n_norm_bad"], out["fp_bad"] = cnt((lab == 0) & bad & sinus)
    out["n_abn_good"], out["tp_good"] = cnt((lab > 0) & ~bad)
    out["n_norm_good"], out["fp_good"] = cnt((lab == 0) & ~bad & sinus)
    out["far_sinus"] = out["fp_sinus"] / out["n_norm_sinus"] if out["n_norm_sinus"] else np.nan
    yy = (lab[ev] > 0).astype(int)
    out["auroc"] = roc_auc_score(yy, auc_score[ev]) if 0 < yy.sum() < len(yy) else np.nan
    b = res["bound"][ev & (lab == 0)]
    out["bound_sum"] = float(np.nansum(b))
    out["bound_n"] = int(np.isfinite(b).sum())
    out["reviews"] = res["reviews"]
    out["reviews_grouped"] = res["reviews_grouped"]
    out["startup_reviews"] = res["startup_reviews"]
    if ep_alarm is not None:
        pos = np.where(r["eval_mask"])[0]
        abn_seq = (lab[pos] == 1) | (lab[pos] == 2)
        al_seq = ep_alarm[pos]
        eps = episodes(abn_seq, min_len=ep_min_len)
        det, delays = 0, []
        for (f, l) in eps:
            w = np.where(al_seq[f:min(len(al_seq), l + 6)])[0]
            if len(w):
                det += 1
                delays.append((w[0], r["t"][pos[f + w[0]]] - r["t"][pos[f]]))
        in_ep = np.zeros(len(pos), bool)
        for (f, l) in eps:
            in_ep[f:min(len(pos), l + 6)] = True
        out["n_ep"], out["ep_det"] = len(eps), det
        out["ep_delay_beats"] = [d[0] for d in delays]
        out["ep_delay_s"] = [d[1] for d in delays]
        out["ep_false"] = int((al_seq & ~in_ep).sum())
    return out


def pool(rows, alpha):
    def s(k):
        return sum(r[k] for r in rows)
    def ratio(a, b):
        return 100.0 * s(a) / s(b) if s(b) else float("nan")
    far = np.array([r["far_sinus"] for r in rows if np.isfinite(r["far_sinus"])]) * 100
    hours = s("hours")
    out = {
        "auroc": float(np.nanmean([r["auroc"] for r in rows])),
        "se": ratio("tp", "n_abn"), "se_v": ratio("tp_v", "n_v"), "se_s": ratio("tp_s", "n_s"),
        "ppv": 100.0 * s("tp") / (s("tp") + s("fp")) if (s("tp") + s("fp")) else float("nan"),
        "far": ratio("fp", "n_norm"), "far_sinus": ratio("fp_sinus", "n_norm_sinus"),
        "se_badecg": ratio("tp_bad", "n_abn_bad"), "far_badecg": ratio("fp_bad", "n_norm_bad"),
        "se_goodecg": ratio("tp_good", "n_abn_good"), "far_goodecg": ratio("fp_good", "n_norm_good"),
        "bound": 100.0 * s("bound_sum") / s("bound_n") if s("bound_n") else float("nan"),
        "rev_h": (s("reviews")) / hours if hours else float("nan"),
        "rev_grouped_h": s("reviews_grouped") / hours if hours else float("nan"),
        "startup_per_patient": s("startup_reviews") / len(rows),
        "far_median": float(np.median(far)) if len(far) else float("nan"),
        "far_p90": float(np.percentile(far, 90)) if len(far) else float("nan"),
        "far_max": float(np.max(far)) if len(far) else float("nan"),
        "pct_above_2a": float(100 * np.mean(far > 200 * alpha)) if len(far) else float("nan"),
        "pct_below_a2": float(100 * np.mean(far < 50 * alpha)) if len(far) else float("nan"),
        "n_records": len(rows), "hours": hours,
    }
    if "n_ep" in rows[0]:
        nep = s("n_ep")
        out["ep_se"] = 100.0 * s("ep_det") / nep if nep else float("nan")
        db = [d for r in rows for d in r["ep_delay_beats"]]
        ds = [d for r in rows for d in r["ep_delay_s"]]
        out["ep_delay_beats"] = float(np.median(db)) if db else float("nan")
        out["ep_delay_s"] = float(np.median(ds)) if ds else float("nan")
        out["ep_false_h"] = s("ep_false") / hours if hours else float("nan")
        out["n_ep"] = nep
    return out
