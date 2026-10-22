"""Population classifiers and per-patient score fusion.

Three random forests are trained on labelled beats of training patients (normal vs S/V):
  ECG-RF (ECG features), PPG-RF (PPG features), Joint-RF (early fusion, both feature sets).
Per patient, scores are combined in four ways, all fixed by the patient's fitting beats so that
the fused score is a fixed function of each beat (conformal validity is preserved):
  late   : -log min(T_ecg, T_ppg), T = empirical upper-tail probability among fitting beats;
  gated  : late fusion when the PPG quality index is >= SQI_MIN, ECG tail alone otherwise;
and two p-value combinations computed in the conformal layer (see conformal.py):
  p-bonf : p = min(1, 2 min(p_ecg, p_ppg))           (valid under arbitrary dependence)
  p-mean : p = min(1, p_ecg + p_ppg)  (= 2 x mean)   (valid under arbitrary dependence)
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier

SQI_MIN = 0.5
FUSIONS = ["ecg", "ppg", "early", "late", "gated", "pbonf", "pmean"]
P_FUSIONS = {"pbonf", "pmean"}


def _rf(seed):
    return RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=-1, random_state=seed)


def stack_training(records, max_per_record=4000, seed=42):
    rng = np.random.default_rng(seed)
    Xe, Xp, y = [], [], []
    for r in records:
        ok = (r["label"] >= 0) & (r["label"] <= 2)  # AF-conducted beats (3) not used for training
        idx = np.where(ok)[0]
        if len(idx) > max_per_record:
            ab = idx[r["label"][idx] > 0]
            no = idx[r["label"][idx] == 0]
            keep_no = max(0, max_per_record - len(ab))
            no = rng.choice(no, size=min(len(no), keep_no), replace=False)
            idx = np.sort(np.r_[ab, no])
        Xe.append(r["Xe"][idx])
        Xp.append(r["Xp"][idx])
        y.append((r["label"][idx] > 0).astype(int))
    return np.vstack(Xe), np.vstack(Xp), np.concatenate(y)


class PopulationModels:
    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, records):
        Xe, Xp, y = stack_training(records, seed=self.seed)
        self.ecg = _rf(self.seed).fit(Xe, y)
        self.ppg = _rf(self.seed).fit(Xp, y)
        self.joint = _rf(self.seed).fit(np.hstack([Xe, Xp]), y)
        return self

    @staticmethod
    def _logit(model, X):
        p = model.predict_proba(X)[:, 1]
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    def raw_scores(self, r):
        """Scores are cached per record id (records may be re-loaded from disk several times)."""
        if not hasattr(self, "_cache"):
            self._cache = {}
        key = str(r["id"])
        if key not in self._cache:
            self._cache[key] = {k: v.astype(np.float32) for k, v in self._raw_scores(r).items()}
        return {k: v.astype(float) for k, v in self._cache[key].items()}

    def _raw_scores(self, r):
        return {
            "ecg": self._logit(self.ecg, r["Xe"]),
            "ppg": self._logit(self.ppg, r["Xp"]),
            "early": self._logit(self.joint, np.hstack([r["Xe"], r["Xp"]])),
        }


def upper_tail(s, ref):
    """Empirical upper-tail probability of s among ref, extended exponentially beyond max(ref)."""
    ref = np.sort(np.asarray(ref, float))
    n = len(ref)
    cnt = n - np.searchsorted(ref, s, side="left")
    T = (1.0 + cnt) / (n + 1.0)
    mx = ref[-1]
    mad = 1.4826 * np.median(np.abs(ref - np.median(ref))) + 1e-6
    beyond = s > mx
    T[beyond] = (1.0 / (n + 1.0)) * np.exp(-(s[beyond] - mx) / mad)
    return np.maximum(T, 1e-300)


def fused_scores(raw, r):
    """Return dict fusion -> score array (or tuple of two arrays for p-value fusions)."""
    fit = r["fit_mask"]
    te = upper_tail(raw["ecg"], raw["ecg"][fit])
    tp = upper_tail(raw["ppg"], raw["ppg"][fit])
    late = -np.log(np.minimum(te, tp))
    gated = np.where(r["sqi"] >= SQI_MIN, late, -np.log(te))
    out = {"ecg": raw["ecg"], "ppg": raw["ppg"], "early": raw["early"],
           "late": late, "gated": gated,
           "pbonf": (raw["ecg"], raw["ppg"]), "pmean": (raw["ecg"], raw["ppg"])}
    return out
