"""Biometric performance metrics (PLAN section 9.1): EER, TAR@FAR, ROC, 10-fold accuracy.

"Genuine" scores come from pairs of the same person, "impostor" scores from different people.
Higher score = more similar everywhere in this module.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

_THRESHOLD_CANDIDATES = 1000


def _labels_scores(genuine, impostor) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(genuine, dtype=np.float64).ravel()
    i = np.asarray(impostor, dtype=np.float64).ravel()
    if len(g) == 0 or len(i) == 0:
        raise ValueError("need at least one genuine and one impostor score")
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(i))):
        raise ValueError("scores must be finite")
    return np.r_[np.ones(len(g)), np.zeros(len(i))], np.r_[g, i]


def eer(genuine, impostor) -> tuple[float, float]:
    """Equal Error Rate and the threshold where FAR == FRR."""
    y, s = _labels_scores(genuine, impostor)
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def tar_at_far(genuine, impostor, far: float) -> tuple[float, float]:
    """True Accept Rate at a target False Accept Rate, and the threshold used.

    A genuine pair is accepted when its score is strictly above the threshold. The threshold
    is the (1 - far) quantile of the impostor scores, so the achieved FAR is at most `far`.
    Needs at least 1/far impostor scores to resolve that FAR; about 10/far to be reliable.
    """
    if not 0 < far < 1:
        raise ValueError("far must be between 0 and 1")
    imp = np.asarray(impostor, dtype=np.float64).ravel()
    gen = np.asarray(genuine, dtype=np.float64).ravel()
    if len(gen) == 0 or len(imp) < 1 / far:
        raise ValueError(f"FAR={far} needs at least {int(np.ceil(1 / far))} impostor scores")
    t = float(np.quantile(imp, 1 - far, method="higher"))
    return float(np.mean(gen > t)), t


def roc_points(genuine, impostor, n: int = 200) -> dict:
    """ROC curve thinned to at most n points, plus AUC, ready to be stored as JSON."""
    y, s = _labels_scores(genuine, impostor)
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.unique(np.linspace(0, len(fpr) - 1, n).astype(int))
    return {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist(), "auc": float(roc_auc_score(y, s))}


def tenfold_accuracy(
    scores: np.ndarray, issame: np.ndarray, folds: np.ndarray
) -> tuple[float, float]:
    """Standard LFW protocol: pick the best threshold on 9 folds, test on the 10th.

    Returns (mean accuracy, standard deviation). Candidate thresholds are 1,000 evenly spaced
    quantiles of the training scores, so it stays fast on 6,000 pairs.
    """
    scores = np.asarray(scores, dtype=np.float64)
    issame = np.asarray(issame, dtype=bool)
    folds = np.asarray(folds)
    if not (len(scores) == len(issame) == len(folds)):
        raise ValueError("scores, issame and folds must have the same length")
    accs = []
    for f in np.unique(folds):
        train, test = folds != f, folds == f
        s_tr = scores[train]
        cand = np.unique(
            np.r_[s_tr.min() - 1e-9, np.quantile(s_tr, np.linspace(0, 1, _THRESHOLD_CANDIDATES))]
        )
        correct = ((s_tr[None, :] > cand[:, None]) == issame[train][None, :]).mean(axis=1)
        best = cand[int(np.argmax(correct))]
        accs.append(np.mean((scores[test] > best) == issame[test]))
    return float(np.mean(accs)), float(np.std(accs))
