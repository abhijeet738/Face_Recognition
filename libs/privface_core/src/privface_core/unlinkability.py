"""Unlinkability score D_sys (Gomez-Barrero et al., IEEE TIFS 2018; PLAN section 9.1).

Mated pairs: two templates of the SAME person made with DIFFERENT keys.
Non-mated pairs: templates of different people, made with different keys.
If an attacker cannot tell the two score distributions apart, the templates cannot be linked.
D_sys = 0 means fully unlinkable, 1 means fully linkable.
"""

import numpy as np


def d_sys(mated, nonmated, bins: int = 100, omega: float = 1.0) -> float:
    """Global unlinkability measure in [0, 1] (lower is better for privacy).

    For each score bin the likelihood ratio LR = p(s | mated) / p(s | non-mated) * omega gives a
    local measure 2 * LR / (1 + LR) - 1 when LR > 1 (else 0). D_sys is its average weighted by
    how often mated scores fall in that bin. omega is the prior odds of mated vs non-mated.
    """
    m = np.asarray(mated, dtype=np.float64).ravel()
    n = np.asarray(nonmated, dtype=np.float64).ravel()
    if len(m) == 0 or len(n) == 0:
        raise ValueError("need at least one mated and one non-mated score")
    if not (np.all(np.isfinite(m)) and np.all(np.isfinite(n))):
        raise ValueError("scores must be finite")
    if bins < 1 or omega <= 0:
        raise ValueError("bins must be at least 1 and omega must be positive")
    lo, hi = float(min(m.min(), n.min())), float(max(m.max(), n.max()))
    if hi <= lo:
        return 0.0  # every score is identical, so the scores carry no information
    hm, edges = np.histogram(m, bins=bins, range=(lo, hi), density=True)
    hn, _ = np.histogram(n, bins=bins, range=(lo, hi), density=True)
    d_local = np.zeros_like(hm)
    both = (hm > 0) & (hn > 0)
    lr = np.zeros_like(hm)
    lr[both] = hm[both] / hn[both] * omega
    d_local[both] = np.where(lr[both] > 1, 2 * lr[both] / (1 + lr[both]) - 1, 0.0)
    d_local[(hm > 0) & (hn == 0)] = 1.0  # only mated scores reach here: fully linkable
    width = edges[1] - edges[0]
    return float(np.sum(hm * d_local) * width)
