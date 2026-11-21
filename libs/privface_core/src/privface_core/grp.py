"""Gaussian random projection (GRP), the baseline protection (PLAN section 9.1).

A secret random matrix R (k x d) turns a 512-d embedding into a k-d template. By the
Johnson-Lindenstrauss lemma angles are roughly preserved, so matching still works when k is
large enough. The weakness is linearity: whoever has R and the template can compute
pinv(R) @ y (experiment E7), which is why IoM hashing (iom.py) is the main scheme.
"""

import numpy as np

from .keys import seed_to_int


def grp_matrix(seed: bytes, k: int, d: int = 512) -> np.ndarray:
    """Deterministic (k, d) float32 matrix, scaled by 1/sqrt(k), from a seed."""
    if k < 1 or d < 1:
        raise ValueError("k and d must be positive")
    rng = np.random.default_rng(seed_to_int(seed))
    return (rng.standard_normal((k, d), dtype=np.float32) / np.sqrt(k)).astype(np.float32)


def grp_transform(x: np.ndarray, R: np.ndarray) -> np.ndarray:
    """x: (d,) or (n, d) L2-normalised embeddings -> (k,) or (n, k), L2-normalised."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] != R.shape[1]:
        raise ValueError(f"embedding has {x.shape[-1]} values, the matrix expects {R.shape[1]}")
    y = x @ R.T
    norm = np.linalg.norm(y, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("projection is the zero vector")
    return (y / norm).astype(np.float32)


def grp_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of two already-normalised templates (works on (k,) or (n, k))."""
    return np.sum(a * b, axis=-1)