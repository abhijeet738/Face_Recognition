"""Index-of-Max (IoM) hashing with Gaussian random projection (PLAN section 9.1).

GRP-based IoM hashing (Jin et al., IEEE TIFS 2018). For each of m secret random Gaussian
matrices W_i (q x d), project the embedding x and keep only argmax(W_i @ x), an integer in
[0, q). The magnitudes are thrown away, so the transform is many-to-one: the stored code
cannot be inverted back to the embedding in the information-theoretic sense.
"""

from functools import lru_cache

import numpy as np

from .keys import seed_to_int

SCHEME_NAME = "iom-grp-v1"
MATRIX_CACHE_SIZE = 8  # 8 matrices of about 8.4 MB each (SPEC_AMENDMENTS C8)


class IoMHasher:
    """Hash L2-normalised embeddings into m small integers (uint8 codes in [0, q))."""

    def __init__(self, m: int = 256, q: int = 16, d: int = 512) -> None:
        if not 2 <= q <= 256:
            raise ValueError("q must be in [2, 256] so codes fit in uint8")
        if m < 1 or d < 1:
            raise ValueError("m and d must be positive")
        self.m, self.q, self.d = m, q, d

    def matrices(self, seed: bytes) -> np.ndarray:
        """(m, q, d) float32 Gaussian matrices from a seed. About 8.4 MB for m=256, q=16, d=512."""
        rng = np.random.default_rng(seed_to_int(seed))
        return rng.standard_normal((self.m, self.q, self.d), dtype=np.float32)

    def _check(self, W: np.ndarray, x: np.ndarray) -> None:
        if W.shape != (self.m, self.q, self.d):
            raise ValueError(f"matrices must have shape {(self.m, self.q, self.d)}")
        if x.shape[-1] != self.d:
            raise ValueError(f"embedding has {x.shape[-1]} values, expected {self.d}")

    def hash(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """x: (d,) -> code (m,) uint8."""
        x = np.asarray(x, dtype=np.float32)
        self._check(W, x)
        proj = (W.reshape(self.m * self.q, self.d) @ x).reshape(self.m, self.q)
        return np.argmax(proj, axis=1).astype(np.uint8)

    def hash_batch(self, X: np.ndarray, W: np.ndarray, chunk: int = 512) -> np.ndarray:
        """X: (n, d) -> codes (n, m) uint8. Processed in chunks to bound memory."""
        X = np.asarray(X, dtype=np.float32)
        self._check(W, X)
        W2 = W.reshape(self.m * self.q, self.d)
        out = np.empty((X.shape[0], self.m), dtype=np.uint8)
        for i in range(0, X.shape[0], chunk):
            part = X[i : i + chunk]
            proj = (W2 @ part.T).reshape(self.m, self.q, part.shape[0])
            out[i : i + chunk] = proj.argmax(axis=1).T
        return out

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fraction of positions where the codes agree, in [0, 1]. Works on (m,) or (n, m)."""
        return np.mean(a == b, axis=-1)


@lru_cache(maxsize=MATRIX_CACHE_SIZE)
def cached_matrices(seed: bytes, m: int, q: int, d: int) -> np.ndarray:
    """Used by the vault: regenerating 8 MB of Gaussians costs about 20-40 ms.

    The cached array is read-only so one request cannot corrupt the matrices of the next.
    """
    W = IoMHasher(m, q, d).matrices(seed)
    W.setflags(write=False)
    return W
