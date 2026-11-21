"""Embedding <-> text conversion for sending embeddings between services (PLAN section 9.1).

An embedding travels as base64 of little-endian float32 bytes. The decoder is strict, because
it is the first code that touches data coming from another service.
"""

import base64
import binascii

import numpy as np

EMBEDDING_DIM = 512
_MIN_NORM, _MAX_NORM = 0.9, 1.1


def f32_to_b64(x: np.ndarray) -> str:
    """Encode a float array as base64 of little-endian float32 bytes."""
    return base64.b64encode(np.asarray(x, dtype="<f4").tobytes()).decode("ascii")


def b64_to_f32(s: str, d: int = EMBEDDING_DIM) -> np.ndarray:
    """Decode and validate an L2-normalised embedding of exactly d float32 values."""
    if not isinstance(s, str):
        raise ValueError("bad embedding")
    try:
        raw = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("bad embedding") from exc
    if len(raw) != 4 * d:
        raise ValueError("bad embedding")
    x = np.frombuffer(raw, dtype="<f4")
    if not np.all(np.isfinite(x)):
        raise ValueError("bad embedding")
    if not _MIN_NORM < float(np.linalg.norm(x)) < _MAX_NORM:
        raise ValueError("embedding must be L2-normalised")
    return x.astype(np.float32)