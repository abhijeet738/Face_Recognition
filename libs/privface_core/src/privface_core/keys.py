"""Key derivation: deterministic seeds from a master secret (PLAN section 9.1).

Keys are never stored. A seed is derived on demand with HMAC-SHA256 from the master secret and
a purpose, a subject and a version number. Revoking a template only means bumping the version.
"""

import hashlib
import hmac

SCHEME = "privface/v1"
_SEP = "|"
MIN_MASTER_BYTES = 32


def derive_seed(master: bytes, purpose: str, subject: str, version: int) -> bytes:
    """Deterministic 32-byte seed. Same inputs -> same seed; any change -> unrelated seed.

    purpose: "verify" (per-user key), "identify" (app key, subject="__app__"),
             "lab-a" / "lab-b" (Breach Lab namespaces).
    """
    if not isinstance(master, bytes | bytearray) or len(master) < MIN_MASTER_BYTES:
        raise ValueError(f"master secret must be at least {MIN_MASTER_BYTES} bytes")
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise ValueError("version must be a non-negative integer")
    for name, value in (("purpose", purpose), ("subject", subject)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
        if _SEP in value:
            # Without this, ("a", "b|c") and ("a|b", "c") would build the same message.
            raise ValueError(f"{name} must not contain '{_SEP}'")
    msg = f"{SCHEME}{_SEP}{purpose}{_SEP}{subject}{_SEP}{version}".encode()
    return hmac.new(bytes(master), msg, hashlib.sha256).digest()


def seed_to_int(seed: bytes) -> int:
    """NumPy's Generator accepts a big integer as entropy."""
    return int.from_bytes(seed, "big")