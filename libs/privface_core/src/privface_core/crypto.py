"""AES-256-GCM encryption for templates at rest (PLAN section 9.1).

A stored blob is nonce (12 bytes) + ciphertext + tag (16 bytes). The user id, the kind of
template and the key version are bound in as associated data, so a blob copied to another
user's row, or presented under another key version, fails to decrypt.
"""

import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_BYTES = 32
NONCE_BYTES = 12
TAG_BYTES = 16
_SEP = "|"


def _aad(user_id: str, kind: str, key_version: int) -> bytes:
    """Bind the ciphertext to its row. Fields are checked so they cannot be shifted around."""
    for name, value in (("user_id", user_id), ("kind", kind)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
        if _SEP in value:
            raise ValueError(f"{name} must not contain '{_SEP}'")
    if not isinstance(key_version, int) or isinstance(key_version, bool) or key_version < 0:
        raise ValueError("key_version must be a non-negative integer")
    return f"{user_id}{_SEP}{kind}{_SEP}{key_version}".encode()


def _check_key(key: bytes) -> None:
    if not isinstance(key, bytes | bytearray) or len(key) != KEY_BYTES:
        raise ValueError(f"key must be exactly {KEY_BYTES} bytes")


def encrypt_template(key: bytes, code: bytes, user_id: str, kind: str, key_version: int) -> bytes:
    """Return nonce + ciphertext + tag. A fresh random nonce is used for every call."""
    _check_key(key)
    aad = _aad(user_id, kind, key_version)
    nonce = os.urandom(NONCE_BYTES)  # MUST be unique per encryption
    return nonce + AESGCM(bytes(key)).encrypt(nonce, bytes(code), aad)


def decrypt_template(key: bytes, blob: bytes, user_id: str, kind: str, key_version: int) -> bytes:
    """Return the plaintext code. Raises InvalidTag for a wrong key, row or tampered blob."""
    _check_key(key)
    aad = _aad(user_id, kind, key_version)
    if len(blob) < NONCE_BYTES + TAG_BYTES:
        raise InvalidTag
    nonce, ct = bytes(blob[:NONCE_BYTES]), bytes(blob[NONCE_BYTES:])
    return AESGCM(bytes(key)).decrypt(nonce, ct, aad)
