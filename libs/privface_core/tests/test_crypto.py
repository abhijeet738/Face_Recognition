import pytest
from cryptography.exceptions import InvalidTag
from privface_core.crypto import decrypt_template, encrypt_template

KEY = bytes(range(32))
CODE = bytes(range(256))  # a pretend 256-position IoM code


def enc(**kw: object) -> bytes:
    args = {"key": KEY, "code": CODE, "user_id": "u-1", "kind": "verify", "key_version": 1}
    args.update(kw)
    return encrypt_template(**args)  # type: ignore[arg-type]


def dec(blob: bytes, **kw: object) -> bytes:
    args = {"key": KEY, "blob": blob, "user_id": "u-1", "kind": "verify", "key_version": 1}
    args.update(kw)
    return decrypt_template(**args)  # type: ignore[arg-type]


def test_round_trip() -> None:
    assert dec(enc()) == CODE


def test_blob_is_nonce_plus_ciphertext_plus_tag() -> None:
    assert len(enc()) == 12 + len(CODE) + 16


def test_every_encryption_uses_a_fresh_nonce() -> None:
    a, b = enc(), enc()
    assert a[:12] != b[:12]
    assert a != b
    assert dec(a) == dec(b) == CODE


def test_wrong_key_fails() -> None:
    with pytest.raises(InvalidTag):
        dec(enc(), key=bytes(reversed(KEY)))


@pytest.mark.parametrize("position", [0, 11, 12, 100, -17, -16, -1])  # nonce, body, tag
def test_a_changed_byte_anywhere_fails(position: int) -> None:
    blob = bytearray(enc())
    blob[position] ^= 0x01
    with pytest.raises(InvalidTag):
        dec(bytes(blob))


@pytest.mark.parametrize(
    "changed",
    [{"user_id": "u-2"}, {"kind": "identify"}, {"key_version": 2}],
)
def test_blob_is_bound_to_its_row(changed: dict) -> None:
    with pytest.raises(InvalidTag):
        dec(enc(), **changed)


@pytest.mark.parametrize("blob", [b"", b"x" * 5, b"x" * 27])
def test_short_blobs_fail_with_the_same_error(blob: bytes) -> None:
    with pytest.raises(InvalidTag):
        dec(blob)


def test_truncated_blob_fails() -> None:
    with pytest.raises(InvalidTag):
        dec(enc()[:-1])


@pytest.mark.parametrize("key", [b"", b"x" * 16, b"x" * 31, b"x" * 33, "k" * 32])
def test_key_must_be_exactly_32_bytes(key: object) -> None:
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        enc(key=key)
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        dec(b"x" * 40, key=key)


def test_row_fields_are_validated() -> None:
    with pytest.raises(ValueError, match="must not contain"):
        enc(user_id="a|b")
    with pytest.raises(ValueError, match="user_id"):
        enc(user_id="")
    with pytest.raises(ValueError, match="key_version"):
        enc(key_version=-1)
    with pytest.raises(ValueError, match="key_version"):
        enc(key_version=True)
