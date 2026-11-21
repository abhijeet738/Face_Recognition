import pytest
from privface_core.keys import derive_seed, seed_to_int

MASTER = bytes(range(32))


def test_same_inputs_give_the_same_32_byte_seed() -> None:
    a = derive_seed(MASTER, "verify", "alice", 1)
    assert a == derive_seed(MASTER, "verify", "alice", 1)
    assert len(a) == 32


@pytest.mark.parametrize(
    "other",
    [
        (bytes(reversed(MASTER)), "verify", "alice", 1),  # different master
        (MASTER, "identify", "alice", 1),  # different purpose
        (MASTER, "verify", "bob", 1),  # different subject
        (MASTER, "verify", "alice", 2),  # different version
    ],
)
def test_changing_any_input_changes_the_seed(other: tuple) -> None:
    assert derive_seed(*other) != derive_seed(MASTER, "verify", "alice", 1)


def test_short_master_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 32 bytes"):
        derive_seed(b"x" * 31, "verify", "alice", 1)
    with pytest.raises(ValueError, match="at least 32 bytes"):
        derive_seed("not bytes", "verify", "alice", 1)  # type: ignore[arg-type]


def test_separator_cannot_be_used_to_forge_a_collision() -> None:
    with pytest.raises(ValueError, match="must not contain"):
        derive_seed(MASTER, "verify", "a|1|b", 1)
    with pytest.raises(ValueError, match="must not contain"):
        derive_seed(MASTER, "ver|ify", "alice", 1)


@pytest.mark.parametrize("version", [-1, 1.5, True, "1"])
def test_bad_versions_are_rejected(version: object) -> None:
    with pytest.raises(ValueError, match="version"):
        derive_seed(MASTER, "verify", "alice", version)  # type: ignore[arg-type]


def test_empty_purpose_or_subject_is_rejected() -> None:
    with pytest.raises(ValueError, match="purpose"):
        derive_seed(MASTER, "", "alice", 1)
    with pytest.raises(ValueError, match="subject"):
        derive_seed(MASTER, "verify", "", 1)


def test_seed_to_int_is_big_endian() -> None:
    assert seed_to_int(b"\x00\x01") == 1
    assert seed_to_int(derive_seed(MASTER, "verify", "alice", 1)) > 2**200