import base64

import numpy as np
import pytest
from privface_core.codec import b64_to_f32, f32_to_b64


def unit(seed: int = 0, d: int = 512) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(d).astype(np.float32)
    return v / np.linalg.norm(v)


def test_round_trip_is_exact() -> None:
    x = unit()
    y = b64_to_f32(f32_to_b64(x))
    assert y.dtype == np.float32
    assert np.array_equal(x, y)


def test_encoding_is_little_endian_float32() -> None:
    assert base64.b64decode(f32_to_b64(np.array([1.0]))) == b"\x00\x00\x80\x3f"


def test_wrong_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="bad embedding"):
        b64_to_f32(f32_to_b64(unit(d=256)))  # too short
    with pytest.raises(ValueError, match="bad embedding"):
        b64_to_f32(f32_to_b64(np.r_[unit(), np.float32(0)]))  # too long
    assert b64_to_f32(f32_to_b64(unit(d=256)), d=256).shape == (256,)


def test_not_normalised_is_rejected() -> None:
    with pytest.raises(ValueError, match="L2-normalised"):
        b64_to_f32(f32_to_b64(unit() * 3))
    with pytest.raises(ValueError, match="L2-normalised"):
        b64_to_f32(f32_to_b64(np.zeros(512)))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_values_are_rejected(bad: float) -> None:
    x = unit()
    x[7] = bad
    with pytest.raises(ValueError, match="bad embedding"):
        b64_to_f32(f32_to_b64(x))


@pytest.mark.parametrize("bad", ["not base64!!", "abc", "", None, 123, b"bytes"])
def test_garbage_input_is_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match="bad embedding"):
        b64_to_f32(bad)  # type: ignore[arg-type]
