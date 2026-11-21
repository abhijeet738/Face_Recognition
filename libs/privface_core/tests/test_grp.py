import hashlib

import numpy as np
import pytest
from privface_core.grp import grp_matrix, grp_similarity, grp_transform
from privface_core.keys import derive_seed

MASTER = bytes(range(32))
SEED = derive_seed(MASTER, "verify", "alice", 1)


def unit(rng: np.random.Generator, n: int = 0, d: int = 512) -> np.ndarray:
    v = rng.standard_normal((n, d) if n else d).astype(np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_matrix_is_deterministic_and_has_the_right_shape() -> None:
    R = grp_matrix(SEED, 128)
    assert R.shape == (128, 512)
    assert R.dtype == np.float32
    assert np.array_equal(R, grp_matrix(SEED, 128))


def test_different_seeds_give_different_matrices() -> None:
    other = derive_seed(MASTER, "verify", "alice", 2)
    assert not np.array_equal(grp_matrix(SEED, 64), grp_matrix(other, 64))


def test_golden_matrix_hash_guards_against_numpy_changes() -> None:
    # If this fails after upgrading NumPy, every stored template would silently stop matching.
    golden = derive_seed(MASTER, "verify", "golden", 1)
    digest = hashlib.sha256(grp_matrix(golden, 64).tobytes()).hexdigest()
    assert digest == "e3125904423fdab3f93b428580d0dd1d2cec4e837d07e9715afad2d43ce8f85f"


def test_transform_output_is_unit_length_for_one_vector_and_a_batch() -> None:
    rng = np.random.default_rng(0)
    R = grp_matrix(SEED, 256)
    y = grp_transform(unit(rng), R)
    assert y.shape == (256,)
    assert np.linalg.norm(y) == pytest.approx(1.0, abs=1e-5)
    Y = grp_transform(unit(rng, n=10), R)
    assert Y.shape == (10, 256)
    assert np.allclose(np.linalg.norm(Y, axis=1), 1.0, atol=1e-5)


def test_batch_transform_matches_one_at_a_time() -> None:
    rng = np.random.default_rng(1)
    R = grp_matrix(SEED, 128)
    X = unit(rng, n=5)
    assert np.allclose(grp_transform(X, R), np.stack([grp_transform(x, R) for x in X]), atol=1e-6)


def test_same_embedding_scores_one_and_similar_ones_score_high() -> None:
    rng = np.random.default_rng(2)
    R = grp_matrix(SEED, 512)
    x = unit(rng)
    assert grp_similarity(grp_transform(x, R), grp_transform(x, R)) == pytest.approx(1.0, abs=1e-5)
    near = x + 0.3 * unit(rng)  # a perturbed copy of the same "face"
    near /= np.linalg.norm(near)
    assert float(np.dot(x, near)) > 0.9
    assert grp_similarity(grp_transform(x, R), grp_transform(near, R)) > 0.85


def test_unrelated_vectors_stay_near_zero() -> None:
    rng = np.random.default_rng(3)
    R = grp_matrix(SEED, 256)
    a, b = grp_transform(unit(rng, n=500), R), grp_transform(unit(rng, n=500), R)
    assert abs(float(grp_similarity(a, b).mean())) < 0.02


def test_projection_preserves_cosine_roughly() -> None:
    rng = np.random.default_rng(4)
    R = grp_matrix(SEED, 512)
    a, b = unit(rng, n=200), unit(rng, n=200)
    b = 0.7 * a + 0.3 * b  # make the pairs genuinely similar
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    raw = np.sum(a * b, axis=1)
    projected = grp_similarity(grp_transform(a, R), grp_transform(b, R))
    assert np.abs(raw - projected).mean() < 0.05


def test_bad_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        grp_matrix(SEED, 0)
    R = grp_matrix(SEED, 64)
    with pytest.raises(ValueError, match="expects 512"):
        grp_transform(np.ones(100, dtype=np.float32), R)
    with pytest.raises(ValueError, match="zero vector"):
        grp_transform(np.zeros(512, dtype=np.float32), R)