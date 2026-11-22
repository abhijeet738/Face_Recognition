import hashlib

import numpy as np
import pytest
from privface_core.iom import IoMHasher, cached_matrices
from privface_core.keys import derive_seed

MASTER = bytes(range(32))
SEED = derive_seed(MASTER, "verify", "alice", 1)
SEED_V2 = derive_seed(MASTER, "verify", "alice", 2)


def unit(rng: np.random.Generator, n: int, d: int = 512) -> np.ndarray:
    v = rng.standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_matrices_are_deterministic_and_have_the_right_shape() -> None:
    h = IoMHasher(m=16, q=8)
    W = h.matrices(SEED)
    assert W.shape == (16, 8, 512)
    assert W.dtype == np.float32
    assert np.array_equal(W, h.matrices(SEED))
    assert not np.array_equal(W, h.matrices(SEED_V2))


def test_golden_matrix_hash_guards_against_numpy_changes() -> None:
    # If this fails after upgrading NumPy, every stored template would silently stop matching.
    golden = derive_seed(MASTER, "verify", "golden", 1)
    digest = hashlib.sha256(IoMHasher(32, 16).matrices(golden).tobytes()).hexdigest()
    assert digest == "bf50d257b2721bf5382630cd9f76fd111cd6ab4d22510ea46d8b6abf7412e2e9"


def test_code_is_uint8_with_values_below_q() -> None:
    h = IoMHasher(m=64, q=16)
    code = h.hash(unit(np.random.default_rng(0), 1)[0], h.matrices(SEED))
    assert code.shape == (64,)
    assert code.dtype == np.uint8
    assert code.max() < 16


def test_batch_hash_matches_hashing_one_at_a_time() -> None:
    h = IoMHasher(m=64, q=16)
    W = h.matrices(SEED)
    X = unit(np.random.default_rng(1), 40)
    batch = h.hash_batch(X, W, chunk=16)  # several chunks, the last one partial
    assert batch.shape == (40, 64)
    assert np.array_equal(batch, np.stack([h.hash(x, W) for x in X]))


def test_same_code_has_similarity_one() -> None:
    h = IoMHasher(m=32, q=16)
    code = h.hash(unit(np.random.default_rng(2), 1)[0], h.matrices(SEED))
    assert h.similarity(code, code) == 1.0


def test_unrelated_faces_score_about_one_over_q() -> None:
    rng = np.random.default_rng(3)
    h = IoMHasher()  # m=256, q=16
    W = h.matrices(SEED)
    sims = h.similarity(h.hash_batch(unit(rng, 3000), W), h.hash_batch(unit(rng, 3000), W))
    assert sims.mean() == pytest.approx(1 / 16, abs=0.005)


def test_similar_faces_score_far_above_chance() -> None:
    rng = np.random.default_rng(4)
    h = IoMHasher()
    W = h.matrices(SEED)
    a = unit(rng, 300)
    b = a + 0.3 * unit(rng, 300)  # same "face" with noise, cosine about 0.96
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    assert float(np.sum(a * b, axis=1).mean()) > 0.9
    sims = h.similarity(h.hash_batch(a, W), h.hash_batch(b, W))
    assert sims.mean() > 0.5  # chance is 0.0625


def test_a_renewed_key_makes_the_same_face_look_unrelated() -> None:
    rng = np.random.default_rng(5)
    h = IoMHasher()
    X = unit(rng, 500)
    old = h.hash_batch(X, h.matrices(SEED))
    new = h.hash_batch(X, h.matrices(SEED_V2))
    assert h.similarity(old, new).mean() == pytest.approx(1 / 16, abs=0.01)


@pytest.mark.parametrize("q", [1, 0, 257])
def test_q_must_fit_in_uint8(q: int) -> None:
    with pytest.raises(ValueError, match="q must be"):
        IoMHasher(q=q)


def test_other_bad_parameters_and_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        IoMHasher(m=0)
    h = IoMHasher(m=8, q=4)
    W = h.matrices(SEED)
    with pytest.raises(ValueError, match="expected 512"):
        h.hash(np.ones(100, dtype=np.float32), W)
    with pytest.raises(ValueError, match="must have shape"):
        h.hash(np.ones(512, dtype=np.float32), W[:4])
    with pytest.raises(ValueError, match="expected 512"):
        h.hash_batch(np.ones((3, 100), dtype=np.float32), W)


def test_cached_matrices_are_shared_read_only_and_bounded() -> None:
    cached_matrices.cache_clear()
    a = cached_matrices(SEED, 8, 4, 512)
    assert cached_matrices(SEED, 8, 4, 512) is a
    assert np.array_equal(a, IoMHasher(8, 4, 512).matrices(SEED))
    with pytest.raises(ValueError, match="read-only"):
        a[0, 0, 0] = 1.0
    assert cached_matrices.cache_info().maxsize == 8
