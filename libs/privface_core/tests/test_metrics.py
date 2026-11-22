import numpy as np
import pytest
from privface_core.metrics import eer, roc_points, tar_at_far, tenfold_accuracy

RNG = np.random.default_rng(0)


def normals(mu_gen: float, n: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    return RNG.normal(mu_gen, 1, n), RNG.normal(0, 1, n)


def test_eer_matches_the_analytic_value_for_two_normals() -> None:
    gen, imp = normals(2.0)
    value, threshold = eer(gen, imp)
    assert value == pytest.approx(0.1587, abs=0.01)  # Phi(-1)
    assert threshold == pytest.approx(1.0, abs=0.1)  # halfway between the means


def test_eer_is_zero_when_perfectly_separated_and_half_when_identical() -> None:
    assert eer([5, 6, 7], [0, 1, 2])[0] == 0.0
    gen, imp = normals(0.0)
    assert eer(gen, imp)[0] == pytest.approx(0.5, abs=0.02)


def test_tar_at_far_matches_the_analytic_value() -> None:
    gen, imp = normals(3.0, n=100_000)
    tar, t = tar_at_far(gen, imp, 0.01)
    assert t == pytest.approx(2.326, abs=0.1)  # 99th percentile of N(0,1)
    assert tar == pytest.approx(0.75, abs=0.02)  # P(N(3,1) > 2.326)
    assert float(np.mean(imp > t)) <= 0.01  # achieved FAR never exceeds the target


def test_tar_at_far_needs_enough_impostor_scores() -> None:
    with pytest.raises(ValueError, match="at least 10000 impostor"):
        tar_at_far([1.0] * 5, [0.0] * 9999, 1e-4)
    with pytest.raises(ValueError, match="between 0 and 1"):
        tar_at_far([1.0], [0.0] * 100, 0)


def test_roc_points_are_thinned_monotonic_and_report_auc() -> None:
    gen, imp = normals(2.0, n=5000)
    roc = roc_points(gen, imp, n=100)
    assert len(roc["fpr"]) == len(roc["tpr"]) <= 100
    assert roc["fpr"][0] == 0.0 and roc["fpr"][-1] == 1.0
    assert np.all(np.diff(roc["fpr"]) >= 0) and np.all(np.diff(roc["tpr"]) >= 0)
    assert roc["auc"] == pytest.approx(0.921, abs=0.01)  # Phi(2 / sqrt(2))


def test_tenfold_accuracy_perfect_and_chance() -> None:
    folds = np.repeat(np.arange(10), 60)
    issame = np.tile([True, False], 300)
    perfect = np.where(issame, 0.9, 0.1) + RNG.normal(0, 0.01, 600)
    assert tenfold_accuracy(perfect, issame, folds) == pytest.approx((1.0, 0.0), abs=1e-9)
    mean, std = tenfold_accuracy(RNG.random(600), issame, folds)
    assert 0.4 < mean < 0.65 and std < 0.1


def test_tenfold_accuracy_threshold_is_not_chosen_on_the_test_fold() -> None:
    # Fold 9 is flipped (genuine scores low). A threshold fitted on it would score 100%.
    folds = np.repeat(np.arange(10), 60)
    issame = np.tile([True, False], 300)
    scores = np.where(issame, 0.9, 0.1)
    scores[folds == 9] = 1 - scores[folds == 9]
    accs_mean, _ = tenfold_accuracy(scores, issame, folds)
    assert accs_mean == pytest.approx(0.9, abs=1e-9)  # nine perfect folds, one all wrong


def test_bad_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        eer([], [1.0])
    with pytest.raises(ValueError, match="finite"):
        eer([np.nan, 1.0], [0.0])
    with pytest.raises(ValueError, match="same length"):
        tenfold_accuracy(np.zeros(5), np.zeros(4, dtype=bool), np.zeros(5))
