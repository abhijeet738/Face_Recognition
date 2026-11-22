import numpy as np
import pytest
from privface_core.unlinkability import d_sys

RNG = np.random.default_rng(0)


def test_identical_distributions_are_unlinkable() -> None:
    a, b = RNG.normal(0.06, 0.02, 50_000), RNG.normal(0.06, 0.02, 50_000)
    assert d_sys(a, b) < 0.05


def test_disjoint_distributions_are_fully_linkable() -> None:
    mated, nonmated = RNG.uniform(0.6, 0.9, 5000), RNG.uniform(0.0, 0.3, 5000)
    assert d_sys(mated, nonmated) == pytest.approx(1.0)


def test_d_sys_grows_as_the_distributions_separate() -> None:
    nonmated = RNG.normal(0, 1, 50_000)
    values = [d_sys(RNG.normal(shift, 1, 50_000), nonmated) for shift in (0.0, 0.5, 1.0, 2.0, 4.0)]
    assert values == sorted(values)
    assert values[0] < 0.05 and values[-1] > 0.9


def test_result_is_always_between_zero_and_one() -> None:
    for shift in (0.0, 1.0, 3.0):
        value = d_sys(RNG.normal(shift, 1, 2000), RNG.normal(0, 1, 2000))
        assert 0.0 <= value <= 1.0


def test_a_prior_odds_below_one_lowers_the_score() -> None:
    mated, nonmated = RNG.normal(1, 1, 20_000), RNG.normal(0, 1, 20_000)
    assert d_sys(mated, nonmated, omega=0.1) < d_sys(mated, nonmated, omega=1.0)


def test_all_identical_scores_carry_no_information() -> None:
    assert d_sys([0.5] * 10, [0.5] * 10) == 0.0


def test_bad_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        d_sys([], [1.0])
    with pytest.raises(ValueError, match="finite"):
        d_sys([np.nan], [1.0])
    with pytest.raises(ValueError, match="omega"):
        d_sys([1.0, 2.0], [1.0, 2.0], omega=0)
