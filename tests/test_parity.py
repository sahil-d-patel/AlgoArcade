import numpy as np
import pytest

pytest.importorskip("matplotlib")

from parity import bootstrap, intersect, iqm


def test_iqm_cuts_a_quarter_from_each_end() -> None:
    # OPS-05's hand case.
    assert iqm(np.array([0, 1, 2, 3, 4, 5, 6, 100])) == 3.5
    # Below four seeds trim_mean cuts nothing.
    assert iqm(np.array([1.0, 2.0, 9.0])) == 4.0


def test_iqm_is_per_checkpoint() -> None:
    scores = np.array([[0, 10], [1, 20], [2, 30], [100, 40], [3, 50]])
    np.testing.assert_array_equal(iqm(scores), [2.0, 30.0])


def test_constant_scores_give_a_zero_width_band() -> None:
    low, high = bootstrap(np.full((5, 3), 0.4))
    np.testing.assert_allclose(low, [0.4] * 3)
    np.testing.assert_allclose(high, [0.4] * 3)


def test_bands_intersect_when_they_touch() -> None:
    a = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    b = (np.array([1.0, 1.5, -2.0]), np.array([2.0, 2.0, -1.0]))
    np.testing.assert_array_equal(intersect(a, b), [True, False, False])
