import numpy as np
import pytest

pytest.importorskip("matplotlib")

from parity import curve, intersect


def test_curve_is_per_checkpoint() -> None:
    scores = np.array([[0, 10], [1, 20], [2, 30], [100, 40], [3, 50]], dtype=float)
    centre, low, high = curve(scores)
    np.testing.assert_array_equal(centre, [2.0, 30.0])
    assert (low <= centre).all()
    assert (centre <= high).all()


def test_bands_intersect_when_they_touch() -> None:
    a = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    b = (np.array([1.0, 1.5, -2.0]), np.array([2.0, 2.0, -1.0]))
    np.testing.assert_array_equal(intersect(a, b), [True, False, False])
