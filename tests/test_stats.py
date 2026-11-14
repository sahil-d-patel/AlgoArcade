import warnings

import pytest

from algoarcade.training import stats


def test_iqm_trims_a_quarter_from_each_end():
    assert stats.iqm([0, 1, 2, 3, 4, 5, 6, 100]) == 3.5


def test_iqm_of_three_seeds_warns_that_it_is_the_mean():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert stats.iqm([1.0, 2.0, 6.0]) == 3.0
    assert [str(w.message) for w in caught] == [
        "3 seeds is exploratory: a headline needs five",
        "IQM of 3 seeds trims nothing: it equals the mean",
    ]


def test_probability_of_improvement_counts_a_tie_as_half():
    with pytest.warns(UserWarning, match="2 seeds"):
        assert stats.probability_of_improvement([1, 2], [1, 0]) == 3.5 / 4
    with pytest.warns(UserWarning, match="2 seeds"):
        assert stats.probability_of_improvement([1, 0], [1, 2]) == 0.5 / 4


def test_interval_of_equal_scores_is_exact():
    assert stats.interval([0.4] * 5) == (0.4, 0.4)


def test_interval_holds_the_iqm_and_is_reproducible():
    scores = [0.1, 0.5, 0.2, 0.9, 0.4, 0.3]
    low, high = stats.interval(scores)
    assert low <= stats.iqm(scores) <= high
    assert stats.interval(scores) == (low, high)
    assert stats.interval(scores, level=0.5) != (low, high)


def test_pooled_tasks_resample_within_each_task():
    # Each resample keeps five scores from each task, so its IQM is always 5. Resampling the
    # pooled list would draw uneven counts and spread the interval.
    low, high = stats.interval({"a": [0.0] * 5, "b": [10.0] * 5})
    assert (low, high) == (5.0, 5.0)


def test_performance_profile_is_the_fraction_above_each_threshold():
    with pytest.warns(UserWarning, match="4 seeds is exploratory"):
        profile = stats.performance_profile([0.1, 0.4, 0.6, 0.9], [0.0, 0.5, 1.0])
    assert profile.tolist() == [1.0, 0.5, 0.0]


def test_every_printed_estimate_has_its_seed_count():
    scores = [0.62, 0.48, 0.71, 0.55, 0.6]
    assert stats.describe(scores).startswith("IQM ")
    assert stats.describe(scores).endswith("(n=5)")
    others = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert stats.describe_improvement(scores, others).endswith("(n=5, m=5)")


def test_pooled_tasks_print_seeds_per_task():
    three = {"a": [0.1, 0.2, 0.3], "b": [0.4, 0.5, 0.6], "c": [0.7, 0.8, 0.9]}
    with pytest.warns(UserWarning, match="3 seeds"):
        assert stats.describe(three).endswith("(n=3 per task, 3 tasks)")
    uneven = {"a": [0.1, 0.2, 0.3, 0.4, 0.5], "b": [0.6, 0.7, 0.8, 0.9]}
    with pytest.warns(UserWarning, match="4 seeds"):
        assert stats.describe(uneven).endswith("(n=4 per task, 2 tasks)")


def test_five_seeds_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stats.describe([0.1, 0.2, 0.3, 0.4, 0.5])
        stats.probability_of_improvement([0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5])


def test_four_seeds_are_exploratory_but_trimmed():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stats.iqm([0.1, 0.2, 0.3, 0.4])
    assert [str(w.message) for w in caught] == ["4 seeds is exploratory: a headline needs five"]


def test_fewer_than_five_seeds_warns_for_every_estimate():
    with pytest.warns(UserWarning, match="3 seeds"):
        stats.describe([0.1, 0.2, 0.3])
    with pytest.warns(UserWarning, match="4 seeds"):
        stats.describe_improvement([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4, 0.5])


def test_each_warning_names_the_callers_line():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stats.describe([0.1, 0.2, 0.3])
        stats.describe_improvement([0.1, 0.2], [0.3, 0.4])
        stats.interval({"a": [0.1, 0.2], "b": [0.3, 0.4]})
    assert len(caught) == 4
    assert {w.filename for w in caught} == {__file__}
