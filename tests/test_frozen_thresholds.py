import numpy as np
import pytest

from training.validate_frozen_thresholds import choose_threshold, exact_mcnemar, wilson


def test_threshold_respects_inclusive_ties_and_sensitivity():
    truth = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    scores = np.array([-2, 0, 3, -1, 1, 1, 2, 4])
    result = choose_threshold(truth, scores)
    assert result["threshold"] == 1
    assert result["dev_counts"] == {"TP": 4, "FN": 1, "TN": 2, "FP": 1}


def test_threshold_tie_uses_highest_cutoff():
    truth = np.array([0, 0, 1, 1, 1, 1, 1])
    scores = np.array([-3, -2, 0, 1, 2, 3, 4])
    assert choose_threshold(truth, scores)["threshold"] == 1


def test_nonfinite_threshold_input_fails():
    with pytest.raises(ValueError, match="Non-finite"):
        choose_threshold([0, 1], [0, float("nan")])


def test_wilson_and_exact_paired_test():
    lower, upper = wilson(80, 100)
    assert lower == pytest.approx(0.7111708344)
    assert upper == pytest.approx(0.8666330667)
    truth = np.array([0, 0, 1, 1])
    result = exact_mcnemar(truth, truth, np.array([-1, 1, 0, 0]))
    assert result["method_correct_baseline_wrong"] == 4
    assert result["method_wrong_baseline_correct"] == 0
    assert result["exact_two_sided_p"] == 0.125
