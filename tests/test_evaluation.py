import numpy as np
import pytest

from mi_lab.evaluation import binary_metrics, bootstrap_intervals, choose_threshold


def test_confusion_metrics_and_positive_class():
    result = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.8, 0.4]), 0.5)
    assert result['confusion_matrix'] == [[1, 1], [1, 1]]
    assert result['sensitivity'] == 0.5
    assert result['specificity'] == 0.5
    assert result['auroc'] == 0.75


def test_threshold_comes_from_validation_and_can_separate_classes():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.6, 0.8])
    threshold = choose_threshold(y, p)
    assert 0.3 < threshold <= 0.6
    assert binary_metrics(y, p, threshold)['balanced_accuracy'] == 1


def test_bootstrap_is_reproducible_and_perfect_ranking_is_one():
    y = np.tile([0, 1], 20)
    p = y * 0.8 + 0.1
    a = bootstrap_intervals(y, p, 0.5, n_bootstrap=30, seed=4)
    b = bootstrap_intervals(y, p, 0.5, n_bootstrap=30, seed=4)
    assert a == b
    assert a['auroc'] == [1.0, 1.0]


@pytest.mark.parametrize('p', [[0.2, float('nan')], [-0.1, 0.9], [0.1, 1.1]])
def test_invalid_probabilities_rejected(p):
    with pytest.raises(ValueError):
        binary_metrics(np.array([0, 1]), np.array(p), 0.5)
