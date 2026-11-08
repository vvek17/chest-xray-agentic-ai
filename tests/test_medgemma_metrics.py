import pytest
from prompt_engineering.evaluate_medgemma import metrics


def test_unparsed_answers_count_as_missed_cases():
    result = metrics([
        {'true': 'Normal', 'predicted': 'Normal'},
        {'true': 'Normal', 'predicted': 'Pneumonia'},
        {'true': 'Pneumonia', 'predicted': 'Other'},
        {'true': 'Pneumonia', 'predicted': 'Pneumonia'},
    ])
    assert result['accuracy'] == .5
    assert result['unparsed_answers'] == 1
    assert result['confusion_matrix'] == [[1, 1, 0], [0, 1, 1], [0, 0, 0]]
    assert result['per_class']['Pneumonia']['recall'] == .5
    assert result['per_class']['Pneumonia']['f1-score'] == .5
    assert result['macro_f1'] == pytest.approx((2/3 + .5) / 2)
