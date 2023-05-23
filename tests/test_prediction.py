import numpy as np

from bhousing_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_shape = (506, 15)

    # When
    result = make_prediction(input_data=sample_input_data)
    # Then
    preds = result.get("preds")

    assert isinstance(preds, list)
    assert isinstance(preds[0], np.float64)
    assert result.get("errors") is None
    assert len(preds) == expected_shape[0]
