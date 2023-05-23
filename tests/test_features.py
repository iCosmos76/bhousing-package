from bhousing_model.config.core import config
from bhousing_model.processing.features import ExtractLetterTransformer
import numpy as np


def test_temporal_variable_transformer(sample_input_data):
    # Given
    transformer = ExtractLetterTransformer(variables=config.model_config.var_for_letter_extraction)

    assert sample_input_data["B"].iat[17] == "386.75"
    assert sample_input_data["B"].iat[20] == "376.57"

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    # test 1
    test_subject_1 = subject["B"].iat[17]
    assert isinstance(test_subject_1, np.float64)
    assert test_subject_1 == "386"
    assert len(test_subject_1) == 3

    # test 2
    test_subject_2 = subject["B"].iat[20]
    assert isinstance(test_subject_2, np.float64)
    assert test_subject_2 == "376"
    assert len(test_subject_2) == 3
