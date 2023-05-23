import numpy as np

from bhousing_model.config.core import config


def test_temporal_variable_transformer(sample_input_data):
    assert sample_input_data["B"].iat[17] == 386.75
    assert sample_input_data["B"].iat[20] == 376.57

    subject = sample_input_data

    # Then
    # test 1
    test_subject_1 = subject["B"].iat[17]
    assert isinstance(test_subject_1, np.float64)
    assert test_subject_1 == 386.75
    lengths_1 = np.char.str_len(np.char.mod("%d", test_subject_1))
    assert lengths_1 == 3

    # test 2
    test_subject_2 = subject["B"].iat[20]
    assert isinstance(test_subject_2, np.float64)
    assert test_subject_2 == 376.57
    lengths_2 = np.char.str_len(np.char.mod("%d", test_subject_2))
    assert lengths_2 == 3
