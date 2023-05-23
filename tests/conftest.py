import pytest

from bhousing_model.config.core import config
from bhousing_model.processing.data_manager import load_dataset


@pytest.fixture()
def sample_input_data():
    return load_dataset(file_name=config.app_config.data_file)
