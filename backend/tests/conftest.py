import pytest

@pytest.fixture
def sample_features():
    import numpy as np
    return np.array([[0.5, 0.8, 0.7, 0.6]])