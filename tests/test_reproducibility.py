import random

import numpy as np

from v_fashion_insight.common.reproducibility import seed_everything


def test_seed_everything_repeats_python_and_numpy_sequences() -> None:
    seed_everything(123)
    first_python_values = [random.random() for _ in range(3)]
    first_numpy_values = np.random.random(3)

    seed_everything(123)
    second_python_values = [random.random() for _ in range(3)]
    second_numpy_values = np.random.random(3)

    assert second_python_values == first_python_values
    np.testing.assert_array_equal(second_numpy_values, first_numpy_values)
