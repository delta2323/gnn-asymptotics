import mock
import pytest
from unittest import mock
import numpy as np

import lib.run.run


def test_run():
    with mock.patch('lib.run.run.run_single',
               return_value=(0.1, 0.2, 0.3)) as mock_method:
        train_acc, val_acc, test_acc = lib.run.run.run(None, None, 10)
        np.testing.assert_almost_equal(train_acc, 0.1)
        np.testing.assert_almost_equal(val_acc, 0.2)
        np.testing.assert_almost_equal(test_acc, 0.3)
        assert mock_method.call_count == 10
