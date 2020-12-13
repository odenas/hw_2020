
import unittest

import numpy as np
from ghw.block_matrix import cosine_metric, cosine_metric_idx


class Test(unittest.TestCase):
    def setUp(self):
        pass

    def _comp(self, m, f):
        res = np.zeros_like(m)
        for i in range(m.shape[0]):
            for j in range(m.shape[0]):
                res[i, j] = f(i, j, m)
        return res

    def test_equal(self):
        n = 10
        m = np.random.rand(n, n).astype(np.float32)
        m = np.array([[1, 3, 4], [2, 2, 5]]).astype(np.float32)

        np.testing.assert_almost_equal(
            self._comp(m, cosine_metric),
            self._comp(m, cosine_metric_idx),
            decimal=4,
        )


if __name__ == '__main__':
    unittest.main()
