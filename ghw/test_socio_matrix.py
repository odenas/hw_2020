import unittest

import pandas as pd
from ghw.socio_matrix import SocioMatrix

class Test(unittest.TestCase):
    def setUp(self):
        self.ddf = pd.DataFrame(
                data=[(0, 12, set([1, 2, 3])), (1, 31, set([3])), (2, 22, set([3, 2]))],
                columns=['index', 'castid', 'relation']
        )

    def test_adjm(self):
        sm = SocioMatrix._adj_matrix(self.ddf.relation, SocioMatrix.sim_functions('genre'))
        self.assertEqual(sm.shape[1], sm.shape[0])


if __name__ == '__main__':
    unittest.main()

