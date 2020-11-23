import os
import numpy as np
import csv

from .matrix import similarity_function
from .weights import Weight


class Affiliations(object):
    """represents affiliations of a set of actors. maintains

    data
        a dict structure of the type (castid) --> array,
        where array is an indicator array for each of the
        organizations

    affiliations
        a list of names of affiliations

    the data is in the configuration file so this gets
    initialized automatically
    """

    def __init__(self):
        fname = os.path.join(os.path.dirname(__file__), "affiliations.csv")
        reader = csv.reader(open(fname, 'rt', encoding='latin1'))
        self.data = {}
        for i, row in enumerate(reader):
            if i == 0:
                self.affiliations = row[3:]
                W = len(self.affiliations)
                continue
            if row[1] == '':
                continue
            self.data[int(row[1])] = np.array(list(map(int, row[3:])), dtype=np.int16)
            assert len(row[3:]) == W, "[%s] %d != %d" % (row[1], len(row[3:]), W)

    def corr(self, s, r):
        """affiliation-based correlation of two actors

        :param int s: sender
        :param int r: receiver
        :returns: correlation or 0 if not in the data
        """

        if s in self.data and r in self.data:
            a, b = self.data[s], self.data[r]
            C = np.corrcoef(np.vstack((a, b)))
            assert np.abs(C[0, 1] - C[1, 0]) < 0.000001
            return float(C[0, 1])
        return 0

    def sim(self, s, r):
        """affiliation-based similarity of two actors

        :param int s: sender
        :param int r: receiver
        :returns: similarity or 0 if (s,r) not in the data
        """

        to_lst = lambda a: set(self.data.get(a, np.array([], dtype=np.float32)).tolist())

        A = set(to_lst(s))
        B = set(to_lst(r))
        if not A:
            return 0
        return similarity_function(A, B, Weight.one)

    def __pairwise_f(self, f):
        for i in self.data:
            for j in self.data:
                yield (i, j, f(i, j))

    def pairwise_corr(self):
        """generate correlations over the affiliation array
        for each pair of actors

        this calls :py:meth:`Affiliations.corr` on pairs

        :returns: yields tuples (actorid, actorid, corr)
        """

        self.__pairwise_f(self.corr)

    def corr_comm(self, artist):
        """is the artist part of comm_pol_ass or comm_party?

        :param int artist: the artist
        :returns: bool
        """

        assert 'comm_pol_ass' in self.affiliations and 'comm_party' in self.affiliations, str(self.affiliations)

        if artist in self.data:
            i1 = self.affiliations.index('comm_pol_ass')
            i2 = self.affiliations.index('comm_party')
            t = self.data[artist]
            return t[i1] + t[i2] > 0
        return False

    def sums(self, artist):
        if artist in self.data:
            i = min(self.affiliations.index('comm_pol_ass'),
                    self.affiliations.index('comm_party'))
            return np.sum(self.data[artist][:i])
        return 0
