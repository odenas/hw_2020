import logging
import numpy as np
from itertools import product
import pandas as pd
import sqlite3

from .affiliations import Affiliations
from .matrix import Matrix

log = logging.getLogger()


def floyd(M, weighted):
    """
    The Floyd-Wallshall algorithm for the shortest path.

    Because we work on similarity matrices and the
    algorithm on distance matrices a conversion needs
    to be made. I set the weight matrix to 1 / M.

    :param array M: a (square) similarity matrix.
    :return: the shortest path matrix w_{ij}.
    """

    n = M.shape[0]
    M = 1/M

    if not weighted:
        M[M > 0] = 1

    # log.error("skipping ...")
    for k in range(n):
        M = np.minimum(M, np.add.outer(M[:, k], M[k, :]))
    return M


def closure(A, nA, B, nB):
    """transitive closure

    compute the scalar product on the indicator matrices
    of the given sociomatrices

    :param np.array A: sociomatrix
    :param dict nA: indexes of artists in the matrix
    :param np.array B: sociomatrix
    :param dict nB: indexes of artists in the matrix
    :returns: np.array, dict
    """

    def identity(v):
        if v > 0:
            return 1
        return 0

    nm = set(nA.keys()) & set(nB.keys())
    nmi = dict(map(lambda t: (t[1], t[0]), enumerate(nm)))

    va = np.zeros((len(nm), len(nm)), np.int8)
    vb = np.zeros((len(nm), len(nm)), np.int8)

    # log.error("Skipping ... ")
    for r, c in product(nm, nm):
        i, j = nmi[r], nmi[c]
        va[i, j] = identity(A[nA[r], nA[c]])
        vb[i, j] = identity(B[nB[r], nB[c]])
    result = np.dot(va, vb)
    return (result, nmi)
    return va, nmi


def centrality(T, V, i):
    """centrality of given actor

    compute the centrality of actor *i* over the
    tie strength matrix T. I.e., count the
    non-zero components of the row of T corresponding
    to *i*

    :param np.array T: tie strength matrix (nr. of shared films)
    :param dict V: maps artist_id to index
    :param int i: row index
    :returns: :math:`sum_j T'_{i,j}` where *T'* is the indicator
        matrix of *T*
    """

    if not (i in V):
        return np.nan
    return len(list(filter(lambda v: v > 0, T[V[i], :].flat)))


def bonacich_centrality(M):
    """compute the Bonacich centrality on M

    this is implemented as in
    https://facwiki.cs.byu.edu/DML/index.php/Eigenvector_Centrality
    and
    http://www-personal.umich.edu/~mejn/papers/palgrave.pdf (page 5) """

    # log.error("skipping...")
    # return  np.zeros( (M.shape[0],), dtype=np.int8 )
    try:
        l, v = np.linalg.eig(M)
    except np.linalg.linalg.LinAlgError:
        log.critical("cannot compute the Bonacich Centrality: eig did not converge")
        evec = np.zeros((M.shape[0],), dtype=np.int8)
        # eval = 0
    else:
        evec = v[:, np.argmax(l)]
        # eval = np.max(v)
    return np.abs(evec)


class Report(object):
    """represents the results of a variety of calculation analyses

        the object maintains the data that is the
        basis for any of the supported analyses:

        * SM, the sociomatrix
        * BM, the blockmatrix
        * Y, the reference year (on which the naming happens)

        Using these pieces of data, the object will run the analysis
        on instantiation. One can get the result row of a particular
        (sender, receiver, year, trial) by using the :py:meth:`Report.get` function

        to instantiate:

        :param int year: year
        :param instance adata: instance of :py:class:`giacomo.hollywood.bmat.Affiliations`
        :param instance bdata: instance of :py:class:`giacomo.hollywood.bmat.BlacklistData`
        :param instance SM: instance of :py:class:`giacomo.hollywood.op.SocioMatrix`
        :param list BM: instance of :py:class:`giacomo.hollywood.op.BlockMatrix`
        :param list TS: tie strenths
        :param list dist_names: list of names of distances (used to refernece
            distance functions)
        :param list rel_names: orderedDict of relations and attributes of
            the type relation --> attribute (used to refernece relation functions)
        :param dict weights: dictionary of the type attribute_value --> weight
    """

    def __init__(self, year, dbpath, SM, BM, TS, dist_names, rel_names):
        self._dbpath = dbpath
        self.header = self.__set_header(rel_names, dist_names)
        self.affiliations = Affiliations()
        self.B_data = dict(
            ((row.sender, row.receiver, row.year), row)
            for row in pd.read_sql_query('select * from trials',
                                         sqlite3.connect('data/input/adata.db')).itertuples()
        )

        self.year = year
        self.actors = SM.artists.keys()

        self.SM = SM
        self.BM = BM

        self.namings_idx, self.namings = self._naming(self.actors, self.year-1)

        self.tstrengths = list(TS)
        self.V = self.tstrengths[0].artists
        assert set(self.V.keys()) == set(self.SM.artists.keys())
        log.info("reporting on %d actors ..." % len(self.actors))
        log.info("\tbonacich centrality")
        self.boncent = bonacich_centrality(self.tstrengths[0].matrix)
        log.info("\tweighted shortest paths...")
        self.weighted_shpaths = floyd(self.tstrengths[0].matrix, weighted=True)
        log.info("\tshortest paths...")
        self.shpaths = floyd(self.tstrengths[0].matrix, weighted=False)

        log.info("\ttriadic closure (1 of 4) ...")
        self.tr1, self.tr1_idx = closure(self.tstrengths[0].matrix, self.V,
                                         self.namings, self.namings_idx)
        log.info("\ttriadic closure (2 of 4) ...")
        self.tr2, self.tr2_idx = closure(self.namings, self.namings_idx,
                                         self.tstrengths[0].matrix, self.V)
        log.info("\ttriadic closure (3 of 4) ...")
        self.tr3, self.tr3_idx = closure(self.namings, self.namings_idx,
                                         self.namings, self.namings_idx)
        log.info("\ttriadic closure (4 of 4) ...")
        self.tr4, self.tr4_idx = closure(self.tstrengths[0].matrix, self.V,
                                         self.tstrengths[0].matrix, self.V)

    def _naming(self, actors, year):
        ai = dict(map(lambda t: (t[1], t[0]), enumerate(actors)))
        n = len(actors)
        m = np.zeros((n, n), dtype=np.int8)

        df = pd.read_sql_query('select sender, receiver, year from trials',
                               sqlite3.connect('data/input/adata.db'))
        for row in df.itertuples():
            if row.year <= year and (row.sender in ai) and (row.receiver in ai):
                m[ai[row.sender], ai[row.receiver]] += 1
                log.info("%d namings" % (np.sum(m)))
        return ai, m

    def _reciprocity(self, s, r, y):
        trial = ((s, r, y) in self.B_data and self.B_data[(s, r, y)].trial or 1000)

        def cond(n):
            return ((n.sender, n.receiver) == (r, s) and
                    ((n.year < y) or (n.year == y and n.trial < trial)))

        event = list(filter(cond, self.B_data.values()))
        assert len(event) < 2
        return len(event), (len(event) and (y - event[0].year) or 0)

    def get(self, s, r, Y, t, receiver_card):
        def format_val(val):
            """format a value into a string

            :param object val: the value to format. allowed
                types are: np.infinite, int, np.int, str,
                float, np.float
            :returns: a string
            """

            tval = type(val)

            if not np.isfinite(val):
                return " "
            if tval in (float, np.float32, np.float64):
                return "%.4f" % val
            elif tval in (int, np.int64, np.int8):
                return "%d" % val
            elif tval in (np.bool_, bool):
                return (val and '1' or '0')
            elif tval != str:
                raise ValueError("unexpected type, entering pdb ...")
            return val

        def bonCentral(a):
            if a in self.V:
                return self.boncent[self.V[a]]
            return 0

        resrow = (
            (s, r, Y, t, ((s, r, Y) in self.B_data)) +
            Matrix._get(s, r, [self.SM.matrix], self.SM.artists) +
            self.BM.get(s, r, None) +
            Matrix._get(s, -1, [self.namings], self.namings_idx) +
            Matrix._get(-1, r, [self.namings], self.namings_idx) +
            self._reciprocity(s, r, self.year) +
            # might be empty since dist_lst wants both s,r in V.keys()
            Matrix._get(s, r, [self.tstrengths[0].matrix,
                               self.tstrengths[1].matrix,
                               self.tstrengths[2].matrix], self.V) +
            (self.affiliations.corr(s, r), self.affiliations.sim(s, r)) +
            Matrix._get(s, r, [self.weighted_shpaths, self.shpaths], self.V) +
            Matrix._get(s, r, [self.tr1], self.tr1_idx) +
            Matrix._get(s, r, [self.tr2], self.tr2_idx) +
            Matrix._get(s, r, [self.tr3], self.tr3_idx) +
            Matrix._get(s, r, [self.tr4], self.tr4_idx) +
            (receiver_card,) +
            (centrality(self.tstrengths[0].matrix, self.V, s),
             centrality(self.tstrengths[0].matrix.T, self.V, r),
                bonCentral(s), bonCentral(r)) +
            (self.affiliations.corr_comm(s),
             self.affiliations.corr_comm(r),
             self.affiliations.sums(s),
             self.affiliations.sums(r)) +
            ()
        )
        assert len(resrow) == len(self.header), "%d != %d" % (len(resrow), len(self.header))
        return map(format_val, resrow)

    def __set_header(self, rel_names, dist_names):
        __hdrel = [rel_names]
        header = ['sender', 'receiver', 'year', 'trial', 'naming']
        # return header
        header += map(lambda s: "similarity_%s" % s, __hdrel)
        for dn in dist_names:
            header += map(lambda s: "%s_%s" % (dn, s), __hdrel)
        header += (["prior_naming_sender", "prior_naming_receiver"] +
                   ["reciprocity", "reciprodicy_clock"] +
                   ["tie_strength_sender_den", "tie_strength_receiver_den",
                    "tie_strength_intersect"] +
                   ["affiliation_corr", "affiliation_sim"] +
                   ["weighted_shortest_path", "shortest_path"] +
                   ["tr1", "tr2", "tr3", "tr4"] +
                   ["receiver_card"] +
                   ["centrality_s", "centrality_r", "bonacich_centrality_s",
                    "bonacich_centrality_r"] +
                   ["comm_corr_s", "comm_corr_r", "summ_s", "summ_r"])
        return header
