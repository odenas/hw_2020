
import logging
import warnings
from functools import reduce
from operator import concat
from itertools import product


import numpy as np

from .matrix import Matrix

log = logging.getLogger()


def run_strict(metric):
    """decorator for metric functions.

    returns a np.nan in case of a RuntimeWarning
    """

    def new(i, j, m):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                return metric(i, j, m)
            except RuntimeWarning:
                log.error("Runtime Warning: i=%d, j=%d" % (i, j))
                return np.nan
    return new


@run_strict
def euclidean_metric(i, j, m):
    """euclidean distance

    compute distance on the given sociomatrix and artist pair
    :param int i: artist id
    :param int j: artist id
    :param np.array m: sociomatrix
    :returns: the distance
    """

    row = np.dot(m[i, :] - m[j, :], m[i, :] - m[j, :])
    row -= (m[i, i] - m[j, i])*(m[i, i] - m[j, i])  # k != i
    row -= (m[i, j] - m[j, j])*(m[i, j] - m[j, j])  # k != j

    col = np.dot(m[:, i] - m[:, j], m[:, i] - m[:, j])
    col -= (m[i, i] - m[i, j])*(m[i, i] - m[i, j])  # k != i
    col -= (m[j, i] - m[j, j])*(m[j, i] - m[j, j])  # k != j

    d = row + col
    if d < 0:
        log.error("bad value entering pdb ...")
    return np.sqrt(d)


@run_strict
def correlation_metric(i, j, m):
    """correlation

    compute distance on the given sociomatrix and artist pair
    :param int i: artist id
    :param int j: artist id
    :param np.array m: sociomatrix
    :returns: the distance
    """

    N = float(m.shape[0] - 1)
    row_mean_i = (np.sum(m[i, :]) - m[i, i]) / N
    row_mean_j = (np.sum(m[j, :]) - m[j, j]) / N
    col_mean_i = (np.sum(m[:, i]) - m[i, i]) / N
    col_mean_j = (np.sum(m[:, j]) - m[j, j]) / N

    col_num = np.dot(m[:, i] - col_mean_i,  m[:, j] - col_mean_j)
    col_num -= (m[i, i] - col_mean_i)*(m[i, j] - col_mean_j)  # k != i
    col_num -= (m[j, i] - col_mean_i)*(m[j, j] - col_mean_j)  # k != j

    row_num = np.dot(m[i, :] - row_mean_i,  m[j, :] - row_mean_j)
    row_num -= (m[i, i] - row_mean_i)*(m[j, i] - row_mean_j)  # k != i
    row_num -= (m[i, j] - row_mean_i)*(m[j, j] - row_mean_j)  # k != j

    den_col_i = np.dot(m[:, i] - col_mean_i, m[:, i] - col_mean_i)
    den_col_i -= (m[i, i] - col_mean_i)*(m[i, i] - col_mean_i)  # k != i
    den_col_i -= (m[j, i] - col_mean_i)*(m[j, i] - col_mean_i)  # k != j

    den_row_i = np.dot(m[i, :] - row_mean_i,  m[i, :] - row_mean_i)
    den_row_i -= (m[i, i] - row_mean_i)*(m[i, i] - row_mean_i)  # k != i
    den_row_i -= (m[i, j] - row_mean_i)*(m[i, j] - row_mean_i)  # k != j

    den_col_j = np.dot(m[:, j] - col_mean_j,  m[:, j] - col_mean_j)
    den_col_j -= (m[i, j] - col_mean_j)*(m[i, j] - col_mean_j)  # k != i
    den_col_j -= (m[j, j] - col_mean_j)*(m[j, j] - col_mean_j)  # k != j

    den_row_j = np.dot(m[j, :] - row_mean_j,  m[j, :] - row_mean_j)
    den_row_j -= (m[j, i] - row_mean_j)*(m[j, i] - row_mean_j)  # k != i
    den_row_j -= (m[j, j] - row_mean_j)*(m[j, j] - row_mean_j)  # k != j

    num = col_num + row_num
    den = np.sqrt(den_col_i + den_row_i) * np.sqrt(den_col_j + den_row_j)

    if den:
        result = num / den
        if abs(result) > 1.001:
            log.error("bad correlation value(%f)! entering pdb ..." % result)
        return result
    else:
        return np.nan


@run_strict
def cosine_metric(i, j, m):
    """cosine distance

    compute distance on the given sociomatrix and artist pair
    :param int i: artist id
    :param int j: artist id
    :param np.array m: sociomatrix
    :returns: the distance
    """

    def stabilize(v):
        if v >= 0:
            return v
        if v < -0.000001:
            log.warning("suspicious value: %10f" % v)
            return 0

    dot_row = np.dot(m[i, :], m[j, :]) - (m[i, i]*m[j, i]) - (m[i, j]*m[j, j])
    dot_col = np.dot(m[:, i],  m[:, j]) - (m[i, i]*m[i, j]) - (m[j, i]*m[j, j])

    norm_row_i = np.dot(m[i, :],  m[i, :]) - (m[i, i]*m[i, i]) - (m[i, j]*m[i, j])
    norm_col_i = np.dot(m[:, i],  m[:, i]) - (m[i, i]*m[i, i]) - (m[j, i]*m[j, i])
    norm_i = stabilize(norm_row_i + norm_col_i)

    norm_row_j = np.dot(m[j, :],  m[j, :]) - (m[j, i]*m[j, i]) - (m[j, j]*m[j, j])
    norm_col_j = np.dot(m[:, j],  m[:, j]) - (m[i, j]*m[i, j]) - (m[j, j]*m[j, j])
    norm_j = stabilize(norm_row_j + norm_col_j)

    if norm_i and norm_j:
        result = (dot_row + dot_col) / (np.sqrt(norm_i) * np.sqrt(norm_j))
        if abs(result) > 1.001:
            log.error("bad cosine value(%f)! entering pdb ..." % result)
        return result
    else:
        return np.nan


dflist = {
    "cosine_metric": cosine_metric,
    "euclidean_metric": euclidean_metric,
    "correlation_metric": correlation_metric
}


class BlockMatrix(Matrix):
    def __init__(self, socio_matrix, distances, actors=None):
        self.socio_matrix = socio_matrix

        self.M = {}
        self.distances = distances

        if actors is None:
            actors = self.socio_matrix.V
        else:  # some might have not had a career till this year
            actors = set(actors) & set(self.socio_matrix.V.keys())
        log.info("%d senders will be considered" % len(actors))

        log.info("blockmatrix over:")
        for dn in distances:
            log.info("\t{}".format(dn))
            df = dflist["%s_metric" % dn]
            relation_matrices = (
                [socio_matrix.M['combined']] +
                list(map(lambda rn: socio_matrix.M[rn], socio_matrix.relations))
            )
            self.M[dn] = self.__auto_distance(relation_matrices, df, actors)

    def get(self, s, r, dname=None, **kwd):
        V = self.socio_matrix.V
        if dname:
            return self._get(s, r, self.M[dname], V, **kwd)
        return reduce(concat,
                      map(lambda dn: self._get(s, r, self.M[dn], V, **kwd), self.distances))

    def __auto_distance(self, M, metric_f, actors):
        for m in M[1:]:
            if m.shape != M[0].shape:
                _msg = "incompatible sociomatrix sizes: %s"
                raise ValueError(_msg % str(map(lambda m: str(m.shape), M)))

        DM = []
        V = self.socio_matrix.V
        for m in M:
            sq_dmat = np.zeros((m.shape[0], m.shape[0]), np.float32)
            for a1, a2 in product(actors, V.keys()):
                i, j = V[a1], V[a2]
                sq_dmat[i, j] = metric_f(i, j, m)
                sq_dmat[j, i] = sq_dmat[i, j]
            DM.append(sq_dmat)
        return DM
