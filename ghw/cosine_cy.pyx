import numpy as np
cimport cython


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef dot(float[:] a, float[:] b, float correct):
    cdef Py_ssize_t n = a.shape[0]
    cdef Py_ssize_t k
    cdef float dot_row = 0
    for k in range(n):
        dot_row += a[k] * b[k]
    return dot_row - correct


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def cosine_metric(Py_ssize_t i, Py_ssize_t j, float[:, :] m):
    """cosine distance

    compute distance on the given sociomatrix and artist pair
    :param int i: artist id
    :param int j: artist id
    :param np.array m: sociomatrix
    :returns: the distance
    """

    cdef float epsilon = 1e-4

    cdef float mii = m[i, i]
    cdef float mij = m[i, j]
    cdef float mji = m[j, i]
    cdef float mjj = m[j, j]

    cdef float[:] mi_ = m[i, :]
    cdef float[:] m_i = m[:, i]

    cdef float[:] mj_ = m[j, :]
    cdef float[:] m_j = m[:, j]

    cdef float dot_row = dot(mi_, mj_, (mii*mji) + (mij*mjj))
    cdef float dot_col = dot(m_i, m_j, (mii*mij) + (mji*mjj))

    cdef float norm_row_i = dot(mi_,  mi_, (mii*mii) + (mij*mij))
    cdef float norm_col_i = dot(m_i,  m_i, (mii*mii) + (mji*mji))
    cdef float norm_i = norm_row_i + norm_col_i
    if norm_i < -epsilon:
        raise ValueError("suspicious value: %10f" % norm_i)

    cdef float norm_row_j = dot(mj_,  m_j, (mji*mji) + (mjj*mjj))
    cdef float norm_col_j = dot(m_j,  m_j, (mij*mij) + (mjj*mjj))
    cdef float norm_j = norm_row_j + norm_col_j
    if norm_j < -epsilon:
        raise ValueError("suspicious value: %10f" % norm_j)

    if norm_i and norm_j:
        result = (dot_row + dot_col) / (np.sqrt(norm_i) * np.sqrt(norm_j))
        #if result > 1 + epsilon:
        #    raise ValueError("bad cosine value(%f)! ..." % result)
        return result
    return 0.0
