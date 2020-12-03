import numpy as np

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
            raise ValueError("suspicious value: %10f" % v)

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
            raise ValueError("bad cosine value(%f)! ..." % result)
        return result
    else:
        return np.nan
