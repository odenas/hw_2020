
"""
defines classes for operations over the network such as block analysis
"""

import logging

import numpy  as np


log = logging.getLogger(__name__)


def similarity_function(A, B):
    """similarity function defined on sets or numerical values

    computes the similarity value of A and B. note that this
    is not symmetrical

    :param set/int A: first operand
    :param set/int B: second operand
    :returns: :math:`|A \\cap B| / |A|` if A and B are sets.
        :math:`1 - ( |A-B| / max\\{|A|, |B|\\} )` if A and B are ints.
    """

    weights = lambda _a: 1
    if type(A) != type(B):
        raise ValueError("A(%r) != B(%r)" % (type(A), type(B)))

    if type(A) == set:
        num = sum(map(lambda a: weights(a), (A & B)))
        den = float(sum(map(lambda a: weights(a), A)))
        return num / den
    elif type(A) == int:
        M = float(max(A, B))
        if M:
            return min(A, B) / M
        return 0
    else:
        raise ValueError("unsupported type %r" % type(A))


class Matrix(object):
    """
    stub class with a method for acessing values for a pair of artists
    """

    @classmethod
    def _get(cls, s, r, M, V, collector=np.sum, nanval=np.nan):
        """
        format the values of all matrices in M (M is a list)
        corresponding to s and r. Use V as an index to map the
        actor_id to a matrix index.

        if s=-1 (resp. r=-1) return the *collector* of values
        in the corresponding row (resp. column)

        if s,r are not indexed by V, return an empty string as
        the value in the matrix.

        :param int s: sender id
        :param int r: receiver id
        :param dict V: map of the type artist_id -> matrix index
        :param iterable M: a list of numpty arrays
        :param callable collector: a function to call on the M
            values indexed by s,r
        :param object nanval: representing missing values

        :returns: a list of strings of length len(M).
        """

        empty = tuple(map(lambda m: nanval, M))
        assert type(M) in (tuple, list), str(type(M))

        if (s, r) == (-1, -1):
            values = map(collector, M)
        elif s == -1:
            if not (r in V):
                return empty
            values = map(lambda m: collector(m[:, V[r]]), M)
        elif r == -1:
            if not (s in V):
                return empty
            values = map(lambda m: collector(m[V[s], :]), M)
        else:
            if not(s in V) or not(r in V):
                return empty
            values = map(lambda m: collector(m[V[s], V[r]]), M)
        return tuple(values)
