
"""defines classes for operations over the network such as block analysis"""

import sys, pdb, logging
from operator import concat
import inspect, warnings
from itertools import combinations
from . import *

import bmat

log = logging.getLogger(__name__)


class Matrix(object):
    """stub class with a method for acessing values for a pair of artists"""

    @classmethod
    def _get(cls, s, r, M, V, collector=np.sum, nanval=np.nan):
        """format the values of all matrices in M (M is a list)
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
            values = map(lambda m: collector(m[:,V[r]]), M)
        elif r == -1:
            if not (s in V):
                return empty
            values = map(lambda m: collector(m[V[s],:]), M)
        else:
            if not(s in V) or not(r in V):
                return empty
            values = map(lambda m: collector(m[V[s], V[r]]), M)
        return tuple(values)


class SocioMatrix(Matrix):
    """matrix defined over the similarity (jaccard) measure. maintains

    M an ordered dict of the type (relation_name) ---> matrix

    V dict of the type (artist) --> index

    to initialize needs:

    data
       :py:class:`giacomo.hollywood.bmat.ArtistInfoData`
    year
       int
    actors
       iterable
    relations
       a dictionary of the type (relation_name) --> attribute_name
    weights
       dictionary mapping attributes to weights
    combined
        bool
    kwd
       other options passed by name to ArtistInfoData.adj_matrix
    """

    # TODO: combined should be part of the relations argument
    def __init__(self, data, year, actors, relations, weights, combined=True, **kwd):
        self.relations = relations
        self.M = OrderedDict()
        self.V = None
        log.info("sociomatrix (%d actors) over relations:" % len(actors))
        for relname, relval in relations.iteritems():
            v, m = data.adj_matrix(year, relval, actors, weights.weights_of(relname), **kwd)
            self.M[relname] = m
            log.info("\t%s (%d actors)" % (relname, len(v)))
        if not self.V:
            self.V = v
        if set(v.keys()) < set(self.V.keys()):
            self.V = v

        if combined:
            N = float(len(self.M))
            self.M["combined"] = sum(self.M.values()) / N

    def get(self, s, r, **kwd):
        """calls :py:func:`Matrix._get` on all sociomatrices"""

        mlist = [self.M['combined']] + map(lambda rn: self.M[rn], self.relations)
        return self._get(s, r, mlist, self.V, **kwd)


class BlockMatrix(Matrix):
    """performs block analysis on the given :py:class:`SocioMatrix`

    maintains:
      * M, a dict of the type (distance_name) --> [distance matrix for all socio matrices]

      * socio_matrix, a refernence to the :py:class:`SocioMatrix` instance

    to initialize:
      * socio_matrix, a refernence to the :py:class:`SocioMatrix` instance

      * distances, a list of distance names (will look for dname_metric in this module)

      * actors, a list of actor ids for which you want the positional distances (on the whole net)
    """

    def __init__(self, socio_matrix, distances, actors=None):
        self.socio_matrix = socio_matrix

        self.M = {}
        dflist = dict( filter(lambda (n,f): n.endswith("metric"),
                inspect.getmembers(sys.modules[__name__], inspect.isfunction)) )
        self.distances = distances
        log.info("blockmatrix over:")
        for dn in distances:
            log.info("\t{}".format(dn))
            df = dflist["%s_metric" % dn]
            relation_matrices = [socio_matrix.M['combined']] + map(lambda rn: socio_matrix.M[rn], socio_matrix.relations)
            self.M[dn] = self.__auto_distance(relation_matrices, df, actors)

    def get(self, s, r, dname=None, **kwd):
        """calls :py:func:`Matrix._get` on all lists of sociomatrices (one for distance function)"""

        V = self.socio_matrix.V
        if dname:
            return self._get(s, r, self.M[dname], V, **kwd)
        return reduce(concat,
                map(lambda dn: self._get(s, r, self.M[dn], V, **kwd), self.distances))
        #map(lambda (k,v): self._get(s, r, v, V, **kwd),
        #            self.M.iteritems()))

    def __auto_distance(self, M, metric_f, actors):
        """perform positional analysis on the given sociomatrices

        compute a distance function between all pairs of
        matrices in M

        :param list(numpy.array) M: the list of sociomatrices
        :param callable metric_f: a valid metric defined in this class
        :param list actors: draw actor pairs from this list, but compute distances over the whole network
        :return: list(numpy.array) of distance matrices.
            same length as M
        """

        for m in M[1:]:
            if m.shape != M[0].shape:
                raise ValueError("incompatible sociomatrix sizes: %s" % str(map(lambda m: str(m.shape), M)))

        DM = []
        V = self.socio_matrix.V
        for m in M:
            sq_dmat = np.zeros((m.shape[0], m.shape[0]), np.float32)
            for a1, a2 in combinations(V.keys(), 2):
                i, j = V[a1], V[a2]
                sq_dmat[i, j] = metric_f(i, j, m)
                sq_dmat[j, i] = sq_dmat[i, j]
            DM.append(sq_dmat)
        return DM


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
                log.error("Runtime Warning: i=%d, j=%d" % (i,j))
                pdb.set_trace()
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

    row = np.dot(m[i,:] - m[j,:], m[i,:] - m[j,:])
    row -= (m[i,i] - m[j,i])*(m[i,i] - m[j,i]) # k != i
    row -= (m[i,j] - m[j,j])*(m[i,j] - m[j,j]) # k != j

    col = np.dot(m[:,i] - m[:,j], m[:,i] - m[:,j])
    col -= (m[i,i] - m[i,j])*(m[i,i] - m[i,j]) # k != i
    col -= (m[j,i] - m[j,j])*(m[j,i] - m[j,j]) # k != j

    d = row + col
    if d < 0:
        log.error("bad value entering pdb ...")
        pdb.set_trace()
    return np.sqrt( d )

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
    row_mean_i = (np.sum(m[i,:]) - m[i,i]) / N
    row_mean_j = (np.sum(m[j,:]) - m[j,j]) / N
    col_mean_i = (np.sum(m[:,i]) - m[i,i]) / N
    col_mean_j = (np.sum(m[:,j]) - m[j,j]) / N

    col_num = np.dot(m[:,i] - col_mean_i, m[:,j] - col_mean_j)
    col_num -= (m[i,i] - col_mean_i)*(m[i,j] - col_mean_j) # k != i
    col_num -= (m[j,i] - col_mean_i)*(m[j,j] - col_mean_j) # k != j

    row_num = np.dot(m[i,:] - row_mean_i, m[j,:] - row_mean_j)
    row_num -= (m[i,i] - row_mean_i)*(m[j,i] - row_mean_j) # k != i
    row_num -= (m[i,j] - row_mean_i)*(m[j,j] - row_mean_j) # k != j

    den_col_i = np.dot(m[:,i] - col_mean_i, m[:,i] - col_mean_i)
    den_col_i -= (m[i,i] - col_mean_i)*(m[i,i] - col_mean_i) # k != i
    den_col_i -= (m[j,i] - col_mean_i)*(m[j,i] - col_mean_i) # k != j

    den_row_i = np.dot(m[i,:] - row_mean_i, m[i,:] - row_mean_i)
    den_row_i -= (m[i,i] - row_mean_i)*(m[i,i] - row_mean_i) # k != i
    den_row_i -= (m[i,j] - row_mean_i)*(m[i,j] - row_mean_i) # k != j

    den_col_j = np.dot(m[:,j] - col_mean_j, m[:,j] - col_mean_j)
    den_col_j -= (m[i,j] - col_mean_j)*(m[i,j] - col_mean_j) # k != i
    den_col_j -= (m[j,j] - col_mean_j)*(m[j,j] - col_mean_j) # k != j

    den_row_j = np.dot(m[j,:] - row_mean_j, m[j,:] - row_mean_j)
    den_row_j -= (m[j,i] - row_mean_j)*(m[j,i] - row_mean_j) # k != i
    den_row_j -= (m[j,j] - row_mean_j)*(m[j,j] - row_mean_j) # k != j

    num = col_num + row_num
    den = np.sqrt(den_col_i + den_row_i) * np.sqrt(den_col_j + den_row_j)

    if den:
        result = num / den
        if abs(result) > 1.001:
            log.error("bad correlation value(%f)! entering pdb ..." % result)
            pdb.set_trace()
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

    dot_row = np.dot(m[i,:], m[j,:]) - (m[i,i]*m[j,i]) - (m[i,j]*m[j,j])
    dot_col = np.dot(m[:,i], m[:,j]) - (m[i,i]*m[i,j]) - (m[j,i]*m[j,j])

    norm_row_i = np.dot(m[i,:], m[i,:]) - (m[i,i]*m[i,i]) - (m[i,j]*m[i,j])
    norm_col_i = np.dot(m[:,i], m[:,i]) - (m[i,i]*m[i,i]) - (m[j,i]*m[j,i])
    norm_i = stabilize( norm_row_i + norm_col_i )

    norm_row_j = np.dot(m[j,:], m[j,:]) - (m[j,i]*m[j,i]) - (m[j,j]*m[j,j])
    norm_col_j = np.dot(m[:,j], m[:,j]) - (m[i,j]*m[i,j]) - (m[j,j]*m[j,j])
    norm_j = stabilize( norm_row_j + norm_col_j )

    if norm_i and norm_j:
        result = (dot_row + dot_col) / ( np.sqrt( norm_i ) * np.sqrt( norm_j ) )
        if abs(result) > 1.001:
            log.error("bad cosine value(%f)! entering pdb ..." % result)
            pdb.set_trace()
        return result
    else:
        return np.nan


def floyd(M, weighted=True):
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
        M[ M > 0 ] = 1

    log.error("skipping ...")
    #for k in range(n):
    #    M = np.minimum(M, np.add.outer(M[:,k],M[k,:]))
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

    def I(v):
        if v > 0:
            return 1
        return 0

    nm = set(nA.keys()) & set(nB.keys())
    nmi = dict( map(lambda t: (t[1], t[0]), enumerate(nm)) )

    va = np.zeros( (len(nm), len(nm)), np.int8 )
    vb = np.zeros( (len(nm), len(nm)), np.int8 )

    log.error("Skipping ... ")
    #for r, c in product(nm, nm):
    #    i,j = nmi[r], nmi[c]
    #    va[i,j] = I( A[nA[r], nA[c]] )
    #    vb[i,j] = I( B[nB[r], nB[c]] )
    #result = np.dot(va, vb)
    #return (result, nmi)
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
    :returns: :math:`\sum_j T'_{i,j}` where *T'* is the indicator
        matrix of *T*
    """

    if not (i in V):
        return np.nan
    return len( filter(lambda v: v>0, T[V[i],:].flat) )


def bonacich_centrality(M):
    """compute the Bonacich centrality on M

    this is implemented as in
    https://facwiki.cs.byu.edu/DML/index.php/Eigenvector_Centrality
    and
    http://www-personal.umich.edu/~mejn/papers/palgrave.pdf (page 5) """

    log.error("skipping...")
    return  np.zeros( (M.shape[0],), dtype=np.int8 )
    try:
        L,V = np.linalg.eig(M)
    except np.linalg.linalg.LinAlgError:
        log.critical("cannot compute the Bonacich Centrality: eig did not converge")
        evec = np.zeros( (M.shape[0],), dtype=np.int8 )
        eval = 0
    else:
        evec = V[:, np.argmax(L)]
        eval = np.max(V)
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

        :param list dist_names: list of names of distances (used to refernece distance functions)

        :param list rel_names: orderedDict of relations and attributes of the type relation --> attribute (used to refernece relation functions)

        :param dict weights: dictionary of the type attribute_value --> weight
    """

    def __init__(self, year, adata, bdata, SM, BM, TS, dist_names, rel_names):
        self.header = self.__set_header(rel_names, dist_names)

        #affiliations, blacklist, artist
        self.A, self.B = adata, bdata
        self.Y = year
        self.actors = SM.V.keys()

        self.SM = SM
        self.BM = BM

        self.namings_idx, self.namings = self.B.naming(self.actors, self.Y-1)

        self.tstrengths = TS
        self.V = self.tstrengths[0][0]
        assert set(self.V.keys()) == set(self.SM.V.keys())
        self.tstrengths = self.tstrengths[1]
        log.info("reporting on %d actors ..." % len(self.actors))
        log.info("\tbonacich centrality")
        self.boncent = bonacich_centrality(self.tstrengths[0])
        log.info("\tweighted shortest paths...")
        self.weighted_shpaths = floyd(self.tstrengths[0], weighted=True)
        log.info("\tshortest paths...")
        self.shpaths = floyd(self.tstrengths[0], weighted=False)

        log.info("\ttriadic closure (1 of 4) ...")
        self.tr1, self.tr1_idx = closure(self.tstrengths[0], self.V, self.namings, self.namings_idx)
        log.info("\ttriadic closure (2 of 4) ...")
        self.tr2, self.tr2_idx = closure(self.namings, self.namings_idx, self.tstrengths[0], self.V)
        log.info("\ttriadic closure (3 of 4) ...")
        self.tr3, self.tr3_idx = closure(self.namings, self.namings_idx, self.namings, self.namings_idx)
        log.info("\ttriadic closure (4 of 4) ...")
        self.tr4, self.tr4_idx = closure(self.tstrengths[0], self.V, self.tstrengths[0], self.V)

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
            if tval in (float, np.float32):
                return "%.4f" % val
            elif tval in (int, np.int64, np.int8):
                return "%d" % val
            elif tval in (np.bool_, bool):
                return (val and '1' or '0')
            elif tval != str:
                log.error("unexpected type, entering pdb ...")
                pdb.set_trace()
            return val

        def bonCentral(a):
            if a in self.V:
                return self.boncent[self.V[a]]
            return 0

        resrow = ((s, r, Y, t, ((s,r,Y) in self.B.data)) +
                #self.SM.get(s, r) +
                #self.BM.get(s, r, None) +
                #Matrix._get(s, -1, [self.namings], self.namings_idx) + Matrix._get(-1, r, [self.namings], self.namings_idx) +
                #self.B.reciprocity(s, r, self.Y) +
                #Matrix._get(s, r, self.tstrengths, self.V) + ##might be empty since dist_lst wants both s,r in V.keys()
                #(self.A.corr(s, r), self.A.sim(s, r)) +
                #Matrix._get(s, r, [self.weighted_shpaths, self.shpaths], self.V) +
                #Matrix._get(s, r, [self.tr1], self.tr1_idx) +
                #Matrix._get(s, r, [self.tr2], self.tr2_idx) +
                #Matrix._get(s, r, [self.tr3], self.tr3_idx) +
                #Matrix._get(s, r, [self.tr4], self.tr4_idx) +
                #(receiver_card,)+
                #(centrality(self.tstrengths[0], self.V, s), centrality(self.tstrengths[0].T, self.V, r),
                #    bonCentral(s), bonCentral(r))+
                #(self.A.corr_comm(s), self.A.corr_comm(r), self.A.sums(s), self.A.sums(r)) +
                ()
                )
        assert len(resrow) == len(self.header), "%d != %d" % (len(resrow), len(self.header))
        return map(format_val, resrow)

    def __set_header(self, rel_names, dist_names):
        __hdrel = ['combined'] + rel_names.keys()
        header = ['sender', 'receiver', 'year', 'trial', 'naming']
        return header
        header += map(lambda s: "similarity_%s"%s, __hdrel)
        for dn in dist_names:
            header += map(lambda s: "%s_%s"%(dn,s), __hdrel)
        header += (["prior_naming_sender", "prior_naming_receiver"] +
                ["reciprocity", "reciprodicy_clock"] +
                ["tie_strength_sender_den", "tie_strength_receiver_den", "tie_strength_intersect"] +
                ["affiliation_corr", "affiliation_sim"] +
                ["weighted_shortest_path", "shortest_path"] +
                ["tr1", "tr2", "tr3", "tr4"] +
                ["receiver_card"] +
                ["centrality_s", "centrality_r", "bonacich_centrality_s", "bonacich_centrality_r"] +
                ["comm_corr_s", "comm_corr_r", "summ_s", "summ_r"])
        return header
