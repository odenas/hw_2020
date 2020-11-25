
from collections import OrderedDict
import logging

from .matrix import Matrix
from .artist_data import selector_functions


log = logging.getLogger(__name__)


class SocioMatrix(Matrix):
    """matrix defined over the similarity (jaccard) measure. maintains

    M an ordered dict of the type (relation_name) ---> matrix
    V dict of the type (artist) --> index

    to initialize needs:

    data :py:class:`giacomo.hollywood.bmat.ArtistInfoData`
    year int
    actors iterable
    relations a dictionary of the type (relation_name) --> attribute_name
    combined bool
    kwd other options passed by name to ArtistInfoData.adj_matrix
    """

    # TODO: combined should be part of the relations argument
    def __init__(self, data, year, actors, relations, combined=True, **kwd):
        self.relations = relations
        self.M = OrderedDict()
        self.V = None
        log.info("sociomatrix (%d actors) over relations:" % len(actors))
        for relname, relval in relations.items():
            v, m = data.adj_matrix(year, relval, actors, selector_functions[relval], **kwd)
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

        mlist = [self.M['combined']] + list(map(lambda rn: self.M[rn], self.relations))
        return self._get(s, r, mlist, self.V, **kwd)
