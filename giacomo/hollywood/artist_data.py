from collections import namedtuple
import csv
import logging
from operator import attrgetter, concat
from functools import reduce

import numpy as np

from .matrix import similarity_function

log = logging.getLogger()


artist_fields = 'artist role year nominated castid i genre film champ prod_house'


class ArtistInfoRow(namedtuple('ArtistInfoRow', artist_fields)):
    """Features of an actor. The input file
    has many features, but the one maintained here are

    artist: the name of the artist (string)
    role: the roles of the artist (tuple of ints)
    year: year of the performance (int)
    nominated: whether the performance had a nomination (bool)
    castid: id of the cast (int)
    i: an id???  no idea what this is (int)
    genre: genres of the performance (tuple of strings)
    film: title of the performance (string)
    champ: whether this is a champ movie (bool)
    production_house: production house (string)
    """

    __slots__ = ()

    @classmethod
    def _factory(cls, i):
        "Factory base on red_artist_data.csv format"
        (name, roles, year, nom, castid, _i, genres, film, champ, prn) = i
        return cls._make([name,
                         tuple(map(int, roles.split(';'))),
                         int(year),
                         (nom == 'True'),
                         int(castid),
                         _i,
                         tuple(genres.split(';')),
                         film,
                         (champ == 'True'),
                         prn])

    def _csvtup(self):
        t = (self.artist,
             ";".join(map(str, self.role)),
             self.year,
             (self.nominated and "True" or "False"),
             self.castid,
             self.i,
             ";".join(self.genre),
             self.film,
             (self.champ and "True" or "False"),
             self.prod_house)
        return tuple(map(lambda v: (type(v) == int and str(v) or v), t))


class ArtistInfoData(object):
    """loads and maintains a collection of
    :py:class:`ArtistInfoRow` instances. An instance has

    data
        a dictionary of the type
        data[:py:class:`ArtistInfoRow`.castid] = [:py:class:`ArtistInfoRow`].
        In other words: (artist_id) --> list of performances of this actor
    sim_attributes
        a tuple of attributes (of :py:class:`ArtistInfoRow`)
        that can be used as relations.
    """

    def __init__(self, fname, select=None):
        reader = csv.reader(open(fname, 'rt', encoding='latin1'))
        self.data = {}
        invalid = 0
        for i, r in enumerate(reader):
            if (select is None) or (int(r[4]) in select):
                try:
                    a = ArtistInfoRow._factory(r)
                    self.data.setdefault(a.castid, []).append(a)
                except ValueError:
                    invalid += 1
        log.info("load %d actors discarded %d ..." % (len(self.data), invalid))
        self.sim_attributes = ("role", "genre",
                               "prod_house", "film",
                               "year", "nominated", "champ")

    def adj_matrix(self, year, attr, actors, weights,
                   similarity_function=similarity_function, lag=1000, progress=False):
        """compute the adjacency matrix of a relation on
        the given actors.

        The method will remove all artists that had not have a
        career until now.

        :param int year: the features of actors after or on
            this year will be filtered out
        :param string attr: defines the relation. this is
            an attribute of :py:class:`ArtistInfoRow`
        :param list actors: the matrix will be based on these
            actors
        :param callable similarity_function: function defined
            on two sets or integers giving the similarity.
        :param dict weights: has the weight for each value of the relation domain
        :param int lag: nr. of years back for which to
            consider records. So records will be in the
            period (year-lag, year) both excluded
        :return: a tuple (list of actor_ids, numpy matrix)
        """

        # a projection of self.data, s.t.
        # actors have had a career before this year
        log.info("filtering out career-less actors ...")
        data = {}
        for act in actors:
            career = list(filter(lambda r: (year - lag) < r.year < year, self.data.get(act, [])))
            if career:
                data[act] = career
            else:
                log.info("dropping actor %s", act)
        log.info("indexing (%d) actors ..." % len(data))
        # actor indexes in the matrix
        V = dict(map(lambda i_a: (i_a[1], i_a[0]),
                 enumerate(filter(lambda a: a in data, actors))))

        matrix = np.zeros((len(V), len(V)), dtype=np.float32)
        pbar = None
        prg = 0
        for a1 in V.keys():
            op1 = self.__selector(data[a1], attr)
            i = V[a1]
            for a2 in V.keys():
                j = V[a2]
                op2 = self.__selector(data[a2], attr)
                matrix[i, j] = similarity_function(op1, op2, weights)
            if progress:
                prg += 1
                pbar.update(prg)
        return V, matrix

    def __selector(self, ainfol, sname):
        """selects attributes of a set of artists and packs
        them into an int or a set. this value can be feeded as
        an operant to :py:func:`similarity_function`. Note that
        this is a private method and should be called by
        adj_matrix

        :param iterable ainfol: list of artists (instances of
            :py:class:`ArtistInfoRow`
        :param string sname: attribute of
            :py:class:`ArtistInfoRow` to select
        :returns: set or int
        """

        if not ainfol:
            raise ValueError("empty list of artists")

        ainfo_vals = map(attrgetter(sname), ainfol)
        assert ainfo_vals, "ainfol was not empty, but this is"

        if sname in ("role", "genre"):
            return set(reduce(concat, ainfo_vals, ()))
        elif sname in ("prod_house", "film"):
            return set(ainfo_vals)
        elif sname in ("year",):
            return len(set(ainfo_vals))
        elif sname in ("nominated", "champ"):
            return len(filter(lambda p: p, ainfo_vals))
        else:
            assert not (sname in self.sim_attributes)
            raise ValueError("cannot handle %s (%r)" % (sname, type(sname)))
