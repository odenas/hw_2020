from collections import namedtuple
import csv
import logging
from operator import concat
from functools import reduce

import numpy as np
from tqdm import tqdm

from .matrix import similarity_function

log = logging.getLogger(__name__)


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


def selector_champ(ainfol):
    ainfo_vals = [a.champ for a in ainfol if a.champ]
    return len(ainfo_vals)


def selector_nominated(ainfol):
    ainfo_vals = [a.nominated for a in ainfol if a.nominated]
    return len(ainfo_vals)


def selector_role(ainfol):
    ainfo_vals = [a.role for a in ainfol]
    return set(reduce(concat, ainfo_vals, ()))


def selector_genre(ainfol):
    ainfo_vals = [a.genre for a in ainfol]
    return set(reduce(concat, ainfo_vals, ()))


def selector_prod_house(ainfol):
    ainfo_vals = [a.prod_house for a in ainfol]
    return set(ainfo_vals)


def selector_film(ainfol):
    ainfo_vals = [a.film for a in ainfol]
    return set(ainfo_vals)


def selector_year(ainfol):
    ainfo_vals = [a.year for a in ainfol]
    return len(set(ainfo_vals))


selector_functions = {
    'champ': selector_champ, 'nominated': selector_nominated,
    'role': selector_role, 'genre': selector_genre,
    'prod_house': selector_prod_house, 'film': selector_film,
    'year': selector_year
}


def sim_champ(i, j):
    if i == j:
        return 0
    return 1 - np.abs(i - j) / max(i, j)


def sim_role(s1, s2):
    if s1:
        return len(s1 & s2) / len(s1)
    return 0


sim_functions = {
    'champ': sim_champ, 'nominated': sim_champ,
    'role': sim_role, 'genre': sim_role,
    'prod_house': sim_role, 'film': sim_role,
    'year': sim_champ,
}


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
        log.info("read %d actors discarded %d ..." % (len(self.data), invalid))
        self.sim_attributes = tuple(selector_functions)

    def adj_matrix(self, year, attr, actors, selector, sim_func, lag=1000):
        log.info("filtering out career-less actors ...")
        data = {}
        for act in tqdm(actors):
            career = list(filter(lambda r: (year - lag) < r.year < year, self.data.get(act, [])))
            if career:
                data[act] = career
            else:
                log.debug("dropping actor %s", act)
        log.info("indexing (%d) actors ..." % len(data))
        # actor indexes in the matrix
        V = dict(map(lambda i_a: (i_a[1], i_a[0]),
                 enumerate(filter(lambda a: a in data, actors))))

        matrix = np.zeros((len(V), len(V)), dtype=np.float32)
        for a1 in tqdm(V.keys()):
            op1 = selector(data[a1])
            i = V[a1]
            for a2 in V.keys():
                j = V[a2]
                op2 = selector(data[a2])
                matrix[i, j] = sim_func(op1, op2)
        return V, matrix
