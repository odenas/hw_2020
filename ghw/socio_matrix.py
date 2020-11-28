
from dataclasses import dataclass
import logging
from functools import reduce
from operator import concat

import numpy as np
from tqdm import tqdm

from .db import Db


log = logging.getLogger(__name__)


@dataclass(eq=False)
class SocioMatrix:
    relation: str
    year: int
    matrix: np.array
    artists: dict
    lag: int = 1000

    @property
    def pkl_fname(self):
        return f"sm_{self.year}_{self.relation}.pkl"

    @classmethod
    def parse_fname(self, path):
        fname, ext = os.path.splitext(os.path.basename(path))
        p, y, r = fname.split("_")
        return int(y), r

    @classmethod
    def sim_functions(cls, relation):
        def sim_set(s1, s2):
            if s1:
                return len(s1 & s2) / len(s1)
            return 0.0

        def sim_scalar(i, j):
            if i == j == 0:
                return 0
            return 1 - np.abs(i - j) / max(i, j)

        if relation in {'champ', 'nominated', 'year'}:
            return sim_scalar
        elif relation in {'roles', 'genre', 'house', 'film'}:
            return sim_set
        elif relation == 'ts1':
            return lambda s1, s2: len(s1)
        elif relation == 'ts2':
            return lambda s1, s2: len(s2)
        elif relation == 'ts3':
            return lambda s1, s2: len(s1 & s2)
        raise ValueError(f"unknown relation {relation}")

    @classmethod
    def selector_functions(cls, relation):
        return {
            'champ': len,
            'nominated': sum,
            'roles': lambda tl: set(reduce(concat, tl, ())),
            'genre': lambda tl: set(reduce(concat, tl, ())),
            'house': set,
            'film': set, 'ts1': set, 'ts2': set, 'ts3': set,
            'year': lambda tl: len(set(tl))}[relation]

    @classmethod
    def _get_df(cls, dbpath, relation, year):
        # tie strengths are a special case for film
        if relation in ('ts1', 'ts2', 'ts3', 'film'):
            relation = 'film'
            query = f"""select
            artist, perf.title as relation
            from perf join title on title.id = perf.title
            where year > {year - cls.lag} and year < {year}
            """
        else:
            query = f"""select
            artist, {relation} as relation
            from perf join title on title.id = perf.title
            where year > {year - cls.lag} and year < {year}
            """

        df = Db.from_path(dbpath).query_as_df(query)
        if relation in ('roles', 'genre'):
            df = (df.assign(relation=lambda x: x.relation.apply(Db.decode)))
        return df

    @classmethod
    def from_db(cls, dbpath, relation, year):
        df = cls._get_df(dbpath, relation, year)
        log.info("%d rows", df.shape[0])

        sel_f = cls.selector_functions(relation)
        ddf = (df.groupby('artist')
               .aggregate(sel_f)
               .reset_index().reset_index())
        log.info("%d actors", ddf.shape[0])
        sim_f = cls.sim_functions(relation)
        m = cls._adj_matrix(ddf.relation, sim_f)
        v = dict(t for _, t in ddf[['artist', 'index']].iterrows())
        return cls(relation, year, m, v)

    @classmethod
    def _adj_matrix(cls, array, sim_f):

        m = np.zeros((array.shape[0], array.shape[0]))
        for i, row in tqdm(enumerate(array), total=m.shape[0]):
            for j, col in enumerate(array):
                if i == j:
                    continue
                m[i, j] = sim_f(row, col)
        return m
