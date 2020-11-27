
from dataclasses import dataclass
import logging
from functools import reduce
from operator import concat
import sqlite3

import numpy as np
import pandas as pd
from tqdm import tqdm


log = logging.getLogger(__name__)


@dataclass(eq=False)
class SocioMatrix:
    relation: str
    year: int
    matrix: np.array
    artists: dict
    lag: int = 1000

    @classmethod
    def sim_functions(cls, relation):
        def sim_set(s1, s2):
            if s1:
                return len(s1 & s2) / len(s1)
            return 0.0

        def sim_scalar(i, j):
            return 1 - np.abs(i - j) / max(i, j)

        if relation in {'champ', 'nominated', 'year'}:
            return sim_scalar
        elif relation in {'role', 'genre', 'house', 'film'}:
            return sim_set
        raise ValueError(f"unknown relation {relation}")

    @classmethod
    def selector_functions(cls, relation):
        return {
            'champ': len,
            'nominated': sum,
            'role': lambda tl: set(reduce(concat, tl, ())),
            'genre': lambda tl: set(reduce(concat, tl, ())),
            'house': set,
            'film': set,
            'year': lambda tl: len(set(tl))}[relation]

    @classmethod
    def _get_df(cls, dbpath, relation, year):
        def unpack_dash_val(val):
            try:
                return tuple(map(int, val.split('-')))
            except ValueError:
                return ()

        query = f"""select
        artist, {relation} as relation
        from perf join title on title.id = perf.title
        where year > {year - cls.lag} and year < {year}
        """

        con = sqlite3.connect(dbpath)
        if relation in ('role', 'genre'):
            df = (pd.read_sql_query(query, con)
                  .assign(relation=lambda x: x.relation.apply(unpack_dash_val)))
        else:
            df = pd.read_sql_query(query, con)
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
        v, m = cls.adj_matrix(ddf, relation)
        return cls(relation, year, m, v)

    @classmethod
    def adj_matrix(cls, ddf, relation):
        sim_f = cls.sim_functions(relation)

        m = np.zeros((ddf.shape[0], ddf.shape[0]))

        cprod = ddf.assign(key=1).merge(ddf.assign(key=1), on='key').drop('key', 1)
        for t in tqdm(cprod.itertuples()):
            m[t.index_x, t.index_y] = sim_f(t.relation_x, t.relation_y)
        v = dict(t for _, t in ddf[['artist', 'index']].iterrows())
        return v, m
