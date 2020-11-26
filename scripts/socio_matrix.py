"""
Build and dump one or more socio matrices
"""

import logging
import os
import sys
from collections import OrderedDict
import argparse
import sqlite3
from functools import reduce
from operator import concat

from tqdm import tqdm
import numpy as np
import pandas as pd

from ghw import pklSave, rel_names
from ghw.matrix import Matrix
# from ghw.artist_data import ArtistInfoData
# from ghw.socio_matrix import SocioMatrix


log = logging.getLogger(__name__)


class SocioMatrix1(Matrix):
    def __init__(self, db_path, year, relations, combined=True, **kwd):
        self.db_path = db_path
        self.relations = relations
        self.M = OrderedDict()
        self.V = None
        for relname, relval in relations.items():
            v, m = self.genre_adj_matrix(year, 1000)
            self.M[relname] = m
            log.info("\t%s (%d actors)" % (relname, len(v)))
        if not self.V:
            self.V = v
        if set(v.keys()) < set(self.V.keys()):
            self.V = v

        if combined:
            N = float(len(self.M))
            self.M["combined"] = sum(self.M.values()) / N

    def unpack_dash_val(self, val):
        try:
            return tuple(map(int, val.split('-')))
        except ValueError:
            return ()

    def genre_adj_matrix(self, year, lag):
        con = sqlite3.connect(self.db_path)
        with con:
            query = f"""select
            artist.castid, perf.genre
            from perf join artist on perf.artist = artist.castid
                join title on title.id = perf.title
            where year > {year - lag} and year < {year}
            """
            log.info(query)
            df = (pd.read_sql_query(query, con)
                  .assign(genre=lambda x: x.genre.apply(self.unpack_dash_val)))
            log.info("%d rows", df.shape[0])

        ddf = (df.groupby('castid')
               .aggregate(lambda tl: set(reduce(concat, tl, ())))
               .reset_index().reset_index())
        log.info("%d actors", ddf.shape[0])

        def sim_role(s1, s2):
            if s1:
                return len(s1 & s2) / len(s1)
            return 0

        m = np.zeros((ddf.shape[0], ddf.shape[0]))
        cprod = ddf.assign(key=1).merge(ddf.assign(key=1), on='key').drop('key', 1)
        for t in tqdm(cprod.itertuples()):
            m[t.index_x, t.index_y] = sim_role(t.genre_x, t.genre_y)
        v = dict(t for _, t in ddf[['castid', 'index']].iterrows())
        return v, m

    def get(self, s, r, **kwd):
        """calls :py:func:`Matrix._get` on all sociomatrices"""

        mlist = [self.M['combined']] + list(map(lambda rn: self.M[rn], self.relations))
        return self._get(s, r, mlist, self.V, **kwd)


def main(artist_data, year, relation, output):
    log.info("%s - %s" % (Y, R))
    matrix = SocioMatrix1(artist_data, int(year), 
                         OrderedDict([(relation, rel_names[R])]), lag=args.lag)
    pklSave(output, matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artists', type=str, help='Artist data file')
    parser.add_argument('output', type=str,
                        help='Output file (of the type sm_year_relation.pkl')
    parser.add_argument('--lag', type=int, choices=(5, 1000), default=1000,
                        help='Lag parameter')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    Y, R = os.path.splitext(args.output)[0].split("_")[1:3]
    if R.endswith(".pkl"):
        R = R.split(".")[0]

    # sys.exit(main(ArtistInfoData(args.artists), Y, R, args.output))
    sys.exit(main(args.artists, Y, R, args.output))
