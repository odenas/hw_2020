#!/usr/bin/env python

import logging

import argparse
import pandas as pd

from ghw import pklLoad
from ghw.db import Db
from ghw.report import Report


log = logging.getLogger(__name__)


def row_iter(report, year, receivers):
    db = Db.from_path(report._dbpath)
    df1 = db.query_as_df(f"select sender as s, year, trial from trials where year = {year}")
    df2 = db.query_as_df(f"select castid as s, year, trial from unfriendly where year = {year}")
    trials = dict(((r.s, r.year), r.trial)
                  for r in pd.concat([df1, df2]).itertuples())
    receiver_card = set()
    for sy, t in trials.items():
        for r in receivers:
            if sy[0] == r:
                continue
            _out_row = report.get(sy[0], r, sy[1], t, len(receiver_card))
            yield _out_row
            receiver_card |= set([r])


def main(report, out_file):
    log.info("dumping ...")
    (pd.DataFrame(list(row_iter(report, report.year, SM.artists)), columns=report.header)
     .to_csv(out_file, index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Hollywood Blacklisting analysis.')
    parser.add_argument('sm', type=str, help='Socio Matrix')
    parser.add_argument('bm', type=str, help='Block Matrix')
    parser.add_argument('ts', nargs=3, type=str, help='TieStrengths(1, 2, 3) in that order.')
    parser.add_argument('dbpath', type=str, help='Blacklist data file')
    parser.add_argument('output', type=str, help='Output file')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

    year = int(args.sm.split("_")[1])
    log.info("loading socio matrix")
    SM = pklLoad(args.sm)
    log.info("loading block matrix")
    BM = pklLoad(args.bm)
    log.info("loading tie strengths")
    TS = list(map(pklLoad, args.ts))

    main(Report(year, args.dbpath, SM, BM, TS, BM.distances, SM.relation), args.output)
