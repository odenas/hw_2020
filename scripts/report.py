#!/usr/bin/env python

import logging

import argparse
import pandas as pd
import sqlite3

from ghw import pklLoad
from ghw.report import Report


log = logging.getLogger(__name__)


def row_iter(report, year, receivers):
    df = pd.read_sql_query(
        f"""select sender, year, trial, receiver
         from trials
         where year={year} and sender != receiver
         """,
        sqlite3.connect('data/input/adata.db')
    )

    receiver_card = set()
    for row in df.itertuples():
        for r in receivers:
            _out_row = report.get(row.sender, r, row.year, row.trial, len(receiver_card))
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
