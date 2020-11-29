#!/usr/bin/env python

import logging

import argparse
import pandas as pd

from ghw import pklLoad
from ghw.report import Report
from ghw.block_matrix import BMat


log = logging.getLogger(__name__)


def main(report, out_file):
    log.info("dumping ...")
    (pd.DataFrame(list(report.row_iter(report.SM.artists)), columns=report.header)
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

    year, dmetric, relation = BMat.parse_fname(args.bm)
    main(Report(year, args.dbpath, pklLoad(args.sm), pklLoad(args.bm), list(map(pklLoad, args.ts)),
                [dmetric], relation), args.output)
