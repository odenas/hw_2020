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
    parser.add_argument('cm', type=str, help='Centrality measures file.')
    parser.add_argument('dbpath', type=str, help='Blacklist data file')
    parser.add_argument('output', type=str, help='Output file')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

    year, dmetric, relation = BMat.parse_fname(args.bm)
    sm = pklLoad(args.sm)
    log.info("Loaded sm...")
    bm = pklLoad(args.bm)
    log.info("Loaded bm...")
    ts1 = pklLoad(args.ts[0])
    log.info("Loaded ts1...")
    ts2 = pklLoad(args.ts[1])
    log.info("Loaded ts2...")
    ts3 = pklLoad(args.ts[2])
    log.info("Loaded ts3...")
    cm = pklLoad(args.cm)
    log.info("Loaded cm...")

    report = Report(year, args.dbpath, sm, bm, (ts1, ts2, ts3), cm,
                    [dmetric], relation)
    main(report, args.output)
