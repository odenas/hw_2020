import argparse
import logging
import sys

import numpy as np

from ghw import pklLoad, pklSave
from ghw.block_matrix import BMat, dflist
from ghw.db import Db

log = logging.getLogger(__name__)


def allSenders(dbpath, year):
    return (Db.from_path(dbpath)
            .query_as_df(f"select distinct sender from trials where year={year}")
            .sender.tolist())


def main(socio_matrix, dbpath, distance, year):
    dmat = BMat.dmat(socio_matrix.matrix, dflist[f"{distance}_metric"])
    block_matrix = BMat(socio_matrix, distance, dmat)
    pklSave(args.output, block_matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Hollywood Blacklisting analysis.')
    parser.add_argument('matrix', type=str, help='Socio matrix')
    parser.add_argument('dbpath', type=str, help='Database')
    parser.add_argument('output', type=str, help='Output ')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    year, _, distance = BMat.parse_fname(args.output)
    log.info(f"{distance} - {year}")
    sys.exit(main(pklLoad(args.matrix), args.dbpath, distance, year))
