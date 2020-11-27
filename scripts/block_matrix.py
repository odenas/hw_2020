import argparse
import logging
import os
import sqlite3
import sys

from ghw import pklLoad, pklSave
from ghw.block_matrix import BlockMatrix

import pandas as pd

log = logging.getLogger(__name__)


def allSenders(dbpath, year):
    query = f"select distinct sender from trials where year={year}"
    log.info(f"{query}")
    return pd.read_sql_query(query, sqlite3.connect(dbpath)).sender.tolist()


def main(socio_matrix, dbpath, distances, year):
    block_matrix = BlockMatrix(socio_matrix, distances, allSenders(dbpath, year))
    pklSave(args.output, block_matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Hollywood Blacklisting analysis.')
    parser.add_argument('matrix', type=str, help='Socio matrix')
    parser.add_argument('dbpath', type=str, help='Database')
    parser.add_argument('output', type=str, help='Output ')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

    distances = list(map(lambda s: (s.split(".")[0]),
                         os.path.splitext(args.output)[0].split("_")[3:4]))
    year = int(os.path.splitext(args.matrix)[0].split("_")[1])
    sys.exit(main(pklLoad(args.matrix), args.dbpath, distances, year))
