"""
import data to the out database
"""

import logging
import os
import sys
import argparse
import sqlite3

import pandas as pd

from ghw import pklSave, pklLoad
from ghw.socio_matrix import SocioMatrix

log = logging.getLogger(__name__)


def main(artist_data, year, relation, output):
    matrix = SocioMatrix.from_db(artist_data, relation, year)
    log.info("dumping ...")
    if os.path.splitext(output)[1] == ".pkl":
        pklSave(output, matrix)
    elif os.path.splitext(output)[1] == ".csv":
        pd.DataFrame(list(matrix.serialize())).to_csv(output, index=False)


def import_smat(input_path, outdb):
    year, relation = SocioMatrix.parse_fname(input_path)
    log.info("sm: %s - %s" % (year, relation))
    log.info("\tinitializing db (%s) ...", input_path)
    with sqlite3.connect(outdb) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS smat(
                i INTEGER,
                j INTEGER,
                year INTEGER,
                relation TEXT,
                value REAL,
                FOREIGN KEY(i) REFERENCES artist(castid),
                FOREIGN KEY(j) REFERENCES artist(castid)
               )""")

    log.info("\tloading ...")
    smat = pklLoad(input_path)
    log.info("\timportint to db ...")
    with sqlite3.connect(outdb) as conn:
        conn.executescript("PRAGMA foreign_keys = ON;")
        conn.executemany("INSERT INTO smat VALUES(?,?,?,?,?)",
                         smat.serialize())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=str, nargs="+", help='data to import')
    parser.add_argument('output', type=str, help='Output file')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    for i, inp in enumerate(args.inputs):
        log.info("processing %s, %d of %d", inp, i + 1, len(args.inputs))
        if os.path.basename(inp).startswith("sm_"):
            import_smat(inp, args.output)
