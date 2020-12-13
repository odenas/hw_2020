"""
Build and dump a socio matrix
"""

import logging
import sys
import argparse

from ghw import pklSave
from ghw.socio_matrix import SocioMatrix

log = logging.getLogger(__name__)


def main(artist_data, year, relation, output):
    log.info("%s - %s" % (year, relation))
    matrix = SocioMatrix.from_db(artist_data, relation, year)
    pklSave(output, matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('adata', type=str, help='Artist database')
    parser.add_argument('output', type=str, help='Output file')
    parser.add_argument('--lag', type=int, choices=(5, 1000), default=1000,
                        help='Lag parameter')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    year, relation = SocioMatrix.parse_fname(args.output)
    sys.exit(main(args.adata, year, relation, args.output))
