"""
Build and dump a socio matrix
"""

import logging
import os
import sys
import argparse

from ghw import pklSave
from ghw.socio_matrix import SocioMatrix

log = logging.getLogger(__name__)


def main(artist_data, year, relation, output):
    log.info("%s - %s" % (year, relation))
    matrix = SocioMatrix.from_db(artist_data, relation, int(year))
    pklSave(output, matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('adata', type=str, help='Artist database')
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
    sys.exit(main(args.adata, Y, R, args.output))
