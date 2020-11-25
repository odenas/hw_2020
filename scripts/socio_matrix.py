"""
Build and dump one or more socio matrices
"""

import logging
import os
import sys
from collections import OrderedDict

import argparse

from ghw import pklSave, rel_names
from ghw.artist_data import ArtistInfoData
from ghw.socio_matrix import SocioMatrix


log = logging.getLogger(__name__)
logging.getLogger("ghw.bmat").setLevel(logging.INFO)


def main(artist_data, year, relation, output):
    log.info("%s - %s" % (Y, R))
    receivers = artist_data.data.keys()
    matrix = SocioMatrix(artist_data, int(year), receivers,
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

    sys.exit(main(ArtistInfoData(args.artists), Y, R, args.output))
