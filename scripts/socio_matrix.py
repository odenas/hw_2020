"""
Build and dump one or more socio matrices
"""

import logging
import os
import sys
from collections import OrderedDict

import argparse

from giacomo.hollywood import pklSave, rel_names
from giacomo.hollywood.artist_data import ArtistInfoData
from giacomo.hollywood.weights import Weight
from giacomo.hollywood.socio_matrix import SocioMatrix


log = logging.getLogger(__name__)
logging.getLogger("giacomo.hollywood.bmat").setLevel(logging.INFO)


def main(artist_data, weights, year, relation, output):
    log.info("%s - %s" % (Y, R))
    receivers = artist_data.data.keys()
    matrix = SocioMatrix(artist_data, int(year), receivers,
                         OrderedDict([(relation, rel_names[R])]), weights, lag=args.lag)
    pklSave(output, matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artists', type=str, help='Artist data file')
    parser.add_argument('output', type=str,
                        help='Output file (of the type sm_year_relation.pkl')
    parser.add_argument('--lag', type=int, choices=(5, 1000), default=1000,
                        help='Lag parameter')
    parser.add_argument('-w', '--weights', nargs=2, default=[None, None],
                        help='Genre and role weight files.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    Y, R = os.path.splitext(args.output)[0].split("_")[1:3]
    if R.endswith(".pkl"):
        R = R.split(".")[0]

    sys.exit(main(ArtistInfoData(args.artists), Weight(*args.weights), Y, R, args.output))
