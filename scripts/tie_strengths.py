"""
Build and dump tie strengths
"""

import logging
import os

import argparse
from ghw import pklLoad, pklSave
from ghw.artist_data import ArtistInfoData, selector_functions
from ghw.weights import Weight

log = logging.getLogger(__name__)


def length_first(a, b, w):
    return len(a)


def length_second(a, b, w):
    return len(b)


def length_both(a, b, w):
    return len(a & b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artists', type=str, help='Artist data file')
    parser.add_argument('socioMatrix', type=str, help='Socio matrix')
    parser.add_argument('output', type=str, help='Output ')
    parser.add_argument('--lag', type=int, choices=(5, 1000), default=1000,
                        help='Lag parameter')
    parser.add_argument('-w', '--weights', nargs=2, default=[None, None],
                        help='Genre and role weight files.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    D = ArtistInfoData(args.artists)
    W = Weight(*args.weights)
    SM = pklLoad(args.socioMatrix)
    actors = SM.V.keys()

    Y = int(os.path.splitext(args.socioMatrix)[0].split("_")[1])
    weights = W.weights_of("films")

    sel_f = selector_functions["film"]
    adj_matrices = [
        D.adj_matrix(Y, "film", actors, sel_f, similarity_function=length_first, lag=args.lag),
        D.adj_matrix(Y, "film", actors, sel_f, similarity_function=length_second, lag=args.lag),
        D.adj_matrix(Y, "film", actors, sel_f, similarity_function=length_both, lag=args.lag),
    ]
    tstrengths = list(zip(*adj_matrices))
    pklSave(args.output, tstrengths)
