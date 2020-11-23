"""
Build and dump tie strengths
"""

import logging
import os

import argparse
from ghw import pklLoad, pklSave
from ghw.artist_data import ArtistInfoData
from ghw.weights import Weight

log = logging.getLogger(__name__)

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

    # TODO this is too terse
    sim_functions = [lambda a, b, w: len(a), lambda a, b, w: len(b), lambda a, b, w: len(a & b)]
    tstrengths = list(zip(*map(lambda f: D.adj_matrix(Y, "film", actors, weights, similarity_function=f, lag=args.lag),
                          sim_functions)))
    # pickle.dump(tstrengths, open(args.output, 'w'))
    pklSave(args.output, tstrengths)
