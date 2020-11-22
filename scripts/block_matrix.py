
import argparse
import logging
import os
import sys

from giacomo.hollywood import pklLoad, pklSave
from giacomo.hollywood.blacklist_data import BlacklistData
from giacomo.hollywood.block_matrix import BlockMatrix

log = logging.getLogger(__name__)
logging.getLogger("giacomo.hollywood.bmat").setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG)


def allSenders(bl_path, Y):
    if not os.path.isfile(bl_path):
        return None
    B = BlacklistData(bl_path)
    S = []
    for ((s, y), t) in filter(lambda t: t[0][1] == Y, B.trials.items()):
        S.append(s)
    return list(set(S))


def main(socio_matrix, distances, year):
    block_matrix = BlockMatrix(socio_matrix, distances, allSenders(args.blacklist, year))
    pklSave(args.output, block_matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Hollywood Blacklisting analysis.')
    parser.add_argument('matrix', type=str, help='Socio matrix')
    parser.add_argument('blacklist', type=str, default=None, help='Blacklist data')
    parser.add_argument('output', type=str, help='Output ')
    args = parser.parse_args()

    distances = list(map(lambda s: (s.split(".")[0]), os.path.splitext(args.output)[0].split("_")[3:4]))
    year = int(os.path.splitext(args.matrix)[0].split("_")[1])
    sys.exit(main(pklLoad(args.matrix), distances, year))
