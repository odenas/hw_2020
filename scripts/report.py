#!/usr/bin/env python

import logging

import argparse
from giacomo.hollywood import pklLoad
from giacomo.hollywood.blacklist_data import BlacklistData
from giacomo.hollywood.affiliations import Affiliations
from giacomo.hollywood.weights import Weight
from giacomo.hollywood.report import Report


log = logging.getLogger(__name__)
logging.getLogger("giacomo.hollywood.bmat").setLevel(logging.DEBUG)


def main(report, sep):
    receiver_card = set()
    log.info("dumping ...")
    print(sep.join(report.header))
    for ((s, y), t) in filter(lambda _t: _t[0][1] == Y, B.trials.items()):
        for r in SM.V.keys():
            if s == r:
                continue
            _out_row = report.get(s, r, Y, t, len(receiver_card))
            print(sep.join(_out_row))
            receiver_card |= set([r])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Hollywood Blacklisting analysis.')
    parser.add_argument('network', nargs=3, type=str,
                        help='SocioMatrix, BlockMatrix, TieStrengths in that order.')
    parser.add_argument('blacklist', type=str, help='Blacklist data file')
    parser.add_argument('-w', '--weights', nargs=2, default=[None, None],
                        help='Genre and role weight files.')
    parser.add_argument('-s', '--sep', type=str, nargs='?', default=', ', help='Record separator')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

    W = Weight(*args.weights)
    Y = int(args.network[0].split("_")[1])

    log.info("loading socio matrix")
    SM = pklLoad(args.network[0])
    log.info("loading block matrix")
    BM = pklLoad(args.network[1])
    log.info("loading tie strengths")
    TS = pklLoad(args.network[2])

    A = Affiliations()
    B = BlacklistData(args.blacklist)

    main(Report(Y, A, B, SM, BM, TS, BM.distances, SM.relations), args.sep)
