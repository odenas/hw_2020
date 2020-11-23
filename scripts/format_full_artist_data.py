
import argparse
import logging

import pandas as pd

log = logging.getLogger()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Format the artist data.')
    parser.add_argument('infile', type=str, help='Unformatted artist data')
    parser.add_argument('--outfile', type=str, default='/dev/stdout', help='Output file')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

    cols = ["name", "role", "year", "nomination",
            "castid", "i", "genre", "title", "champ",
            "production_house"]
    df = pd.read_csv(args.infile)
    df[cols].to_csv(args.outfile, index=False)
