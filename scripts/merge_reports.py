
import argparse
from collections import OrderedDict
import logging
import os.path
import sys

import pandas as pd

from ghw import pklLoad

log = logging.getLogger()


def rel_mapping(in_dir, skip):
    col_map = OrderedDict([
        ('sm_1951_nominated', f'{in_dir}/sm_1951_oscarnom.pkl'),
        ('sm_1951_year', f'{in_dir}/sm_1951_experience.pkl'),
        ('sm_1951_champ', f'{in_dir}/sm_1951_champ.pkl'),
        ('sm_1951_genre', f'{in_dir}/sm_1951_genre.pkl'),
        ('sm_1951_house', f'{in_dir}/sm_1951_prodhouse.pkl'),
        ('sm_1951_role', f'{in_dir}/sm_1951_role.pkl'),
        ('bm_1951_nominated_corr.mtx', f'{in_dir}/bm_1951_oscarnom_correlation.pkl'),
        ('bm_1951_nominated_cosine.mtx', f'{in_dir}/bm_1951_oscarnom_cosine.pkl'),
        ('bm_1951_nominated_euclidean.mtx', f'{in_dir}/bm_1951_oscarnom_euclidean.pkl'),

        # not sure why year has no .mtx suffix
        ('bm_1951_year_corr', f'{in_dir}/bm_1951_experience_correlation.pkl'),
        ('bm_1951_year_cosine', f'{in_dir}/bm_1951_experience_cosine.pkl'),
        ('bm_1951_year_euclidean', f'{in_dir}/bm_1951_experience_euclidean.pkl'),

        ('bm_1951_champ_corr.mtx', f'{in_dir}/bm_1951_champ_correlation.pkl'),
        ('bm_1951_champ_cosine.mtx', f'{in_dir}/bm_1951_champ_cosine.pkl'),
        ('bm_1951_champ_euclidean.mtx', f'{in_dir}/bm_1951_champ_euclidean.pkl'),

        ('bm_1951_genre_corr.mtx', f'{in_dir}/bm_1951_genre_correlation.pkl'),
        ('bm_1951_genre_cosine.mtx', f'{in_dir}/bm_1951_genre_cosine.pkl'),
        ('bm_1951_genre_euclidean.mtx', f'{in_dir}/bm_1951_genre_euclidean.pkl'),

        ('bm_1951_house_corr.mtx', f'{in_dir}/bm_1951_prodhouse_correlation.pkl'),
        ('bm_1951_house_cosine.mtx', f'{in_dir}/bm_1951_prodhouse_cosine.pkl'),
        ('bm_1951_house_euclidean.mtx', f'{in_dir}/bm_1951_prodhouse_euclidean.pkl'),

        ('bm_1951_role_corr.mtx', f'{in_dir}/bm_1951_role_correlation.pkl'),
        ('bm_1951_role_cosine.mtx', f'{in_dir}/bm_1951_role_cosine.pkl'),
        ('bm_1951_role_euclidean.mtx', f'{in_dir}/bm_1951_role_euclidean.pkl'),
    ])
    log.info("filter out non-existent paths ...")
    filter_col_map = {}
    for colname, path in col_map.items():
        if os.path.exists(path):
            filter_col_map[colname] = path
        else:
            msg = f"expected report {path} missing."
            if skip:
                log.warning(msg)
            else:
                raise ValueError(msg)
    return filter_col_map


def array_get(s_arr, r_arr, dstr):
    return [dstr.get(s, r)[1] for s, r in zip(s_arr, r_arr)]


def load_report(path):
    _df_cols = [
        'sender', 'receiver', 'year', 'trial', 'naming',
        'prior_naming_sender', 'prior_naming_receiver',
        'reciprocity', 'reciprodicy_clock',
        'tie_strength_sender_den', 'tie_strength_receiver_den', 'tie_strength_intersect',
        'affiliation_corr', 'affiliation_sim',
        'receiver_card',
        'centrality_s', 'centrality_r',
        'comm_corr_s', 'comm_corr_r', 'summ_s', 'summ_r',
    ]
    df = pd.read_csv(path, delimiter=", ", na_values=[' '], engine='python')[_df_cols]
    return df.rename(columns={'tie_strength_sender_den': 'ts_send',
                              'tie_strength_receiver_den': 'ts_receive',
                              'tie_strength_intersect': 'ts_interesction'
                              })


def main(in_dir, out_name, skip):
    from glob import glob
    res = []
    for p in sorted(glob(os.path.join(in_dir, 'rp_195*.csv'))):
        log.info("processing %s", p)
        df = load_report(p)

        res.append(
            df.assign(**{k: lambda x: array_get(x.sender, x.receiver, pklLoad(v))
                         for k, v in rel_mapping(in_dir, skip).items()})
        )
    log.info("dumping to %s", out_name)
    pd.concat(res).to_csv(out_name, index=False, float_format="%.4f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge all reports into one that rules them all.')
    parser.add_argument('indir', type=str, help='Directory containing report files')
    parser.add_argument('output', type=str, help='Output file')
    parser.add_argument('--skip_missing', action='store_true', default=False,
                        help='Skip missing reports.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    sys.exit(main(args.indir, args.output, args.skip_missing))
