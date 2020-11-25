
__docformat__ = "restructuredtext en"

""" the hollywood project

defines a variable CFG, a ConfigParser instance fetching several data values:
 - trial_list: the list of trials with the order they occurred
 - report_fields: the list of attributes (relations) to be computed and reported as results
"""


import pickle
import logging
import os
import pdb
from configparser import ConfigParser
from collections import OrderedDict
from itertools import product

# common imports
import numpy as np

log = logging.getLogger(__name__)

__this_dir__ = os.path.dirname(__file__)

__conf_path = os.path.join(__this_dir__, "conf.cfg")

# load general settings
log.debug("loading settings from %s ..." % __conf_path)
CFG = ConfigParser()
CFG.read_file(open(__conf_path))

report_fields = CFG.get("GENERAL", "fields").split()
log.debug("%d fields in the report" % len(report_fields))

gray_list = list(map(int, CFG.get("GENERAL", "gray_list").split()))
log.debug("%d artists in the gray list" % len(gray_list))

def __parse_unfriendly(t):
    k, v = t
    an, y, t = v.split("\t")
    return (int(k), an, int(y), float(t))


unfriendly = list(map(__parse_unfriendly, CFG.items("UNFRIENDLY")))
log.debug("%d unfriendly artists ..." % len(unfriendly))


def format_val(val):
    """format a value into a string

    :param object val: the value to format. allowed
        types are: np.infinite, int, np.int, str,
        float, np.float
    :returns: a string
    """

    tval = type(val)

    if not np.isfinite(val):
        return " "
    if tval in (float, np.float64):
        return "%.4f" % val
    elif tval in (int, np.int64):
        return "%d" % val
    elif tval != str:
        log.error("unexpected type, entering pdb ...")
        pdb.set_trace()
    return val


def load_data(ad_fname, bd_fname):
    """wrapper that loads three types of data into the
        corresponding objects.

    :param str ad_fname: path to the artist data file
    :param str bd_fname: path to the blacklist data file
    :returns: instances
        (:py:class:`giacomo.hollywood.bmat.Affiliations`,
        :py:class:`giacomo.hollywood.bmat.ArtistInfoData`,
        :py:class:`giacomo.hollywood.bmat.BlacklistData`)
    """

    from bmat import Affiliations, BlacklistData, ArtistInfoData

    def ld(f, t):
        if f.endswith("pkl"):
            return pickle.load(open(f))
        else:
            return t(f)

    return (Affiliations(),
            ld(ad_fname, ArtistInfoData),
            ld(bd_fname, BlacklistData))


# list of years on which namings happened
naming_years = (1951, 1952, 1953, 1954, 1955)

# header name --> relation_name (as defined in ArtistInfoRow)
rel_names = OrderedDict(
    [("champ",  "champ"),
     ("experience", "year"),
     ("genre", "genre"),
     ("oscarnom", "nominated"),
     ("film", "film"),
     ("prodhouse", "prod_house"),
     ("role", "role")]
)

# measure structural equivalence with this distance metrics
equivalence_metrics = ["euclidean", "correlation", "cosine"]

# header columns that must be always present
base_header = ['sender', 'receiver', 'year', 'trial']


# metric header columns
def metric_header(metrics, combined, similarity):
    attr = rel_names.keys()
    if combined:
        attr = ['combined'] + attr
    # metrics = equivalence_metrics
    # if similarity:
    #    metrics = ['similarity'] + equivalence_metrics

    return map(lambda m_rel: "%s_%s" % m_rel, product(metrics, attr))


def pklSave(path, obj):
    import gzip
    import pickle

    if path.endswith(".gz"):
        fd = gzip.open(path, 'wb')
    else:
        fd = open(path, 'wb')

    pickle.dump(obj, fd)


def pklLoad(path):
    import gzip
    import pickle

    if path.endswith(".gz"):
        fd = gzip.open(path, 'rb')
    else:
        fd = open(path, 'rb')
    return pickle.load(fd)
