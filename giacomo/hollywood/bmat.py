
import os, logging
from collections import namedtuple, defaultdict
from operator import itemgetter, attrgetter, concat
import numpy as np
from .. import progressbar

from . import unfriendly, gray_list
from .. import ucsv

log = logging.getLogger(__name__)


def similarity_function(A, B, weights):
    """similarity function defined on sets or numerical values

    computes the similarity value of A and B. note that this
    is not symmetrical

    :param set/int A: first operand
    :param set/int B: second operand
    :returns: :math:`|A \\cap B| / |A|` if A and B are sets.
        :math:`1 - ( |A-B| / max\\{|A|, |B|\\} )` if A and B are ints.
    """

    if type(A) != type(B):
        raise ValueError("A(%r) != B(%r)" % (type(A), type(B)))

    if type(A) == set:
        return sum(map(lambda a: weights(a), (A&B))) / float(sum(map(lambda a: weights(a), A)))
    elif type(A) == int:
        M = float(max(A, B))
        if M:
            return min(A, B) / M
        return 0
    else:
        raise ValueError("unsupported type %r" % type(A))


class Weight( object ):
    """provides weights for genres and roles. Maintains:

    :param dict genre_weights: a map of the type genre --> weight

    :param dict genre_weights: a map of the type genre --> weight

    To initialize provide the .cfg file for genre and roles. each file has weights
    for each year (one year / section). If no files are provided, this will
    implement the identity map (i.e., all attributes have weight = 1).
    """

    def __init__(self, gf, rf):
        if gf or rf:
            raise NotImplementedError('only identity map implemented so far')

    def weights_of(self, attrib):
        return self.one

    @classmethod
    def one(cls, a): return 1


class Similarity(object):
    pseudoprob = 1 / (1 - 0.00001)

    def __init__(self, weights=None):
        if weights:
            self.weights = defaultdict(lambda : self.pseudoprob)
            for k,v in weights.options():
                self.weights[k] = 1 / float(v)
        else:
            self.weights = None

    def __call__(self, A, B):
        if type(A) != type(B):
            raise ValueError("A(%r) != B(%r)" % (type(A), type(B)))
        if type(A) == set:
            if self.weights is None:
                return len(A & B) / float( len(A) )
            return sum(map(lambda a: self.weights[a], (A&B))) / float( sum(map(lambda a: self.weights[a], A)) )
        elif type(A) == int:
            M = float(max(A, B))
            if M:
                # same as: return 1 - (abs(A - B) / M)
                return min(A, B) / M
            return 0
        else:
            raise ValueError("unsupported type %r" % type(A))


class BlacklistRow(namedtuple('BlacklistRow',
                              'receiver_name receiver date year trial sender_name sender')):
    """Represents a record from the balcklist data. The
    record specifies:

    * receiver_name
    * receiver
    * date
    * year
    * trial
    * sender_name
    * sender

    """

    __slots__ = ()

    @classmethod
    def _factory(cls, i):
        """create a BlacklistRow instance out of the passed tuple

        :returns: an instance of :py:class:`BlacklistRow` or None
                if the format of the tuple is not acceptable"""

        def myor(gn, n):
            if len(n) > 0:
                return n
            if len(gn) > 0:
                return gn
            return ''

        def myint(a):
            try:
                return int(a)
            except ValueError:
                return a

        def myfloat(a):
            try:
                return float(a)
            except ValueError:
                return a

        i = map(lambda a: a.strip(), i)
        rname = myor(i[0], i[1])
        sname = myor(i[6], i[7])
        date, year, trial = i[3:6]
        rid, sid = i[2], i[8]
        row = cls._make([rname, myint(rid), date, myint(year), myfloat(trial), sname, myint(sid)])
        return all(map(lambda a: isinstance(a, int), [row.sender, row.year, row.receiver])), row


class BlacklistData(object):
    """class for loading and maintaining the blacklist data.
    an instance of this class maintains three sets of actors
    (of type set)

    senders
        actors that were called to testify and named
        someone other than themselves

    receivers
        actors that were named by someone at some point

    unfriend
        actors that were called to testify by made no
        names other than themselves

    Further, it maintains an index (dict) of the naming events

    data
        a dictionary of the type (serder, receiver, year)
        --> :py:class:`BlacklistRow` instance for all valid
        rows in the input file (validity is given by :py:func:`BlacklistRow._factory`)

    invalid_data
        a dictionary of the type (serder, receiver, year)
        --> :py:class:`BlacklistRow` instance for invalid
        lines in the input file (validity is given by
        :py:func:`BlacklistRow._factory`)

    trials
        a dictionary of the type (sender, year) --> trial

        .. warning::

           notice that trials 30 and 59 are missing,
           also, we updated the trial numbers of 12729 and 4540
           which were missing on the lists of giacomo
    """

    def __init__(self, fname):
        def __valid_cond((v, _)):
            return v

        reader = ucsv.UnicodeReader(open(fname, 'rb'), encoding='latin1')
        bl_data = filter(lambda (v, r): r.sender != r.receiver and r.receiver != 'receiver',
                         map(BlacklistRow._factory, reader))
        log.info("%d namings" % (len(bl_data)))

        self.data = {}
        for vl, br in filter(__valid_cond, bl_data):
            self.data[(br.sender, br.receiver, br.year)] = br

        self.gray = set(gray_list)

        self.senders = set(
            map(attrgetter("sender"), map(itemgetter(1), filter(lambda (v, r): r.sender != '', bl_data)))
        )

        self.receivers = set(
            map(attrgetter("receiver"), map(itemgetter(1), filter(lambda (v, r): r.receiver != '', bl_data)))
        )

        self.unfriend = set(map(itemgetter(0), unfriendly))

        # self.trials[ (sender, year) ] = trial number
        tr = map(lambda t: ((t.sender, t.year), t.trial), self.data.values())
        tr += map(lambda t: ((t[0], t[2]), t[3]), unfriendly)
        self.trials = dict(tr)

        # all actors
        self.actors = self.senders | self.receivers | self.unfriend | self.gray

    def naming(self, actors, year):
        """naming map up to the indicated year

        :param iterable actors: list of actors on which to operate
        :param int year: year
        :returns: a tuple (AI, M) where M is an indicator matrix M[i,j] = 1 for all namings (i -> j) up to
                (including) year, and AI is a dict of the type AI[actor_id] = index in M"""

        ai = dict(map(lambda t: (t[1], t[0]), enumerate(actors)))
        n = len(actors)
        m = np.zeros((n, n), dtype=np.int8)

        for s, r, y in self.data:
            if y <= year and (s in ai) and (r in ai):
                m[ai[s], ai[r]] += 1
        log.info("%d namings" % (np.sum(m)))
        return ai, m

    def reciprocity(self, s, r, y):
        """indicates whether the reciprocal naming happened
        sometime in the past

        the naming event should have happened in this
        direction (*r* names *s*).

        :param int s: (putative) sender
        :param int r: (putative) receiver
        :param int y: year
        :returns: tuple (reciprocity, year difference of naming event)
        """

        trial = ((s,r,y) in self.data and self.data[(s, r, y)].trial or 1000)
        cond = lambda n: (n.sender, n.receiver) == (r, s) and ((n.year < y) or (n.year == y and n.trial < trial))
        event = filter(cond, self.data.values())
        assert len(event) < 2
        return len(event), (len(event) and (y - event[0].year) or 0)


class ArtistInfoRow(namedtuple('ArtistInfoRow',
                               'artist role year nominated castid i genre film champ prod_house')):

    """Record representing features of an actor. The input file
    has many features, but the one maintained here are

    artist
        the name of the artist (string)

    role
        the roles of the artist (tuple of ints)

    year
        year of the performance (int)

    nominated
        whether the performance had a nomination (bool)

    castid
        id of the cast (int)

    i
        an id???  no idea what this is (int)

    genre
        genres of the performance (tuple of strings)

    film
        title of the performance (string)

    champ
        whether this is a champ movie (bool)

    production_house
        production house (string)
    """

    __slots__ = ()

    @classmethod
    def _factory(cls, i):
        """instantiate :py:class:`ArtistInfoRow` from a
        tuple. see also :py:mod:`giacomo.ucsv` record

        :param tuple i: the parsed record
        :returns: :py:class:`ArtistInfoRow`
        """

        (name, roles, year, nom, castid, _i, genres, film, champ, prn) = i

        return cls._make([name,
                         tuple(map(int, roles.split(';'))),
                         int(year),
                         (nom == 'True'),
                         int(castid),
                         _i,
                         tuple(genres.split(';')),
                         film,
                         (champ == 'True'),
                         prn])

    def _csvtup(self):
        t = (self.artist,
             ";".join(map(str, self.role)),
             self.year,
             (self.nominated and "True" or "False"),
             self.castid,
             self.i,
             ";".join(self.genre),
             self.film,
             (self.champ and "True" or "False"),
             self.prod_house)
        return tuple(map(lambda v: (type(v) == int and str(v) or v), t))


class ArtistInfoData(object):
    """loads and maintains a collection of
    :py:class:`ArtistInfoRow` instances. An instance has

    data
        a dictionary of the type
        data[:py:class:`ArtistInfoRow`.castid] = [:py:class:`ArtistInfoRow`].
        In other words: (artist_id) --> list of performances of this actor
    sim_attributes
        a tuple of attributes (of :py:class:`ArtistInfoRow`)
        that can be used as relations.
    """

    def __init__(self, fname, select=None):
        reader = ucsv.UnicodeReader(open(fname, 'rb'), encoding='latin1')
        self.data = {}
        invalid = 0
        for i, r in enumerate(reader):
            if (select is None) or (int(r[4]) in select):
                try:
                    a = ArtistInfoRow._factory(r)
                    self.data.setdefault(a.castid, []).append(a)
                except ValueError:
                    invalid += 1
        log.info("load %d actors discarded %d ..." % (len(self.data), invalid))
        self.sim_attributes = ("role", "genre",
                               "prod_house", "film",
                               "year", "nominated", "champ")

    def adj_matrix(self, year, attr, actors, weights, similarity_function=similarity_function, lag=1000, progress=True):
        """compute the adjacency matrix of a relation on
        the given actors.

        The method will remove all artists that had not have a
        career until now.

        :param int year: the features of actors after or on
            this year will be filtered out
        :param string attr: defines the relation. this is
            an attribute of :py:class:`ArtistInfoRow`
        :param list actors: the matrix will be based on these
            actors
        :param callable similarity_function: function defined
            on two sets or integers giving the similarity.
        :param dict weights: has the weight for each value of the relation domain
        :param int lag: nr. of years back for which to
            consider records. So records will be in the
            period (year-lag, year) both excluded
        :return: a tuple (list of actor_ids, numpy matrix)
        """

        # a projection of self.data, s.t.
        # actors have had a career before this year
        log.info("filtering out career-less actors ...")
        data = {}
        for act in actors:
            career = filter(lambda r: (year - lag) < r.year < year, self.data.get(act, []))
            if career:
                data[act] = career
            else:
                log.info("dropping actor %s", act)
        log.info("indexing (%d) actors ..." % len(data))
        # actor indexes in the matrix
        V = dict(map(lambda (i, a): (a, i),
                 enumerate(filter(lambda a: a in data, actors))))

        matrix = np.zeros((len(V), len(V)), dtype=np.float32)
        widgets = ['Finished: ', progressbar.Percentage(), ' ', progressbar.Bar(marker=progressbar.RotatingMarker()), ' ', progressbar.ETA()]
        if progress:
            pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(data)).start()
        else:
            pbar = None
        prg = 0
        for a1 in V.keys():
            op1 = self.__selector(data[a1], attr)
            i = V[a1]
            for a2 in V.keys():
                j = V[a2]
                op2 = self.__selector(data[a2], attr)
                matrix[i, j] = similarity_function(op1, op2, weights)
            if progress:
                prg += 1
                pbar.update(prg)
        return V, matrix

    def __selector(self, ainfol, sname):
        """selects attributes of a set of artists and packs
        them into an int or a set. this value can be feeded as
        an operant to :py:func:`similarity_function`. Note that
        this is a private method and should be called by
        adj_matrix

        :param iterable ainfol: list of artists (instances of
            :py:class:`ArtistInfoRow`
        :param string sname: attribute of
            :py:class:`ArtistInfoRow` to select
        :returns: set or int
        """

        if not ainfol:
            raise ValueError("empty list of artists")

        ainfo_vals = map(attrgetter(sname), ainfol)
        assert ainfo_vals, "ainfol was not empty, but this is"

        if sname in ("role", "genre"):
            return set(reduce(concat, ainfo_vals, ()))
        elif sname in ("prod_house", "film"):
            return set(ainfo_vals)
        elif sname in ("year",):
            return len(set(ainfo_vals))
        elif sname in ("nominated", "champ"):
            return len(filter(lambda p: p, ainfo_vals))
        else:
            assert not (sname in self.sim_attributes)
            raise ValueError("cannot handle %s (%r)" % (sname, type(sname)))


class Affiliations(object):
    """represents affiliations of a set of actors. maintains

    data
        a dict structure of the type (castid) --> array,
        where array is an indicator array for each of the
        organizations

    affiliations
        a list of names of affiliations

    the data is in the configuration file so this gets
    initialized automatically
    """

    def __init__(self):
        fname = os.path.join(os.path.dirname(__file__), "affiliations.csv")
        reader = ucsv.UnicodeReader(open(fname, 'rb'), encoding='latin1')
        self.data = {}
        for i, row in enumerate(reader):
            if i == 0:
                self.affiliations = row[3:]
                W = len(self.affiliations)
                continue
            if row[1] == '':
                continue
            self.data[int(row[1])] = np.array(map(int, row[3:]), dtype=np.int16)
            assert len(row[3:]) == W, "[%s] %d != %d" % (row[1], len(row[3:]), W)

    def corr(self, s, r):
        """affiliation-based correlation of two actors

        :param int s: sender
        :param int r: receiver
        :returns: correlation or 0 if not in the data
        """

        if s in self.data and r in self.data:
            a,b = self.data[s], self.data[r]
            C = np.corrcoef(np.vstack((a,b)))
            assert np.abs(C[0, 1] - C[1, 0]) < 0.000001
            return float(C[0, 1])
        return 0

    def sim(self, s, r):
        """affiliation-based similarity of two actors

        :param int s: sender
        :param int r: receiver
        :returns: similarity or 0 if (s,r) not in the data
        """

        to_lst = lambda a: set(self.data.get(a, np.array([], dtype=np.float32)).tolist())

        A = set(to_lst(s))
        B = set(to_lst(r))
        if not A:
            return 0
        return similarity_function(A, B, Weight.one)

    def __pairwise_f(self, f):
        for i in self.data:
            for j in self.data:
                yield (i, j, f(i, j))

    def pairwise_corr(self):
        """generate correlations over the affiliation array
        for each pair of actors

        this calls :py:meth:`Affiliations.corr` on pairs

        :returns: yields tuples (actorid, actorid, corr)
        """

        self.__pairwise_f(self.corr)

    def corr_comm(self, artist):
        """is the artist part of comm_pol_ass or comm_party?

        :param int artist: the artist
        :returns: bool
        """

        assert 'comm_pol_ass' in self.affiliations and 'comm_party' in self.affiliations, str(self.affiliations)

        if artist in self.data:
            i1 = self.affiliations.index('comm_pol_ass')
            i2 = self.affiliations.index('comm_party')
            t = self.data[artist]
            return t[i1] + t[i2] > 0
        return False

    def sums(self, artist):
        if artist in self.data:
            i = min(self.affiliations.index('comm_pol_ass'),
                    self.affiliations.index('comm_party'))
            return np.sum(self.data[artist][:i])
        return 0
