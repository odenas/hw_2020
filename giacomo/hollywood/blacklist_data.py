from collections import namedtuple
import csv
import logging
from operator import itemgetter

import numpy as np

from . import unfriendly, gray_list

log = logging.getLogger()
blacklist_fields = 'receiver_name receiver date year trial sender_name sender'.split()


class BlacklistRow(namedtuple('BlacklistRow', blacklist_fields)):
    __slots__ = ()

    @classmethod
    def _factory(cls, i):
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

        i = list(map(lambda a: a.strip(), i))
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
        reader = csv.reader(open(fname, 'rt', encoding='latin1'))
        bl_data = []
        for valid, row in map(BlacklistRow._factory, reader):
            if row.sender != row.receiver and row.receiver != 'receiver':
                bl_data.append((valid, row))
        log.info("%d namings", len(bl_data))

        self.data = {}
        self.senders = []
        self.receivers = []
        for valid, row in bl_data:
            if valid:
                self.data[(row.sender, row.receiver, row.year)] = row
            if row.sender != '':
                self.senders.append(row.sender)
            if row.receiver != '':
                self.receivers.append(row.receiver)
        self.senders = set(self.senders)
        self.receivers = set(self.receivers)

        self.gray = set(gray_list)
        self.unfriend = set(map(itemgetter(0), unfriendly))

        # self.trials[ (sender, year) ] = trial number
        tr = list(map(lambda t: ((t.sender, t.year), t.trial), self.data.values()))
        tr += list(map(lambda t: ((t[0], t[2]), t[3]), unfriendly))
        self.trials = dict(tr)

        # all actors
        self.actors = self.senders | self.receivers | self.unfriend | self.gray

    def naming(self, actors, year):
        """naming map up to the indicated year

        :param iterable actors: list of actors on which to operate
        :param int year: year
        :returns: a tuple (AI, M) where M is an indicator matrix M[i,j] = 1
                for all namings (i -> j) up to
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

        trial = ((s, r, y) in self.data and self.data[(s, r, y)].trial or 1000)

        def cond(n):
            return ((n.sender, n.receiver) == (r, s) and
                    ((n.year < y) or (n.year == y and n.trial < trial)))
        event = list(filter(cond, self.data.values()))
        assert len(event) < 2
        return len(event), (len(event) and (y - event[0].year) or 0)
