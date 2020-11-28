"""
Access to the database
"""

from dataclasses import dataclass
import sqlite3

import pandas as pd


@dataclass
class Db(object):
    connection: sqlite3.Connection

    @classmethod
    def from_path(cls, path: str):
        return cls(sqlite3.connect(path))

    def query_as_df(self, query):
        return pd.read_sql_query(query, self.connection)

    @classmethod
    def encode(cls, lst):
        return "-".join(map(str, lst))

    @classmethod
    def decode(cls, val):
        try:
            return tuple(map(int, val.split('-')))
        except ValueError:
            return ()

    def tb(self, table):
        return self.query_as_df(f"select * from {table}")

    @property
    def naming(self):
        return self.query_as_df('select sender, receiver, year from trials')

    @property
    def all_actor_ids(self):
        a = self.query_as_df('select distinct castid from artist')
        u = self.query_as_df('select distinct castid from unfriendly')
        g = self.query_as_df('select distinct castid from gray')

        return set(pd.concat([a, u, g]).castid)
