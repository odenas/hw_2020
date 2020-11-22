
class Weight(object):
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
