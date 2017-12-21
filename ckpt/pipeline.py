from .misc import mark_final

from functools import wraps
from collections import OrderedDict

import json
import pickle
import logging

def lazy(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not hasattr(fn, "__cache__"):
            fn.__cache__ = fn()

        return fn.__cache__

    return wrapper

def load_config(filename):
    if filename.endswith("json"):
        with open(filename) as fd:
            return json.load(open(filename), object_pairs_hook=OrderedDict)
    elif filename.endswith("pkl"):
        with open(filename, "rb") as fd:
            data = pickle.load(fd)
            return data['config']


class Pipeline(object):
    def __init__(self, pipes):
        self.labels, self.pipes = list(zip(*pipes))
        self.logger = logging.getLogger("ckpt.pipeline")

    @classmethod
    def from_file(cls, filename, pipes):
        return cls.from_dict(load_config(filename), pipes)

    @classmethod
    def from_dict(cls, d, pipes):
        return cls([(label, pipes[label](**config))
                    for label, config in d.items()])

    def get_name(self):
        return "+".join(self.labels)

    def get_params(self, show_defaults=False):
        return OrderedDict(((label, pipe.get_params(show_defaults))
                            for label, pipe in zip(self.labels, self.pipes)))

    def fit(self, X, y=None, use_checkpoints=True):
        self.logger.info("Fitting pipeline {}".format(self.get_name()))

        for is_final, pipe in mark_final(self.pipes):
            pipe._fit(X, y, use_checkpoints)

            if not is_final:
                X, y = pipe.transform(X, y)

    def predict(self, X):
        y = None

        for pipe in self.pipes[:-1]:
            X, y = pipe.transform(X, y)

        return self.pipes[-1].predict(X)
