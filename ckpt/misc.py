import os
import json
import hashlib
import csv
import inspect
import pickle

from functools import wraps
from contextlib import contextmanager

ckpt_path = ".ckpt"

def set_ckpt_path(path):
    global ckpt_path

    ckpt_path = path

def get_ckpt_path():
    return ckpt_path

def mkdirp(dirname):
    """Same as `mkdir -p <dirname>`"""

    try:
        os.makedirs(dirname)
    except OSError as err:
        if not (err.errno == 17 and os.path.isdir(dirname)):
            raise

def add_defaults(config, defaults):
    return {key: (value
                  if key not in config
                  else config[key])
            for key, value in defaults.items()}

def save_as_json(data, filename):
    with open(filename, "w") as fd:
        json.dump(data, fd)

def load_json(filename):
    with open(filename) as fd:
        return json.load(fd)

@contextmanager
def open_csv(filename, mode="r", dialect="excel", **fmtparams):
    with open(filename, mode, newline='') as fd:
        if "r" in mode:
            fn = csv.reader
        elif "w" in mode:
            fn = csv.writer

        yield fn(fd, dialect, **fmtparams)

def save_as_csv(data, filename, headers=None):
    with open_csv(filename, "w") as writer:
        if headers:
            writer.writerow(headers)

        for row in data:
            writer.writerow(row)

def mark_final(iterable):
    prev = None

    for item in iterable:
        if prev:
            yield False, prev

        prev = item

    yield True, prev

def get_hash(item):
    m = hashlib.sha256()

    m.update(pickle.dumps(item))

    return m.hexdigest()

def get_file_hash(filename, blocksize=2**20):
    m = hashlib.sha256()

    with open(filename, "rb", buffering=0) as fd:
        while True:
            data = fd.read(blocksize)

            if not data:
                break

            m.update(data)

    return m.hexdigest()

def shortest_unique_prefix(lst, minimum=None):
    n = len(lst)

    for i in range(1, n + 1):
        prefixes = set(item[:i] for item in lst)

        if len(prefixes) == n:
            if minimum:
                return max([i, minimum])
            else:
                return i

def get_short_hashes(hashes, minimum=None):
    k = shortest_unique_prefix(hashes, minimum)

    return [item[:k] for item in hashes]

def get_long_hash(short_hash):
    path = os.path.join(get_ckpt_path(), "experiments")

    for filename in os.listdir(path):
        if filename.startswith(short_hash):
            return filename

def autoinit(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        params = inspect.signature(self.__init__).bind(*args, **kwargs)
        params.apply_defaults()

        for key, value in params.arguments.items():
            setattr(self, key, value)

        return fn(self, *args, **kwargs)

    return wrapper
