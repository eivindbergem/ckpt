import os
import json
import joblib
import hashlib

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

def mark_final(iterable):
    prev = None

    for item in iterable:
        if prev:
            yield False, prev

        prev = item

    yield True, prev

def get_hash(item):
    return joblib.hashing.hash(item, hash_name="sha256")

def get_file_hash(filename, blocksize=2**20):
    m = hashlib.sha256()

    with open(filename, "rb", buffering=0) as fd:
        while True:
            data = fd.read(blocksize)

            if not data:
                break

            m.update(data)

    return m.hexdigest()

def shortest_unique_prefix(lst):
    n = len(lst)

    for i in range(1, n + 1):
        prefixes = set(item[:i] for item in lst)

        if len(prefixes) == n:
            return i

def get_short_hashes(hashes):
    k = shortest_unique_prefix(hashes)

    return [item[:k] for item in hashes]

def get_long_hash(short_hash):
    path = os.path.join(get_ckpt_path(), "experiments")

    for filename in os.listdir(path):
        if filename.startswith(short_hash):
            return filename
