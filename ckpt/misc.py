import os
import json

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
