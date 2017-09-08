import hashlib
import os
import os.path
import time
import logging
import gzip

from functools import wraps

from .misc import mkdirp, get_piper_path

BLOCKSIZE = 2**13

class Checkpoint(object):
    def __init__(self, name, config, prev_checkpoint, dependencies):
        self.name = name
        self.config = config
        self.prev_checkpoint = prev_checkpoint
        self.dependencies = dependencies
        self.path = os.path.join(get_piper_path(), "checkpoints", self.get_hash())

    def __enter__(self):
        if self.exists():
            logging.info("Found checkpoint for {}".format(self.name))
            logging.debug(self.get_path())

        self.mkdir()

        return self

    def __exit__(self, *exc_details):
        if os.path.exists(self.get_path()) and not self.listdir():
            logging.info("Checkpoint dir for {} empty, removing.".format(self.name))
            logging.debug(self.get_path)
            os.rmdir(self.get_path())

    @staticmethod
    def save(fn):
        @wraps(fn)
        def wrapper(ckpt, *args, **kwargs):
            logging.info("Saving checkpoint for {}".format(ckpt.name))
            logging.debug(ckpt.get_path())
            return fn(ckpt, *args, **kwargs)

        return wrapper

    @staticmethod
    def load(fn):
        @wraps(fn)
        def wrapper(ckpt, *args, **kwargs):
            logging.info("Loading checkpoint for {}".format(ckpt.name))
            logging.debug(ckpt.get_path())
            return fn(ckpt, *args, **kwargs)

        return wrapper

    def get_hash(self):
        m = hashlib.sha256()

        m.update(self.name.encode("utf-8"))

        for key, value in sorted(self.config.items()):
            m.update("{}{}".format(key, value).encode("utf-8"))

        files = self.dependencies[:]

        if self.prev_checkpoint:
            files += self.prev_checkpoint.listdir()

        for filename in files:
            with open(filename, "rb") as fd:
                data = fd.read(BLOCKSIZE)

                if not data:
                    break

                m.update(data)

        return m.hexdigest()

    def get_path(self):
        return self.path

    def join_path(self, *paths):
        return os.path.join(self.get_path(), *paths)

    def mkdir(self):
        mkdirp(self.get_path())

    def listdir(self):
        return [self.join_path(path) for path in os.listdir(self.get_path())]

    def exists(self):
        return os.path.exists(self.get_path()) and len(self.listdir()) > 0

    def open_file(self, filename, mode="r", compression=gzip):
        filename = self.join_path(filename)

        if compression:
            # Default to text mode if not specified, as is the case
            # for builtins.open
            if not any(True for c in mode
                        if c in ("t", "b")):
                mode += "t"

            filename += ".gz"

            return compression.open(filename, mode)
        else:
            return open(filename, mode)
