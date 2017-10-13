import hashlib
import os
import os.path
import time
import logging
import gzip

from functools import wraps

from .misc import mkdirp, get_ckpt_path

BLOCKSIZE = 2**13

class Checkpoint(object):
    def __init__(self, name, config, prev_checkpoint, dependencies, logger):
        self.name = name
        self.config = config
        self.prev_checkpoint = prev_checkpoint
        self.dependencies = dependencies
        self.path = os.path.join(get_ckpt_path(), "checkpoints", self.get_hash())
        self.logger = logger

    def __enter__(self):
        self.logger.info("Entering checkpoing '{}'".format(self.name))

        if self.exists():
            self.logger.info("Found checkpoint for {}".format(self.name))

        self.mkdir()

        return self

    def __exit__(self, *exc_details):
        if os.path.exists(self.get_path()) and not self.listdir():
            self.logger.info("Checkpoint dir for {} empty, removing."
                             .format(self.name))
            os.rmdir(self.get_path())

    @staticmethod
    def save(fn):
        @wraps(fn)
        def wrapper(ckpt, *args, **kwargs):
            ckpt.logger.info("Saving checkpoint for {} to {}"
                             .format(ckpt.name, ckpt.get_path()))
            return fn(ckpt, *args, **kwargs)

        return wrapper

    @staticmethod
    def load(fn):
        @wraps(fn)
        def wrapper(ckpt, *args, **kwargs):
            ckpt.logger.info("Loading checkpoint for {} from {}"
                             .format(ckpt.name, ckpt.get_path()))
            return fn(ckpt, *args, **kwargs)

        return wrapper

    def get_hash(self):
        m = hashlib.sha256()

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
