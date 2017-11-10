import os
import os.path
import time
import logging
import gzip
import hashlib

from functools import wraps

from .misc import mkdirp, get_ckpt_path

class Checkpoint(object):
    path = None

    def __init__(self, name, dependencies):
        self.name = name
        self.dependencies = dependencies
        self.logger = logging.getLogger("ckpt.checkpoint")

    def __enter__(self):
        self.logger.info("Entering checkpoint '{}'".format(self.name))

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

        m.update(self.name.encode("utf-8"))

        for dep in sorted(self.dependencies):
            m.update(dep.encode("utf-8"))

        return m.hexdigest()

    def get_path(self):
        if not self.path:
            self.path = os.path.join(get_ckpt_path(), "checkpoints",
                                     self.get_hash())

        return self.path

    def join_path(self, *paths):
        return os.path.join(self.get_path(), *paths)

    def mkdir(self):
        mkdirp(self.get_path())

    def listdir(self):
        return sorted([self.join_path(path)
                       for path in os.listdir(self.get_path())])

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
