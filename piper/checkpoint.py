import hashlib
import os.path

from contextlib import AbstractContextManager

from .misc import mkdirp

class Checkpoint(AbstractContextManager):
    def __init__(self, checkpoint_name, experiment_name, config):
        self.ckpt_name = checkpoint_name
        self.ex_name = experiment_name
        self.config = config

    def __enter__(self):
        return self

    def __exit__(self, *exc_details):
        pass

    def get_hash(self):
        m = hashlib.sha1()

        for s in (self.ckpt_name, self.ex_name):
            m.update(s.encode("utf-8"))

        for key, value in sorted(self.config.items()):
            m.update("{}{}".format(key, value).encode("utf-8"))

        return m.hexdigest()

    def get_ckpt_dir(self):
        path = "checkpoints"

        mkdirp(path)

        return path

    def get_path(self):
        return os.path.join(self.get_ckpt_dir(),
                            self.get_hash())


