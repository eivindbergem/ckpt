import json
import os.path

from contextlib import AbstractContextManager

from .misc import mkdirp
from .checkpoint import Checkpoint

class Experiment(AbstractContextManager):
    def __init__(self, name, config):
        self.name = name
        self.config = config

    def __enter__(self):
        self.metrics = {}

        return self

    def __exit__(self, *exc_details):
        self.save()

    def add_metric(self, key, value):
        self.metrics[key] = value

    def get_path(self):
        path = "experiments"

        mkdirp(path)

        return path

    def config_str(self):
        return "{}-{}".format(self.name,
                              "-".join("{}-{}".format(key, value)
                                       for key, value in self.config.items()))

    def get_filename(self):
        return os.path.join(self.get_path(),
                            "{}.json".format(self.config_str()))

    def save(self):
        filename = self.get_filename()

        data = {"config": self.config,
                "metrics": self.metrics}

        with open(filename, "w") as fd:
            json.dump(data, fd)

    def checkpoint(self, name):
        return Checkpoint(name, self.name, self.config)
