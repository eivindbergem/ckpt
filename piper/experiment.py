import json
import os.path
import time
import logging

from .misc import mkdirp, get_piper_path
from .checkpoint import Checkpoint


class Experiment(object):
    def __init__(self, name, config):
        self.name = name
        self.config = config

    def __enter__(self):
        logging.info("Running experiment '{}'".format(self.name))
        self.metrics = {}
        self.timestamp = time.time()
        self.checkpoints = []

        return self

    def __exit__(self, *exc_details):
        if self.metrics:
            logging.info("Experiment done, saving config and results.")
            self.save()
        else:
            logging.info("No metrics found, not saving experiment.")

    def add_metrics(self, metrics):
        self.metrics.update(metrics)

    def get_path(self):
        path = os.path.join(get_piper_path(), "experiments", self.name)

        mkdirp(path)

        return path

    def get_filename(self):
        return os.path.join(self.get_path(),
                            "{}.json".format(self.timestamp))

    def save(self):
        filename = self.get_filename()

        data = {"config": self.config,
                "metrics": self.metrics}

        with open(filename, "w") as fd:
            json.dump(data, fd)

    def add_checkpoint(self, name, config=None, dependencies=None):
        if not config:
            config = {}

        if not dependencies:
            dependencies = []

        if self.checkpoints:
            prev = self.checkpoints[-1]
        else:
            prev = None

        ckpt = Checkpoint(name, config, prev, dependencies)

        self.checkpoints.append(ckpt)

        return ckpt
