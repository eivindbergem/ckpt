import json
import os.path
import time
import logging

from .misc import mkdirp, get_ckpt_path, save_as_json
from .checkpoint import Checkpoint

LOG_FORMAT = '%(asctime)s %(name)-10s %(message)s'
LOG_DATEFMT = '%H:%M'
LOG_LEVEL = logging.INFO

logging.basicConfig(format=LOG_FORMAT,
                    datefmt=LOG_DATEFMT,
                    level=LOG_LEVEL)

class Experiment(object):
    def __init__(self, name, config):
        self.name = name
        self.config = config

    def __enter__(self):
        self.metrics = {}
        self.timestamp = time.time()
        self.checkpoints = []

        mkdirp(self.get_path())
        self.save_config()

        self.log_handler = logging.FileHandler(self.join_path("log"))
        self.log_handler.setFormatter(logging.Formatter(LOG_FORMAT,
                                                        datefmt=LOG_DATEFMT))
        logging.getLogger('').addHandler(self.log_handler)

        self.logger = logging.getLogger("ckpt.experiment<{}>".format(self.name))

        self.logger.info("Running experiment '{}'".format(self.name))

        return self

    def __exit__(self, *exc_details):
        self.logger.info("Experiment done, saving config and results.")

        self.save_metrics()
        self.save_checkpoints()
        logging.getLogger('').removeHandler(self.log_handler)

    def add_metrics(self, metrics):
        for k, v in metrics.items():
            self.logger.info("Added metric: {} = {}".format(k, v))

        self.metrics.update(metrics)

    def get_path(self):
        return os.path.join(get_ckpt_path(), "experiments", self.name,
                            str(self.timestamp))

    def join_path(self, *paths):
        return os.path.join(self.get_path(), *paths)

    def save_config(self):
        save_as_json(self.config, self.join_path("config.json"))

    def save_metrics(self):
        save_as_json(self.metrics, self.join_path("metrics.json"))

    def save_checkpoints(self):
        save_as_json([ckpt.get_path() for ckpt in self.checkpoints],
                     self.join_path("checkpoints.json"))

    def add_checkpoint(self, name, config=None, dependencies=None):
        if not config:
            config = {}

        if not dependencies:
            dependencies = []

        if self.checkpoints:
            prev = self.checkpoints[-1]
        else:
            prev = None

        ckpt = Checkpoint(name, config, prev, dependencies, self.logger)

        self.checkpoints.append(ckpt)

        return ckpt
