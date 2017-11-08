import os.path
import time
import logging
import hashlib
import pickle

from .misc import mkdirp, get_ckpt_path, save_as_json
from .checkpoint import Checkpoint

LOG_FORMAT = '%(asctime)s %(name)-10s %(message)s'
LOG_DATEFMT = '%H:%M'
LOG_LEVEL = logging.INFO

logging.basicConfig(format=LOG_FORMAT,
                    datefmt=LOG_DATEFMT,
                    level=LOG_LEVEL)

class Experiment(object):
    def __init__(self, name, config, dry_run=False):
        self.name = name
        self.config = config
        self.metadata = {"name": name}
        self.dry_run = dry_run
        self.logger = logging.getLogger("ckpt.experiment")

    def __enter__(self):
        self.logger.info("Running experiment '{}'".format(self.name))
        self.metrics = {}
        self.metadata['start'] = time.time()

        mkdirp(self.get_path())

        return self

    def __exit__(self, *exc_details):
        self.metadata['stop'] = time.time()

        if self.metrics:
            self.logger.info("Experiment done, saving config and results.")
            self.save()
        else:
            self.logger.info("Experiment done, no metrics added, not saving.")

    def add_metrics(self, metrics):
        for k, v in metrics.items():
            self.logger.info("Added metric: {} = {}".format(k, v))

        self.metrics.update(metrics)

    def get_filename(self, data):
        def update_hash(m, d):
            for key, value in sorted(d.items()):
                m.update(key.encode("utf-8"))

                if isinstance(value, dict):
                    update_hash(m, value)
                else:
                    m.update(str(value).encode("utf-8"))

        m = hashlib.sha256()

        update_hash(m, data)

        return os.path.join(self.get_path(), "{}.pkl".format(m.hexdigest()))

    def get_path(self):
        return os.path.join(get_ckpt_path(), "experiments")

    def save(self):
        data = {"config": self.config,
                "metrics": self.metrics,
                "metadata": self.metadata}

        if not self.dry_run:
            with open(self.get_filename(data), "wb") as fd:
                pickle.dump(data, fd)
