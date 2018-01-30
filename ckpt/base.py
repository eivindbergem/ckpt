from abc import ABC, abstractmethod
from .checkpoint import Checkpoint
from .misc import get_hash

import logging
import inspect

class Pipe(ABC):
    def get_default_params(self):
        params = inspect.signature(self.__init__).parameters

        return {key: params[key].default
                for key in params}

    def get_params(self, show_defaults=False):
        defaults = self.get_default_params()

        return {key: getattr(self, key) for key in defaults
                if show_defaults or defaults[key] != getattr(self, key)}

    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def load_model(self, ckpt):
        pass

    @abstractmethod
    def save_model(self, ckpt):
        pass

    def _fit(self, X, y, use_checkpoints):
        if use_checkpoints:
            dependencies = [get_hash(sorted(self.get_params().items())),
                            get_hash((X, y))]

            with Checkpoint("{}.fit".format(self.get_name()),
                            dependencies) as ckpt:
                if ckpt.exists():
                    ckpt.logger.info("Loading checkpoint for {} from {}"
                                     .format(ckpt.name, ckpt.get_path()))
                    self.load_model(ckpt)
                else:
                    self.fit(X, y)
                    ckpt.logger.info("Saving checkpoint for {} to {}"
                                     .format(ckpt.name, ckpt.get_path()))
                    self.save_model(ckpt)
        else:
            self.fit(X, y)

class Predictor(Pipe):
    @abstractmethod
    def predict(self, X):
        pass

class Transformer(Pipe):
    @abstractmethod
    def transform(self, X, y=None):
        pass
