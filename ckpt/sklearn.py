from functools import wraps
from sklearn.externals import joblib
import dill
import inspect

from .base import Transformer, Predictor, Pipe

class SKPipe(Pipe):
    def __init__(self, sk_obj):
        self.sk_obj = sk_obj

    def _get_defaults(self):
        params = inspect.signature(self.sk_obj.__init__).parameters

        return {key: params[key].default
                for key in params}

    def fit(self, X, y):
        self.sk_obj.fit(X, y)

    def get_params(self, show_defaults=False):
        defaults = self._get_defaults()
        params = self.sk_obj.get_params()

        return {key: params[key] for key in params
                if show_defaults or defaults[key] != params[key]}

    def get_name(self):
        return self.sk_obj.__class__.__name__

    def _model_filename(self):
        return "{}.model.xz".format(self.get_name())

    def load_model(self, ckpt):
        for filename in ckpt.listdir():
            if "model" in filename:
                self.sk_obj = joblib.load(filename)
                break

    def save_model(self, ckpt):
        joblib.dump(self.sk_obj, ckpt.join_path(self._model_filename()))

class SKTransformer(SKPipe, Transformer):
    def transform(self, X, y=None):
        X = self.sk_obj.transform(X)

        return X, y

    def _values_filename(self):
        return "{}.values.pkl".format(self.get_name())

    def save_values(self, ckpt, X, y):
        joblib.dump((X, y), ckpt.join_path(self._values_filename()))

    def load_values(self, ckpt):
        return joblib.load(ckpt.join_path(self._values_filename()))

class SKPredictor(SKPipe, Predictor):
    def predict(self, X):
        return self.sk_obj.predict(X)

def has_method(obj, method_name):
    method = getattr(obj, method_name, None)

    if callable(method):
        return True
    else:
        return False

def wrap(sk_class):
    @wraps(sk_class)
    def wrapper(*args, **kwargs):
        return wrapper.cls(sk_class(*args, **kwargs))

    if has_method(sk_class, "transform"):
        wrapper.cls = SKTransformer
    elif has_method(sk_class, "predict"):
        wrapper.cls = SKPredictor

    return wrapper
