import os
import os.path
import pickle

from collections import defaultdict
from tabulate import tabulate

from .misc import get_ckpt_path, load_json, get_short_hashes
from .config import ckpt_config

def flatten(d):
    flattened = {}

    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in flatten(v).items():
                flattened["{}-{}".format(k, k2)] = v2
        else:
            flattened[k] = v

    return flattened

def prune(rows):
    values = defaultdict(set)

    for _, _, config, _ in rows:
        for k, v in config.items():
            values[k].add(str(v))

    keys = set(k for k, v in values.items()
               if len(v) > 1)

    pruned = []

    for short_hash, name, config, metrics in rows:
        row = (short_hash, name,
               {k: v for k, v in config.items()
                if k in keys},
               metrics)

        if not row in pruned:
            pruned.append(row)

    return pruned

def get_experiments(ex_name=None):
    path = os.path.join(get_ckpt_path(), "experiments")

    experiments = []
    filenames = os.listdir(path)
    short_hashes = get_short_hashes(filenames)

    for short_hash, experiment in zip(short_hashes, filenames):
        with open(os.path.join(path, experiment), "rb") as fd:
            data = pickle.load(fd)

        if ex_name and data['metadata']['name'] != ex_name:
            continue

        experiments.append((short_hash, data['metadata']['name'],
                            flatten(data['config']), data['metrics']))

    return prune(experiments)

def values_from_keys(d, keys, default=None):
    return [d[k] if k in d
            else default
            for k in keys]

def default_value(config, default, *keys):
    d = config

    for key in keys:
        if key not in d:
            return default

        d = d[key]

    return set(d)

def tabulate_data(experiments, sort_by=None, floatfmt=".4f"):
    config_keys = set([])
    metrics_keys = set([])

    for _, _, config, metrics in experiments:
        config_keys.update(config.keys())
        metrics_keys.update(metrics.keys())

    config_keys = sorted(config_keys - default_value(ckpt_config, set([]),
                                                     "report", "ignore-config"))
    metrics_keys = sorted(metrics_keys - default_value(ckpt_config, set([]),
                                                       "report", "ignore-metrics"))

    headers = ["id", "name"] + config_keys + metrics_keys
    data = [[short_hash, name] + values_from_keys(config, config_keys)
            + values_from_keys(metrics, metrics_keys)
            for short_hash, name, config, metrics in experiments]

    return tabulate(sorted(data, key=lambda x : x[1]),
                    headers=headers, floatfmt=floatfmt)

def remove_experiment(filename):
    os.remove(os.path.join(get_ckpt_path(), "experiments", filename))