import os
import os.path

from collections import defaultdict
from tabulate import tabulate

from .misc import get_ckpt_path, load_json
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

    for name, config, metrics in rows:
        for k, v in config.items():
            values[k].add(str(v))

    keys = set(k for k, v in values.items()
               if len(v) > 1)

    pruned = []

    for name, config, metrics in rows:
        row = (name,
               {k: v for k, v in config.items()
                if k in keys},
               metrics)

        if not row in pruned:
            pruned.append(row)

    return pruned

def get_experiments(ex_name=None):
    path = os.path.join(get_ckpt_path(), "experiments")

    experiments = []

    for name in os.listdir(path):
        if ex_name and name != ex_name:
            continue

        for experiment in os.listdir(os.path.join(path, name)):
            config = load_json(os.path.join(path, name, experiment,
                                            "config.json"))

            try:
                metrics = load_json(os.path.join(path, name, experiment,
                                                 "metrics.json"))
            except FileNotFoundError:
                continue

            if not metrics:
                continue

            experiments.append((name, flatten(config), metrics))

    return prune(experiments)

def values_from_keys(d, keys, default=None):
    return [d[k] if k in d
            else default
            for k in keys]

def tabulate_data(experiments, sort_by=None, floatfmt=".4f"):
    config_keys = set([])
    metrics_keys = set([])

    for _, config, metrics in experiments:
        config_keys.update(config.keys())
        metrics_keys.update(metrics.keys())

    config_keys = sorted(config_keys - set(ckpt_config['report']['ignore-config']))
    metrics_keys = sorted(metrics_keys - set(ckpt_config['report']['ignore-metrics']))

    headers = ["name"] + config_keys + metrics_keys
    data = [[name] + values_from_keys(config, config_keys)
            + values_from_keys(metrics, metrics_keys)
            for name, config, metrics in experiments]

    return tabulate(data, headers=headers, floatfmt=floatfmt)
