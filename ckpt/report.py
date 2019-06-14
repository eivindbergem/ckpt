import os
import os.path
import pickle
import pprint
import numpy as np

from collections import defaultdict
from tabulate import tabulate
from sklearn.metrics import confusion_matrix

from .misc import get_ckpt_path, load_json, get_short_hashes
from .config import ckpt_config
from .experiment import get_metrics, get_reports

def common_prefix(lists):
    n = 0

    for items in zip(*lists):
        if all(item == items[0] for i, item in enumerate(items)):
            n += 1
        else:
            break

    return n

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

def load_experiments(ids=None):
    path = os.path.join(get_ckpt_path(), "experiments")

    filenames = os.listdir(path)
    short_hashes = get_short_hashes(filenames, minimum=7)

    for short_hash, experiment in zip(short_hashes, filenames):
        with open(os.path.join(path, experiment), "rb") as fd:
            data = pickle.load(fd)

        if not ids or short_hash in ids:
            yield short_hash, data

def get_experiments(ids=None, pipe=None, config_filter=None):
    experiments = []

    for short_hash, data in load_experiments(ids):
        # For new style experiments, the results are saved and metrics
        # calculated later, while old style saves only metrics at
        # experiment time.

        if pipe and pipe not in data['config']:
            continue

        if "results" in data:
            if not "metrics" in data:
                data['metrics'] = {}

            for name, result in data['results'].items():
                for metric, fn in get_metrics().items():
                    score = fn(result['y_true'], result['y_pred'])
                    data['metrics']["{}-{}".format(name, metric)] = score

        config = flatten(data['config'])
        remove = False

        if config_filter:
            for key, value in config_filter.items():
                if not key in config or config[key] != value:
                    remove = True
        if remove:
            continue

        experiments.append((short_hash, [key for key in data['config'].keys()],
                            config, data['metrics']))

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

def tabulate_data(experiments, sort_by=None, reverse_sort=True):
    config_keys = set([])
    metrics_keys = set([])
    names = []

    for _, name, config, metrics in experiments:
        config_keys.update(config.keys())
        metrics_keys.update(metrics.keys())
        names.append(name)

    config_keys = sorted(config_keys - default_value(ckpt_config, set([]),
                                                     "report", "ignore-config"))
    metrics_keys = sorted(metrics_keys - default_value(ckpt_config, set([]),
                                                       "report", "ignore-metrics"))

    name_start = common_prefix(names)

    headers = ["id", "name", "config"] + metrics_keys
    data = [[short_hash, "+".join(n for n in name[name_start:])] +
            #values_from_keys(config, config_keys)
            ["; ".join("{}: {}".format(key, config[key])
                       for key in config_keys
                       if key in config)]
            + values_from_keys(metrics, metrics_keys)
            for short_hash, name, config, metrics in experiments]

    if sort_by:
        index = headers.index(sort_by)
        data.sort(key = lambda row : row[index], reverse=reverse_sort)

    return data, headers

def pretty_print(data, headers, floatfmt=".4f"):
    return print(tabulate(data, headers=headers, floatfmt=floatfmt))

def remove_experiment(filename):
    os.remove(os.path.join(get_ckpt_path(), "experiments", filename))

def inspect_experiment(ex_id):
    _, ex = list(load_experiments([ex_id]))[0]

    print("Experiment {}:".format(ex_id))

    pp = pprint.PrettyPrinter()

    pp.pprint(ex['config'])

    results = ex['results']['dev']

    print()
    for fn in get_reports().values():
        fn(results['y_true'], results['y_pred'])

    for name, fn in (("Mean", np.mean),
                     ("Std", np.std),
                     ("Median", np.median)):
        print("{}: {}".format(name, fn(results['y_pred'])))
