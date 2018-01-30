import argparse
import sys
import os.path
from .misc import get_ckpt_path, get_long_hash, save_as_csv
from .report import (get_experiments, tabulate_data, remove_experiment,
                     pretty_print, inspect_experiment)

def is_bool(item):
    item = item.lower()

    if item == "true":
        return True
    elif item == "false":
        return False
    else:
        raise ValueError()

def autotype(item):
    for t in (int, float, is_bool):
        try:
            return t(item)
        except ValueError:
            pass

    return item

class Parser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='CKPT')
        self.commands = self.parser.add_subparsers(title="commands",
                                                   dest="command")
        self.subparsers = {}

        for name, help_str in (("run", "Run pipeline from config"),
                               ("report", "Report experiments"),
                               ("rerun", "Rerun experiment"),
                               ("remove", "Remove experiment"),
                               ("inspect", "Inspect experiment")):
            self.subparsers[name] = self.commands.add_parser(name, help=help_str)

        self.subparsers['report'].add_argument("--output-format", "-o",
                                               choices=["pretty", "csv"],
                                               default="pretty")
        self.subparsers['report'].add_argument("--filename", "-f",
                                               default=None)
        self.subparsers['report'].add_argument("--sort-by", "-s",
                                               default="name")
        self.subparsers['report'].add_argument("--pipe", "-p",
                                               default=None)
        self.subparsers['report'].add_argument("--config", "-c",
                                               nargs='*', default=[])



        self.subparsers['rerun'].add_argument("experiment_id")
        self.subparsers['remove'].add_argument("experiment_id", nargs='+')
        self.subparsers['report'].add_argument("experiment_id", nargs='*')
        self.subparsers['inspect'].add_argument("experiment_id")


    def get_subparser(self, name):
        return self.subparsers[name]

    def parse_args(self):
        return self.parser.parse_args()

    def run(self):
        args = self.parse_args()

        if args.command == "report":
            config = {key: autotype(value)
                      for key, value in
                      (item.split(":") for item in args.config)}
            print(config)

            data, headers = tabulate_data(get_experiments(args.experiment_id,
                                                          args.pipe, config),
                                          args.sort_by)

            if args.output_format == "csv":
                save_as_csv(data, args.filename, headers)
            else:
                pretty_print(data, headers)
            sys.exit(0)
        elif args.command == "inspect":
            inspect_experiment(args.experiment_id)
            sys.exit(0)
        elif args.command == "rerun":
            long_hash = get_long_hash(args.experiment_id)
            args.config = os.path.join(get_ckpt_path(), "experiments",
                                       long_hash)
            args.dry_run = False
            args.checkpoints = True
            args.n = None
        elif args.command == "remove":
            for ex_id in args.experiment_id:
                filename = get_long_hash(ex_id)
                if filename:
                    remove_experiment(filename)
            sys.exit(0)

        return args
