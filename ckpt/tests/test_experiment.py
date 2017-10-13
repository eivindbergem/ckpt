from unittest import TestCase
from tempfile import mkdtemp

import shutil
import os.path
import json

from ckpt.experiment import Experiment, Checkpoint
from ckpt.misc import set_ckpt_path, get_ckpt_path, mkdirp

@Checkpoint.save
def save(ckpt, filename, data):
    with open(ckpt.join_path(filename), "w") as fd:
        fd.write(data)

@Checkpoint.load
def load(ckpt, filename):
    with open(ckpt.join_path(filename)) as fd:
        data = fd.read()

    return data

class TestCkpt(TestCase):
    def setUp(self):
        self.path = mkdtemp()
        set_ckpt_path(self.path)

    def tearDown(self):
        shutil.rmtree(self.path)

    def test_misc(self):
        filename = "foo/bar/42"

        mkdirp(filename)

        self.assertTrue(os.path.exists(filename))

        while filename:
            os.rmdir(filename)

            filename = os.path.dirname(filename)

    def test_experiment(self):
        config = {"test": "test"}

        self.assertEqual(self.path, get_ckpt_path())

        with Experiment("test", config) as ex:
            self.assertEqual(ex.get_path(),
                             os.path.join(self.path,
                                          "experiments",
                                          "test",
                                          str(ex.timestamp)))

            with ex.add_checkpoint("test", config) as ckpt:
                self.assertFalse(ckpt.exists())

                filename = "test"
                data = "test\n"

                save(ckpt, filename, data)

                self.assertTrue(ckpt.exists())

                self.assertEqual(load(ckpt, filename),
                                 data)

            with ex.add_checkpoint("test", config) as ckpt:
                self.assertNotEqual(ckpt.get_path(),
                                    ckpt.prev_checkpoint.get_path())

            ex.add_metrics({"test": 42})

        with open(ex.join_path("metrics.json")) as fd:
            data = json.load(fd)

        self.assertEqual(data['test'], 42)

        self.assertCountEqual(os.listdir(ex.get_path()),
                              ["log", "metrics.json",
                               "config.json", "checkpoints.json"])
