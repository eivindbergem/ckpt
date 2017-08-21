from unittest import TestCase
from tempfile import mkdtemp

import shutil
import os.path
import json

from piper.experiment import Experiment, Checkpoint
from piper.misc import set_piper_path, get_piper_path, mkdirp

@Checkpoint.save
def save(ckpt, filename, data):
    with open(ckpt.join_path(filename), "w") as fd:
        fd.write(data)

@Checkpoint.load
def load(ckpt, filename):
    with open(ckpt.join_path(filename)) as fd:
        data = fd.read()

    return data

class TestPiper(TestCase):
    def setUp(self):
        self.path = mkdtemp()
        set_piper_path(self.path)

    def tearDown(self):
        shutil.rmtree(self.path)

    def test_misc(self):
        filename = "foo/bar/42"

        mkdirp(filename)

        self.assertTrue(os.path.exists(filename))

    def test_experiment(self):
        config = {"test": "test"}

        self.assertEqual(self.path, get_piper_path())

        with Experiment("test", config) as ex:
            self.assertEqual(ex.get_path(),
                             os.path.join(self.path,
                                          "experiments",
                                          "test"))

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

            filename = ex.get_filename()

        with open(filename) as fd:
            data = json.load(fd)

        self.assertEqual(data['metrics']['test'], 42)
