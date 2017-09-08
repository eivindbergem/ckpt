from piper import Experiment, Checkpoint
import keras.models

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

import os.path

config = {"init": "uniform",
          "activation": "relu",
          "hidden-size": 25,
          "optimizer": "rmsprop",
          "epochs": 2,
          "batch_size": 32}

def train(x, y, config):
    model = Sequential([
        Dense(config['hidden-size'],
              input_dim=784, kernel_initializer=config['init'],
              activation=config['activation']),
        Dense(10, kernel_initializer=config['init'], activation="softmax")
    ])

    model.compile(optimizer=config['optimizer'],
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x, y,
              epochs=config['epochs'],
              batch_size=config['batch_size'])

    return model

def evaluate(model, x, y, config):
    return model.evaluate(x, y,
                          batch_size=config['batch_size'])

def reshape_data(x, y):
    return x.reshape((-1, 784)), to_categorical(y)

def load_data():
    return [reshape_data(x, y) for x, y in mnist.load_data()]

@Checkpoint.load
def load(ckpt, filename):
    return keras.models.load_model(filename)

@Checkpoint.save
def save(ckpt, model, filename):
    model.save(filename)

(x_train, y_train), (x_test, y_test) = load_data()

with Experiment("mnist", config) as ex:
    with ex.add_checkpoint("model", config) as ckpt:
        filename = ckpt.join_path("model.h5")

        if ckpt.exists():
            model = load(ckpt, filename)
        else:
            model = train(x_train, y_train, config)
            save(ckpt, model, filename)

    loss, accuracy = evaluate(model, x_test, y_test, config)

    ex.add_metrics({"loss": loss,
                    "precision": accuracy})
