from piper import Experiment
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

(x_train, y_train), (x_test, y_test) = load_data()

with Experiment("mnist", config) as ex:
    with ex.checkpoint("model") as checkpoint:
        filename = checkpoint.get_path()

        if os.path.exists(filename):
            model = keras.models.load_model(filename)
        else:
            model = train(x_train, y_train, config)
            model.save(filename)

    loss, accuracy = evaluate(model, x_test, y_test, config)

    ex.add_metric("loss", loss)
    ex.add_metric("precision", accuracy)
