from drAIn import models
import pytest
import numpy as np
from keras import Sequential
import tensorflow as tf

path_tf_fake = "tests/static/test_model.keras"
path_litert_fake = "tests/static/test_model.tflite"
path_tf_real = "tests/static/dummy_model.keras"
path_litert_real = "tests/static/dummy_model.tflite"
in_shape = 10
out_shape = 4
model_arch = Sequential([
        tf.keras.layers.InputLayer(input_shape=(in_shape,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(out_shape, activation='sigmoid')
    ])


def test_model_tf_load():
    # test non-existing model with arch
    m = models.CModelGenerator.make(path=path_tf_fake,
                                    engine='tf',
                                    arch=model_arch)
    m.load()
    # TF dynamic batch sizes are expressed as None
    assert m.in_shape == (None, in_shape)
    assert m.out_shape == (None, out_shape)

    # test existing model
    m = models.CModelGenerator.make(path=path_tf_real,
                                    engine='tf')
    m.load()
    assert m.in_shape == (None, in_shape)
    assert m.out_shape == (None, out_shape)


def test_model_litert_load():
    m = models.CModelGenerator.make(path=path_litert_real,
                                    engine='litert')
    m.load()
    # if TF batch size is None, then the LiteRT batch size is 1
    assert m.in_shape == (1, in_shape)
    assert m.out_shape == (1, out_shape)