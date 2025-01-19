from drAIn import models
import pytest
import os
import numpy as np
from keras import Sequential
import tensorflow as tf
import shutil

path_tf = "tests/static/dummy_model.keras"
precs = ['float32', 'float16', 'int8']
path_tf_temp = "tests/static/temp_model.keras"

n = 1000
in_shape = 10
out_shape = 4
model_arch = Sequential([
        tf.keras.layers.InputLayer(input_shape=(in_shape,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(out_shape, activation='sigmoid')
    ])


def custom_calib():
    data = np.random.uniform(low=0.1, high=0.9, size=(100, 10))
    for point in data:
        yield {'input_1': point.astype(np.float32)}


def invalid_calib1():
    data = np.random.uniform(low=0.1, high=0.9, size=(100, 10))
    for point in data:
        yield {'wrong_name': point.astype(np.float32)}


def invalid_calib2():
    data = np.random.uniform(low=0.1, high=0.9, size=(100, 10))
    for point in data:
        yield {'input_1': point.astype(np.float64)}


def test_model_tf_convert():
    m = models.CModelGenerator.make(path=path_tf,
                                    engine='tf')
    m.load()
    assert isinstance(m, models.CModelTF)
    for prec in precs:
        litert_model = m.convert(prec)
        litert_model.load()
        assert isinstance(litert_model, models.CModelLiteRT)
        assert os.path.exists(litert_model.path)
        assert litert_model.in_shape[1:] == m.in_shape[1:]
        os.remove(litert_model.path)


def test_model_tf_conversion_invalid_prec():
    m = models.CModelGenerator.make(path=path_tf,
                                    engine='tf')
    m.load()
    with pytest.raises(ValueError) as e:
        litert_model = m.convert("int32")


def test_model_modif_calib():
    m = models.CModelGenerator.make(path=path_tf,
                                    engine='tf')
    m.load()
    assert isinstance(m, models.CModelTF)
    litert_model = m.convert(prec='int8', n=200, rng=[0.3, 0.5])
    litert_model.load()
    assert isinstance(litert_model, models.CModelLiteRT)
    assert os.path.exists(litert_model.path)
    assert litert_model.in_shape[1:] == m.in_shape[1:]
    assert litert_model.prec == np.int8
    assert litert_model.in_zero_point != 0.0
    assert litert_model.in_scale != 0.0
    assert litert_model.out_zero_point != 0.0
    assert litert_model.out_scale != 0.0


def test_model_custom_calib():
    m = models.CModelGenerator.make(path=path_tf,
                                    engine='tf')
    m.load()
    assert isinstance(m, models.CModelTF)
    litert_model = m.convert(prec='int8', calibration=custom_calib)
    litert_model.load()
    assert isinstance(litert_model, models.CModelLiteRT)
    assert os.path.exists(litert_model.path)
    assert litert_model.in_shape[1:] == m.in_shape[1:]
    assert litert_model.prec == np.int8
    assert litert_model.in_zero_point != 0.0
    assert litert_model.in_scale != 0.0
    assert litert_model.out_zero_point != 0.0
    assert litert_model.out_scale != 0.0


def test_model_invalid_calib():
    m = models.CModelGenerator.make(path=path_tf,
                                    engine='tf')
    m.load()
    assert isinstance(m, models.CModelTF)
    with pytest.raises(ValueError) as e:
        litert_model = m.convert(prec='int8', calibration=invalid_calib1)
    with pytest.raises(ValueError) as e:
        litert_model = m.convert(prec='int8', calibration=invalid_calib2)
