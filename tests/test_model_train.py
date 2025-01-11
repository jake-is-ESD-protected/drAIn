from drAIn import models
import pytest
import numpy as np
from keras import Sequential
import tensorflow as tf
import os
import shutil

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

def test_model_tf_train():
    m = models.CModelGenerator.make(path=path_tf_temp,
                                    engine='tf',
                                    arch=model_arch)
    m.load(optimizer='adam', loss='mse', metrics=['accuracy'])
    input_data = np.random.randn(*(n, in_shape))
    output_data = np.random.randn(*(n, out_shape))
    data = (input_data, output_data)

    results = m.train(data=data, epochs=2)
    assert isinstance(results, models.CModelTrainingResult)
    assert results.zero_return.shape[1:] == m.out_shape[1:] # ignore batch size
    assert os.path.exists('logs')
    assert os.listdir('logs') != None
    shutil.rmtree('logs')