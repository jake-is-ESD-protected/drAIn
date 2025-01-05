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


def pre_custom(data):
    data = np.asarray(data)
    return data * 10


def post_custom(data):
    data = np.asarray(data)
    return data / 5
    

def test_model_base_init():
    # test for model that does not exist yet
    m = models.CModelBase(path_tf_fake)
    assert m.name == models.CModelBase.__name__
    assert m.path == path_tf_fake
    assert m.engine == None
    assert m.arch == "frompath"
    assert m.trained == False
    assert m.in_shape == None
    assert m.out_shape == None
    assert m.prec == "Training native"
    assert m.pre == m._pre
    assert m.post == m._post

    # test for existing model
    m = models.CModelBase(path_tf_real)
    assert m.trained == True


def test_model_user_init():
    # test for model that does not exist yet
    m = models.CModelUser(path_tf_fake,
                          engine='tf',
                          preproc=pre_custom,
                          postproc=post_custom)
    assert m.name == models.CModelUser.__name__
    assert m.pre == pre_custom
    assert m.post == post_custom
    assert m.trained == False
    assert m.arch == "frompath"

    # test for model that does not exist yet,
    # but with a given architecture
    m = models.CModelUser(path_tf_fake,
                          engine='tf',
                          preproc=pre_custom,
                          postproc=post_custom,
                          arch=model_arch)
    assert m.trained == False
    assert isinstance(m.arch, Sequential)

    # test for existing model
    m = models.CModelUser(path_tf_real,
                          engine='tf',
                          preproc=pre_custom,
                          postproc=post_custom)
    assert m.trained == True
    assert m.arch == "frompath"

    # test for existing model + arch (Exception)
    with pytest.raises(ValueError) as e:
        m = models.CModelUser(path_tf_real,
                            engine='tf',
                            preproc=pre_custom,
                            postproc=post_custom,
                            arch=model_arch)


def test_model_generator_engines():
    targets = ['tf', 'litert']
    engs = models.CModelGenerator.get_supported_engines()
    assert len(targets) == len(engs)
    assert sorted(targets) == sorted(engs)

    clss = models.CModelGenerator.get_supported_classes()
    assert len(targets) == len(clss)
    for target, key in zip(targets, clss.keys()):
        assert target == key
    assert models.CModelTF in clss.values()
    assert models.CModelLiteRT in clss.values()


def test_model_generator_tf():
    # test for model that does not exist yet
    m = models.CModelGenerator.make(path=path_tf_fake,
                                    engine='tf',
                                    preproc=pre_custom,
                                    postproc=post_custom)
    assert m.name == models.CModelTF.__name__
    assert m.path == path_tf_fake
    assert m.engine == 'tf'
    assert m.trained == False
    assert m.in_shape == None
    assert m.out_shape == None
    assert m.prec == "Training native"
    assert m.pre == pre_custom
    assert m.post == post_custom
    assert m.arch == "frompath"

    # test for model that does not exist yet,
    # but with a given architecture
    m = models.CModelGenerator.make(path=path_tf_fake,
                                    engine='tf',
                                    preproc=pre_custom,
                                    postproc=post_custom,
                                    arch=model_arch)
    assert m.trained == False
    assert isinstance(m.arch, Sequential) 

    # test for existing model
    m = models.CModelGenerator.make(path=path_tf_real,
                                    engine='tf',
                                    preproc=pre_custom,
                                    postproc=post_custom)
    assert m.trained == True
    assert m.arch == "frompath"

    # test for existing model + arch (Exception)
    with pytest.raises(ValueError) as e:
        m = models.CModelGenerator.make(path=path_tf_real,
                                        engine='tf',
                                        preproc=pre_custom,
                                        postproc=post_custom,
                                        arch=model_arch)


def test_model_generator_litert():
    # test for existing model
    m = models.CModelGenerator.make(path=path_litert_real,
                                    engine='litert',
                                    preproc=pre_custom,
                                    postproc=post_custom)
    assert m.name == models.CModelLiteRT.__name__
    assert m.path == path_litert_real
    assert m.engine == 'litert'
    assert m.trained == True
    assert m.in_shape == None
    assert m.out_shape == None
    assert m.prec == "Training native"
    assert m.pre == pre_custom
    assert m.post == post_custom

    # test for model that does not exist yet (Exception)
    with pytest.raises(FileNotFoundError) as e:
        m = models.CModelGenerator.make(path=path_litert_fake,
                                        engine='litert',
                                        preproc=pre_custom,
                                        postproc=post_custom)
    
    # test for arch override (Exception)
    with pytest.raises(ValueError) as e:
        m = models.CModelGenerator.make(path=path_litert_fake,
                                        engine='litert',
                                        preproc=pre_custom,
                                        postproc=post_custom,
                                        arch=model_arch)