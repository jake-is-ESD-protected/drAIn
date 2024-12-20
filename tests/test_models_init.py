from drAIn import models
import numpy as np

path_tf_fake = "tests/static/test_model.keras"
path_tflite_fake = "tests/static/test_model.tflite"


def pre_custom(data):
    data = np.asarray(data)
    return data * 10


def post_custom(data):
    data = np.asarray(data)
    return data / 5
    

def test_model_base_init():
    m = models.CModelBase(path_tf_fake)
    assert m.name == models.CModelBase.__name__
    assert m.path == path_tf_fake
    assert m.engine == None
    assert m.in_shape == None
    assert m.out_shape == None
    assert m.prec == "Training native"
    assert m.pre == m._pre
    assert m.post == m._post


def test_model_user_init():
    m = models.CModelUser(path_tf_fake,
                          engine='tf',
                          preproc=pre_custom,
                          postproc=post_custom)
    assert m.name == models.CModelUser.__name__
    assert m.pre == pre_custom
    assert m.post == post_custom


def test_model_generator_engines():
    targets = ['tf', 'tflite']
    engs = models.CModelGenerator.get_supported_engines()
    assert len(targets) == len(engs)
    assert sorted(targets) == sorted(engs)

    clss = models.CModelGenerator.get_supported_classes()
    assert len(targets) == len(clss)
    for target, key in zip(targets, clss.keys()):
        assert target == key
    assert models.CModelTF in clss.values()
    assert models.CModelTFLite in clss.values()


def test_model_generator_tf():
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


def test_model_generator_tflite():
    m = models.CModelGenerator.make(path=path_tflite_fake,
                                    engine='tflite',
                                    preproc=pre_custom,
                                    postproc=post_custom)
    assert m.name == models.CModelTFLite.__name__
    assert m.path == path_tflite_fake
    assert m.engine == 'tflite'
    assert m.trained == False
    assert m.in_shape == None
    assert m.out_shape == None
    assert m.prec == "Training native"
    assert m.pre == pre_custom
    assert m.post == post_custom