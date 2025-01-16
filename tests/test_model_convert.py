from drAIn import models
import pytest
import os

path_tf = "tests/static/dummy_model.keras"
precs = ['float32', 'float16', 'int8']

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


# TODO: custom calibration testing