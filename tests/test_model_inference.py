from drAIn import models
import numpy as np
from numpy import testing as npt

path_tf = "tests/static/dummy_model.keras"
path_litert = "tests/static/dummy_model_int8.tflite"


def test_model_inference_shape():
    m = models.CModelGenerator.make(path=path_tf,
                                    engine='tf')
    m.load()
    assert isinstance(m, models.CModelTF)
    data = np.zeros((100, *m.in_shape[1:]))
    assert data.shape[1:] == m.in_shape[1:]
    res = m.inference(data)
    assert isinstance(res, np.ndarray)
    assert m.batch_size == data.shape[0]
    assert res.shape[1:] == m.out_shape[1:]


def test_model_inference_litert_shape():
    m = models.CModelGenerator.make(path=path_litert,
                                    engine='litert')
    m.load()
    assert isinstance(m, models.CModelLiteRT)
    data = np.zeros(m.in_shape)
    res = m.inference(data)
    assert isinstance(res, np.ndarray)
    assert res.shape == m.out_shape


def test_model_inference_compare():
    mtf = models.CModelGenerator.make(path=path_tf,
                                    engine='tf')
    mtf.load()
    assert isinstance(mtf, models.CModelTF)

    mlrt = models.CModelGenerator.make(path=path_litert,
                                       engine='litert')
    mlrt.load()
    assert isinstance(mlrt, models.CModelLiteRT)

    data = np.zeros(mlrt.in_shape)
    res_tf = mtf.inference(data)
    res_lrt = mlrt.inference(data)
    print(mlrt.in_scale)
    print(mlrt.out_scale)
    print(mlrt.in_zero_point)
    print(mlrt.out_zero_point)
    assert mlrt.prec == np.int8
    npt.assert_allclose(res_tf, res_lrt)