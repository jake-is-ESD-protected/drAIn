from abc import ABC
import os
import numpy as np

class CModelGenerator:
    @classmethod
    def make(cls, path: str, engine: str, preproc=None, postproc=None, arch="frompath") -> "CModelBase":
        if engine not in cls.get_supported_engines():
            raise ValueError(f"Unsupported engine <{engine}>. Use {cls.get_supported_engines()}.")
        for e, c in cls.get_supported_classes().items():
            if e == engine:
                m = c(path, preproc, postproc, arch)
        m.engine = engine
        return m
    
    @classmethod
    def get_supported_engines(cls) -> list:
        return ['tf', 'tflite']

    @classmethod
    def get_supported_classes(cls) -> dict:
        return dict(zip(cls.get_supported_engines(), [CModelTF, CModelTFLite]))
        

class CModelBase(ABC):
    def __init__(self, path: str) -> None:
        self.name = self.__class__.__name__
        self.path = path
        self.trained = False
        if os.path.exists(self.path):
            self.trained = True
        self.engine = None
        self.arch = "frompath"
        self.in_shape = None
        self.out_shape = None
        self.prec = "Training native"
        self.pre = self._pre
        self.post = self._post
    
    def _load_saved(self):
        pass

    def _pre(self, data):
        return data

    def _inf(self, data):
        raise NotImplementedError("Inference not yet defined.")

    def _post(self, data):
        return data
    
    
class CModelUser(CModelBase):
    def __init__(self, path: str, engine: str, preproc=None, postproc=None, arch="frompath") -> None:
        super().__init__(path)
        self.engine = engine
        if preproc:
            self.pre = preproc
        if postproc:
            self.post = postproc
        if arch != "frompath":
            if self.trained:
                raise ValueError("A model architecture was given, yet a saved model exists at the given path.")
        self.arch = arch
        
    def load():
        pass


class CModelTF(CModelUser):
    def __init__(self, path: str, preproc=None, postproc=None, arch="frompath") -> None:
        super().__init__(path, 'tf', preproc, postproc, arch)
        import tensorflow as tf # type: ignore
        import keras # type: ignore
        self.tf = tf
        self.keras = keras
    
    def _load_saved(self):
        model = self.tf.keras.models.load_model(self.path)
        return model


class CModelTFLite(CModelUser):
    def __init__(self, path: str, preproc=None, postproc=None, arch="frompath") -> None:
        if not isinstance(arch, str):
            raise ValueError("TFLite model has a fixed architecture. Consider loading a TF model instead.")
        super().__init__(path, 'tflite', preproc, postproc)
        if not self.trained:
            raise FileNotFoundError(f"No TFLite model found at <{self.path}>.")
        import tflite_runtime as tflite # type: ignore
        self.tflite = tflite
        self.delegate = None
    
    def _load_saved(self):
        self.interpreter = self.tflite.Interpreter(model_path=self.path, 
                                                   experimental_delegates=self.delegate, 
                                                   num_threads=4)
        self.interpreter.allocate_tensors()
        self.in_info = self.interpreter.get_input_details()
        self.out_info = self.interpreter.get_output_details()
        self.in_shape = self.in_info[0]['shape']
        self.out_shape = self.out_info[0]['shape']
        self.batch_size = self.in_shape[0]
        self.prec = self.in_info[0]['dtype']
        if self.prec == np.int8:
            self.in_scale, self.in_zero_point = self.in_info[0]['quantization']
            self.out_scale, self.out_zero_point = self.out_info[0]['quantization']
    