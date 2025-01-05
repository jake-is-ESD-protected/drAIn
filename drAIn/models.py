from abc import ABC, abstractmethod
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
        return ['tf', 'litert']

    @classmethod
    def get_supported_classes(cls) -> dict:
        return dict(zip(cls.get_supported_engines(), [CModelTF, CModelLiteRT]))
        

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

    def load(self):
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


class CModelTF(CModelUser):
    def __init__(self, path: str, preproc=None, postproc=None, arch="frompath") -> None:
        super().__init__(path, 'tf', preproc, postproc, arch)
        import tensorflow as tf # type: ignore
        import keras # type: ignore
        self.tf = tf
        self.keras = keras
        self.__mtf = None
    
    def __load_saved(self):
        model = self.tf.keras.models.load_model(self.path)
        return model
    
    def __build_arch(self, **kwargs_compile):
        self.arch.compile(**kwargs_compile)
        return self.arch
    
    def load(self, **kwargs):
        if self.trained:
            self.__mtf = self.__load_saved()
        else:
            self.__mtf = self.__build_arch(**kwargs)
        self.in_shape = tuple(self.__mtf.layers[0].input.shape.as_list())
        self.out_shape = tuple(self.__mtf.layers[-1].output.shape.as_list())


class CModelLiteRT(CModelUser):
    def __init__(self, path: str, preproc=None, postproc=None, arch="frompath") -> None:
        if not isinstance(arch, str):
            raise ValueError("LiteRT model has a fixed architecture. Consider loading a TF model instead.")
        super().__init__(path, 'litert', preproc, postproc)
        if not self.trained:
            raise FileNotFoundError(f"No LiteRT model found at <{self.path}>.")
        import ai_edge_litert.interpreter as litert # type: ignore
        self.litert = litert
        self.delegate = None
    
    def load(self):
        self.interpreter = self.litert.Interpreter(model_path=self.path, 
                                                               experimental_delegates=self.delegate, 
                                                               num_threads=4)
        self.interpreter.allocate_tensors()
        self.in_info = self.interpreter.get_input_details()
        self.out_info = self.interpreter.get_output_details()
        self.in_shape = tuple(self.in_info[0]['shape'])
        self.out_shape = tuple(self.out_info[0]['shape'])
        self.batch_size = self.in_shape[0]
        self.prec = self.in_info[0]['dtype']
        if self.prec == np.int8:
            self.in_scale, self.in_zero_point = self.in_info[0]['quantization']
            self.out_scale, self.out_zero_point = self.out_info[0]['quantization']
    