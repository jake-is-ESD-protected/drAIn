from abc import ABC
import os
import numpy as np
import datetime
from .utils import printDrAIn
import inspect

class CModelGenerator:
    @classmethod
    def make(cls, path: str, engine: str, preproc=None, postproc=None, arch="frompath") -> "CModelBase":
        """
        Create a drAIn model from a saved model path or Keras architecture.

        Parameters
        ----------
        path : str
            Path to the model. If the model has yet to be built, the given string will
            be used as save path for later saving.
        engine : str
            String describing the framework or engine. Can be ['tf', 'litert'].
        preproc : callable
            Preprocessing function. Can be defined anywhere. Is called in `inference()`.
        preproc : callable
            Postprocessing function. Can be defined anywhere. Is called in `inference()`.
        arch : Sequential
            Model architecture. If the model is saved, the architecture is determined from the
            loaded model.
        """
        if engine not in cls.get_supported_engines():
            raise ValueError(f"Unsupported engine <{engine}>. Use {cls.get_supported_engines()}.")
        for e, c in cls.get_supported_classes().items():
            if e == engine:
                m = c(path, preproc, postproc, arch)
        m.engine = engine
        return m
    
    @classmethod
    def get_supported_engines(cls) -> list:
        """
        Get the supported inference enfines.
        """
        return ['tf', 'litert']

    @classmethod
    def get_supported_classes(cls) -> dict:
        """
        Get the supported model classes fitting the inference engines.
        """
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

    def train(self):
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
    """
    TensorFlow model abstraction class. Manages loading, training, converting
    and inference.

    Parameters
    ----------
    path : str
        Path to model. If it exists, the model will be loaded from there.
        If it does not exist, this path will be used to save it if an
        architecture is given and the model is trained.
    preproc : callable
        See `CModelGenerator.make()`.
    postproc : callable
        See `CModelGenerator.make()`.
    arch : Sequential
        See `CModelGenerator.make()`.
    
    Notes
    -----
    Do not instanciate this class directly. Use CModelGenerator.make() instead.
    """
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
        """
        Load the model abstraction. If an architecture was given, the model will be compiled.
        `**kwargs` are passed to `compile()` of the standard TensorFlow framework. 
        `verbose` can also be set to `True` to obtain further loading information.
        """
        verbose = kwargs.pop('verbose', False)
        if self.trained:
            self.__mtf = self.__load_saved()
        else:
            self.__mtf = self.__build_arch(**kwargs)
        self.in_shape = tuple(self.__mtf.layers[0].input.shape.as_list())
        self.out_shape = tuple(self.__mtf.layers[-1].output.shape.as_list())
        self.input_name = self.__mtf.input.name
        if verbose:
            self.__mtf.summary()
            printDrAIn(f"This model is trained: {self.trained}")
    
    def train(self, data, valid_split=0.2, test_split=0.05, epochs=100, batch=1, callbacks='default', **kwargs):
        """
        Train the model based on the given architecture.

        Parameters
        ----------
        data : np.ndarray | list
            Array of all available data. Is expected to have a top dimension shape of 2, where
            the first dimension is the input data and the second are the ground truths to the
            input data.
        valid_split : float
            Validation split size factor relative to entire data set.
        test_split : float
            Test split size factor relative to entire data set.
        epochs : int
            Number of training epochs as understood by TensorFlow.
        batch : int
            Data batch size (number of parallel data point inputs)
        callbacks : list[tf.keras.callbacks]
            List of TensorFlow callbacks such as TensorBoard or EarlyStopping.
            Use `'default'` for `TensorBoard` and `EarlyStopping`.
        
        Returns
        
        Notes
        -----
        `verbose` can also be set to `True` to obtain further training information.
        """
        if not self.__mtf:
            raise RuntimeError("No model loaded. Did you call `your_model.load()`?")
        if valid_split + test_split > 1.0:
            raise ValueError(f"A validation split of {valid_split} and a test split of {test_split} \
                             does not leave data for training! Lower the percentages.")
        if len(data) != 2:
            raise ValueError(f"drAIn expects training data to be of shape (2, ...), where dimension \
                             0 is the input data and 1 is the ground truth to that input. Your shape \
                             was {len(data)}!")
        
        verbose = kwargs.pop('verbose', False)
        if callbacks == 'default':
            now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = "logs/" + os.path.basename(self.path) + '_' + now
            tensorboard_callback = self.tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                                       histogram_freq=1)
            early_stopping_callback = self.tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                                            patience=10, 
                                                                            restore_best_weights=True)
            callbacks = [tensorboard_callback, early_stopping_callback]
            if verbose:
                printDrAIn(f"Using default callbacks {callbacks}")
                printDrAIn(f"Storing logs to {log_dir}")
        
        elif callbacks == None:
            if verbose:
                printDrAIn(f"No callbacks registered.")
        else:
            printDrAIn(f"Using callbacks {callbacks}")

        from sklearn.model_selection import train_test_split # type: ignore
        xtrain, xval, ytrain, yval = train_test_split(data[0],
                                                      data[1], 
                                                      train_size=1-valid_split, 
                                                      test_size=valid_split, 
                                                      random_state=69420)
        xtrain, xtest, ytrain, ytest = train_test_split(xtrain,
                                                        ytrain, 
                                                        train_size=1-test_split, 
                                                        test_size=test_split, 
                                                        random_state=69420)
        hist = self.__mtf.fit(x=xtrain,
                              y=ytrain,
                              validation_data=(xval, yval),
                              epochs=epochs,
                              callbacks=callbacks,
                              batch_size=batch)
        
        train_metrics = self.__mtf.evaluate(xtrain, ytrain, verbose=0)
        val_metrics = self.__mtf.evaluate(xval, yval, verbose=0)
        test_metrics = self.__mtf.evaluate(xtest, ytest, verbose=0)

        self.in_shape = tuple(dim for dim in self.in_shape if dim is not None)
        self.batch_size = batch
        zero_data = np.zeros((batch, *self.in_shape))
        zero_return = self.__mtf.predict(zero_data)

        self.trained = True
        self.__mtf.save(self.path)

        return CModelTrainingResult(hist, train_metrics, val_metrics, test_metrics, zero_return)
    
    def inference(self, data):
        """
        Infer a data point. Calls the given preprocessing and postprocessing function 
        from the inside.

        Parameters
        ----------
        data : np.ndarray
            Data as expected by the preprocessing function.
        
        Returns
        -------
        Neural network output AFTER postprocessing.
        """
        if not self.__mtf:
            raise RuntimeError("No model loaded. Did you call `your_model.load()`?")
        data = self.pre(np.asarray(data))
        if data.shape[1:] != self.in_shape[1:]:
            raise ValueError(f"Data has shape {data.shape[1:]} while the model input\
                             is of shape {self.in_shape[1:]}! (ignoring batch size)")
        self.batch_size = data.shape[0]
        pred = self.__mtf.predict(data)
        return self.post(pred)

    def convert(self, prec, **kwargs):
        converter = self.tf.lite.TFLiteConverter.from_keras_model(self.__mtf)
        converter.optimizations = [self.tf.lite.Optimize.DEFAULT]

        precs_tf = dict({'float32': self.tf.float32, 
                         'float16': self.tf.float16, 
                         'int8': self.tf.int8})

        if prec not in list(precs_tf.keys()):
            raise ValueError(f"Quantization <{prec}> unknown. Use one of the following: {list(precs_tf.keys())}.")

        if prec == "int8":
            calibration = kwargs.get('calibration', self.__calib_random)
            if not inspect.isgenerator(calibration()):
                raise ValueError("The given calibration function does not return a generator!")
            self.n = kwargs.get('n', 100)
            self.rng = kwargs.get('rng', [0, 1])
            datapoint = next(calibration())
            if self.input_name not in datapoint.keys():
                raise ValueError(f"Your calibration function needs to return datapoints with the key\
                                 {self.input_name}, yours has key {datapoint.keys()}.")
            if np.asarray(datapoint[self.input_name]).dtype != np.float32:
                raise ValueError("Your custom calibration does not yield the required **float32** data!")
            data_shape = np.asarray(datapoint[self.input_name]).shape
            if data_shape != self.in_shape[1:]:
                raise RuntimeError(f"Data shape {data_shape[1:]} and model input {self.in_shape} do not match!")
            npseed = kwargs.get('seed', None)
            if npseed:
                np.random.seed(npseed)
            converter.representative_dataset = calibration
            converter.target_spec.supported_ops = [self.tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = self.tf.int8
            converter.inference_output_type = self.tf.int8
            
        converter.target_spec.supported_types = [precs_tf[prec]]
        q_model = converter.convert()
        path = self.path.split('.')[-2] + f"_{prec}" + ".tflite"
        with open(path, 'wb') as f:
            f.write(q_model)
        return CModelLiteRT(path, self.pre, self.post)
    
    def __calib_random(self):
        data = np.random.uniform(low=self.rng[0], high=self.rng[1], size=(self.n, *self.in_shape[1:]))
        for point in data:
            point = self.pre(point)
            yield {self.input_name: point.astype(np.float32)}
        


class CModelLiteRT(CModelUser):
    """
    LiteRT model abstraction class. Manages loading and inference.

    Parameters
    ----------
    path : str
        Path to model. If it exists, the model will be loaded from there.
        If it does not exist, this path will be used to save it if an
        architecture is given and the model is trained.
    preproc : callable
        See `CModelGenerator.make()`.
    postproc : callable
        See `CModelGenerator.make()`.
    arch : Sequential
        See `CModelGenerator.make()`.
    
    Notes
    -----
    Do not instanciate this class directly. Use CModelGenerator.make() instead.
    """
    def __init__(self, path: str, preproc=None, postproc=None, arch="frompath") -> None:
        if not isinstance(arch, str):
            raise ValueError("LiteRT model has a fixed architecture. Consider loading a TF model instead.")
        super().__init__(path, 'litert', preproc, postproc)
        if not self.trained:
            raise FileNotFoundError(f"No LiteRT model found at <{self.path}>.")
        import ai_edge_litert.interpreter as litert # type: ignore
        self.interpreter = None
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
        
    def inference(self, data):
        if not self.interpreter:
            raise RuntimeError("No model loaded. Did you call `your_model.load()`?")
        data = self.pre(np.asarray(data))
        if data.shape != self.in_shape:
            raise ValueError(f"Data has shape {data.shape} while the model input\
                             is of shape {self.in_shape}!")
        data = self.__scale_in_prec(data)
        if self.batch_size == 1 and data.shape[0] != self.batch_size:
            data = np.expand_dims(data, axis=0) # fixes the 1 batchsize issue
        self.interpreter.set_tensor(self.in_info[0]['index'], data)
        self.interpreter.invoke()
        results = []
        if len(self.out_info) > 1:
            for t in self.out_info:
                results.append(self.interpreter.get_tensor(t['index']))
        else:
            results = self.interpreter.get_tensor(self.out_info[0]['index'])
        return self.post(self.__scale_out_prec(np.asarray(results.copy())))
    
    def __scale_in_prec(self, data):
        if self.prec == np.int8:
            return np.int8(data / self.in_scale + self.in_zero_point)
        else:
            return data
    
    def __scale_out_prec(self, data):
        if self.prec == np.int8:
            return (data.astype(np.float32) - self.out_zero_point) * self.out_scale
        else:
            return data


class CModelTrainingResult:
    def __init__(self, history, train_metrics, val_metrics, test_metrics, zero_return) -> None:
        self.history = history
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.zero_return = zero_return