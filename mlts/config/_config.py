from enum import Enum
import os


class Path(Enum):
    """
    Path
    """
    
    ROOT = os.path.abspath('./mlts')


class DatasetPath(Enum):
    """
    Dataset Paths
    """
    
    AAPL = os.path.join(Path.ROOT.value, 'static/datasets/preprocessed_aapl.csv')
    GMBL = os.path.join(Path.ROOT.value, 'static/datasets/preprocessed_gmbl.csv')
    TSLA = os.path.join(Path.ROOT.value, 'static/datasets/preprocessed_tsla.csv')


class ModelPath(Enum):
    """
    Model Paths
    """
    
    XGB = os.path.join(Path.ROOT.value, 'static/checkpoints/xgb/xgb.h5')
    LSTM = os.path.join(Path.ROOT.value, 'static/checkpoints/lstm/lstm.h5')
    ARIMA = os.path.join(Path.ROOT.value, 'static/checkpoints/arima/arima.h5')


class ModelParams(Enum):
    """
    Model Parameters
    """
    
    TARGET = 'adj_close'
    EPOCHS = 1
    VERBOSE = 2
    BATCH_SIZE = 1
