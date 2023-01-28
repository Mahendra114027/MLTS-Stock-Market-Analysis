from enum import Enum
import os


class Path(Enum):
    """
    Path
    """
    
    ROOT = os.path.abspath('../')


class DatasetPath(Enum):
    """
    Dataset Paths
    """
    
    APPLE = os.path.join(Path.ROOT.value, 'static/datasets/preprocessed_apple.csv')
    GMBL = os.path.join(Path.ROOT.value, 'static/datasets/preprocessed_gmbl.csv')
    MCD = os.path.join(Path.ROOT.value, 'static/datasets/preprocessed_mcd.csv')


class ModelPath(Enum):
    """
    Model Paths
    """
    
    XGB = os.path.join(Path.ROOT.value, 'static/checkpoints/xgb/xgb.tf')
    LSTM = os.path.join(Path.ROOT.value, 'static/checkpoints/lstm/lstm.tf')
    ARIMA = os.path.join(Path.ROOT.value, 'static/checkpoints/arima/arima.tf')
