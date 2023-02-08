from datetime import datetime as dt
from enum import Enum
import os


class Path(Enum):
    """
    Path
    """
    
    ROOT = os.path.abspath('./mlts')


class RawDataset(Enum):
    """
    Original Dataset
    """
    
    DATE_FEATURES = ['Date']
    AAPL = os.path.join(Path.ROOT.value, 'static/datasets/original/aapl.csv')
    GMBL = os.path.join(Path.ROOT.value, 'static/datasets/original/gmbl.csv')
    TSLA = os.path.join(Path.ROOT.value, 'static/datasets/original/tsla.csv')


class Preprocess(Enum):
    """
    Preprocessing Config
    """
    
    DROP_FEATURES = ['open', 'high', 'low', 'close']
    NUM_DAYS = 3  # After several iterations, 3 days is the best


class PreprocessedDataset(Enum):
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
    XGB = os.path.join(Path.ROOT.value, f"static/checkpoints/xgb/{dt.now().strftime('%Y%m%d_%H%M_')}xgb.h5")
    LSTM = os.path.join(Path.ROOT.value, f"static/checkpoints/lstm/{dt.now().strftime('%Y%m%d_%H%M_')}lstm.h5")
    ARIMA = os.path.join(Path.ROOT.value, f"static/checkpoints/arima/{dt.now().strftime('%Y%m%d_%H%M_')}arima.pkl")


class ModelParams(Enum):
    """
    Model Parameters
    """
    
    TARGET = 'adj_close'
    EPOCHS = 1
    VERBOSE = 2
    BATCH_SIZE = 1
