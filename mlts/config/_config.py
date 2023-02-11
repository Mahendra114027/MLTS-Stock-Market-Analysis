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
    
    # For LSTM
    MAX_TRIALS = 2
    
    # For XGB
    XGB_PARAMS = {
        # max_depth :Maximum tree depth for base learners
        'max_depth': range(2, 10, 1),
        
        # n_estimators: Number of boosted trees to fit
        'n_estimators': range(10, 250, 10),
        
        # learning_rate:Boosting learning rate (xgb’s “eta”)
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        
        # min_child_weight : Minimum sum of instance weight(hessian) needed in a child
        'min_child_weight': range(1, 21, 2),
        
        # subsample : Subsample ratio of the training instance
        'subsample': [0, 0.2, 0.4, 0.6, 0.8, 1],
        
        # gamma : Minimum loss reduction required to make a further partition on a leaf node of the tree
        'gamma': [0, 0.2, 0.4, 0.6, 0.8, 1],
        
        # colsample_bytree :Subsample ratio of columns when constructing each tree
        'colsample_bytree': [0, 0.2, 0.4, 0.6, 0.8, 1],
        
        # colsample_bylevel :Subsample ratio of columns for each split, in each level
        'colsample_bylevel': [0, 0.2, 0.4, 0.6, 0.8, 1]
    }
