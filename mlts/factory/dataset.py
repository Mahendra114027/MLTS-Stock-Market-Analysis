from mlts.config import DatasetPath
from mlts.factory import Factory
import pandas as pd


class DatasetFactory(Factory):
    """
    Dataset Factory
    """
    
    def get(self, name, *args, **kwargs):
        print('Loading dataset: {}'.format(name))
        return pd.read_csv(DatasetPath[name].value, parse_dates=['date'], index_col='date')
