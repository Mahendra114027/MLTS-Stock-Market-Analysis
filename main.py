from mlts.models import LSTM
from mlts.config import *
import pandas as pd


def run():
    # Load the historical stock prices for AAPL
    df = pd.read_csv('mlts/static/datasets/AAPL.csv')
    
    my_model = LSTM()
    my_model.fit(df)


if __name__ == '__main__':
    print(DatasetPath.APPLE.value)
    print(ModelPath.LSTM.value)
