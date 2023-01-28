from mlts.models import LSTM
from mlts.config import *
import pandas as pd


def run():
    # Load the historical stock prices for AAPL
    stock_data = pd.read_csv(
        DatasetPath.APPLE.value, parse_dates=['date'],
        index_col='date'
    )
    
    my_model = LSTM()
    my_model.fit(stock_data)


if __name__ == '__main__':
    run()
