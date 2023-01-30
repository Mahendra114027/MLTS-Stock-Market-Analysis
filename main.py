from mlts.models import LSTM
from mlts.config import *
import pandas as pd
import argparse


def run(dataset):
    # Load the historical stock prices for AAPL
    stock_data = pd.read_csv(DatasetPath[dataset].value, parse_dates=['date'],index_col='date')
    
    my_model = LSTM()
    my_model.fit(stock_data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset') #APPL, GMBL, TSLA
    args = parser.parse_args()
    dataset = args.dataset
    run(dataset)
