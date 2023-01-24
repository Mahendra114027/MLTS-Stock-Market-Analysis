from mlts.models import LSTM
import pandas as pd


def run():
    # Load the historical stock prices for AAPL
    df = pd.read_csv('mlts/datasets/AAPL.csv')
    
    my_model = LSTM()
    my_model.fit(df)


if __name__ == '__main__':
    run()
