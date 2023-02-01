from mlts.factory import DatasetFactory, ModelFactory
import argparse


def run(model, dataset):
    # Object instantiation
    dataset_factory = DatasetFactory()
    model_factory = ModelFactory()
    
    # Load the historical stock prices for AAPL
    stock_data = dataset_factory.get(dataset)
    
    # Get model
    my_model = model_factory.get(model)
    my_model.fit(stock_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')  # LSTM, XGB, ARIMA
    parser.add_argument('--dataset')  # APPL, GMBL, TSLA
    
    args = parser.parse_args()
    input_model = args.model
    input_dataset = args.dataset
    
    run(input_model, input_dataset)
