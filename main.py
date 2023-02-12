from mlts.factory import DatasetFactory, ModelFactory, PreprocessorFactory
import argparse


def run(model, dataset_name):
    # Object instantiation
    preprocessor_factory = PreprocessorFactory()
    dataset_factory = DatasetFactory()
    model_factory = ModelFactory()
    
    # Load the historical stock prices for AAPL
    stock_data = dataset_factory.get(dataset_name)
    
    # Preprocess dataset
    preprocessor = preprocessor_factory.get('stock')
    preprocessed_stock_data = preprocessor.preprocess(
        stock_data,
        save=False,  # To save the preprocessed data on File system
        dataset=dataset_name  # Identifier for the dataset
    )
    
    # Get model
    my_model = model_factory.get(model)
    
    # Train model
    my_model.fit(preprocessed_stock_data, dataset=dataset_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')  # LSTM/lstm, XGB/xgb, ARIMA/arima
    parser.add_argument('--dataset')  # AAPL/aapl, GMBL/gmbl, TSLA/tsla
    
    args = parser.parse_args()
    input_model = args.model.upper()
    input_dataset = args.dataset.upper()
    
    # run(input_model, input_dataset)
    
    run('XGB', 'AAPL')
    run('LSTM', 'AAPL')
    run('ARIMA', 'AAPL')
    
    run('XGB', 'TSLA')
    run('LSTM', 'TSLA')
    run('ARIMA', 'TSLA')
    
    run('XGB', 'GMBL')
    run('LSTM', 'GMBL')
    run('ARIMA', 'GMBL')
