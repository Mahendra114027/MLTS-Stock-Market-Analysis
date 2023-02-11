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
        save=True,
        dataset=dataset_name
    )
    
    # Get model
    my_model = model_factory.get(model)
    
    # Train model
    my_model.fit(preprocessed_stock_data, dataset=dataset_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')  # LSTM, XGB, ARIMA
    parser.add_argument('--dataset')  # AAPL, GMBL, TSLA
    
    args = parser.parse_args()
    input_model = args.model
    input_dataset = args.dataset
    
    # run(input_model, input_dataset)
    run('ARIMA', 'AAPL')
    run('ARIMA', 'GMBL')
    run('ARIMA', 'TSLA')
