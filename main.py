from mlts.factory import DatasetFactory, ModelFactory, PreprocessorFactory
import argparse


def run(model, dataset_name):
    # Object instantiation
    preprocessor_factory = PreprocessorFactory()
    dataset_factory = DatasetFactory()
    model_factory = ModelFactory()
    
    # Load the historical stock prices for AAPL
    print('Loading dataset: {}'.format(dataset_name))
    stock_data = dataset_factory.get(dataset_name)
    
    # Preprocess dataset
    print('--->> Preprocessing input data...')
    preprocessor = preprocessor_factory.get('stock')
    preprocessed_stock_data = preprocessor.preprocess(
        stock_data,
        save=False,  # To save the preprocessed data on File system
        dataset=dataset_name  # Identifier for the dataset
    )
    
    # Get model
    print('Instantiating model: {}'.format(model))
    my_model = model_factory.get(model)
    
    # Train model
    print('--->> Training model...')
    my_model.fit(preprocessed_stock_data, dataset=dataset_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')  # LSTM/lstm, XGB/xgb, ARIMA/arima
    parser.add_argument('--dataset')  # AAPL/aapl, GMBL/gmbl, TSLA/tsla
    
    # Parse the arguments
    args = parser.parse_args()
    
    if args.model is not None and args.dataset is not None:
        # If arguments are passed run user defined model and dataset
        input_model = args.model.upper()
        input_dataset = args.dataset.upper()
        
        run(input_model, input_dataset)
    else:
        # If no arguments are passed run all models on all datasets
        run('XGB', 'AAPL')
        run('LSTM', 'AAPL')
        run('ARIMA', 'AAPL')
        
        run('XGB', 'TSLA')
        run('LSTM', 'TSLA')
        run('ARIMA', 'TSLA')
        
        run('XGB', 'GMBL')
        run('LSTM', 'GMBL')
        run('ARIMA', 'GMBL')
