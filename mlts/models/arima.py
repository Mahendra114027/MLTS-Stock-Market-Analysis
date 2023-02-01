from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
from sklearn.metrics import mean_squared_error
from mlts.utils.data import split_data
from mlts.utils.save import save_model
from pmdarima.arima import auto_arima
from mlts.config import ModelParams
from mlts.models import Model
import numpy as np


class ARIMA(Model):
    """
    ARIMA model class
    """
    
    def __init__(self):
        super().__init__()
        self._model = None
        np.random.seed(42)
    
    def fit(self, data, **kwargs):
        # Split the data into training and testing sets
        train_data, test_data = split_data(data)
        
        # Build the LSTM model
        model_autoarima = auto_arima(
            train_data['adj_close'],
            start_p=0,
            start_q=0,
            test='adf',  # use adftest to find optimal 'd'
            max_p=5,
            max_q=5,  # maximum p and q
            m=1,  # frequency of series
            d=None,  # let model determine 'd'
            seasonal=False,  # No Seasonality
            start_P=0,
            D=0,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        print(model_autoarima.summary())
        
        # Compile and train the self._model
        train = train_data[ModelParams.TARGET.value].values
        test = test_data[ModelParams.TARGET.value].values
        
        # Make predictions on the test data
        history = [x for x in train]
        predictions = list()
        
        for i in range(len(test)):
            model = StatsARIMA(history, order=(3, 1, 0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[i]
            history.append(obs)
        
        # Evaluate the model
        error = mean_squared_error(test, predictions)
        print('Testing Mean Squared Error: %.3f' % error)
        
        # Save the model
        save_model(self._model, 'ARIMA')
    
    def predict(self, data, **kwargs):
        pass