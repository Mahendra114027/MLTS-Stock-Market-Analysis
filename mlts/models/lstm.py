from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM as KerasLSTM
from keras.models import Sequential
from keras.layers import Dense
from mlts.models import Model
import numpy as np


class LSTM(Model):
    """
    LSTM model class
    """
    
    def __init__(self):
        super().__init__()
        self._model = None
        np.random.seed(42)
    
    def fit(self, df, **kwargs):
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['Close']])
        
        # Split the data into training and testing sets
        train_data = scaled_data[:int(len(scaled_data) * 0.8)]
        test_data = scaled_data[int(len(scaled_data) * 0.8):]
        
        # Create the training and testing sets
        x_train = []
        y_train = []
        
        for i in range(60, len(train_data)):
            x_train.append(train_data[i - 60:i, 0])
            y_train.append(train_data[i, 0])
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        
        # Reshape the data for the LSTM model
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # Build the LSTM model
        self._model = Sequential()
        self._model.add(KerasLSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        self._model.add(KerasLSTM(units=50))
        self._model.add(Dense(1))
        
        # Compile and train the self._model
        self._model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        
        self._model.fit(x_train, y_train, epochs=1, batch_size=1)
        
        # Make predictions on the test data
        x_test = []
        y_test = []
        
        for i in range(60, len(test_data)):
            x_test.append(test_data[i - 60:i, 0])
            y_test.append(test_data[i, 0])
        
        x_test, y_test = np.array(x_test), np.array(y_test)
        
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        predictions = self._model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        
        # Evaluate the model
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        print('Test RMSE:', rmse)
    
    def predict(self, data, **kwargs):
        pass
