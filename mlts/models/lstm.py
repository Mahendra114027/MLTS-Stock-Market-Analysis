from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM as KerasLSTM
from mlts.utils.save import save_model
from mlts.utils.data import split_data
from mlts.config import ModelParams
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
        # Variables
        scaled_data = df.copy()
        cols_to_scale = df.columns
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        
        # Split the data into training and testing sets
        train_data, test_data = split_data(scaled_data)
        
        # Create the training and testing sets
        target_var = ModelParams.TARGET.value
        input_vars = scaled_data.columns.drop(target_var)
        x_train = np.array(train_data[input_vars], dtype=np.float32)
        y_train = np.array(train_data[[target_var]], dtype=np.float32)
        
        # Reshape the data for the LSTM model
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # Build the LSTM model
        self._model = Sequential()
        self._model.add(KerasLSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        self._model.add(KerasLSTM(units=50))
        self._model.add(Dense(1))
        
        # Compile and train the self._model
        self._model.compile(
            loss='mean_squared_error',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        # Train the model
        history = self._model.fit(
            x_train, y_train,
            epochs=ModelParams.EPOCHS.value,
            batch_size=ModelParams.BATCH_SIZE.value,
            verbose=ModelParams.VERBOSE.value
        )
        print(f'Loss values and metrics: \n {history.history}')
        
        # Create the testing sets
        x_test = np.array(test_data[input_vars], dtype=np.float32)
        y_test = np.array(test_data[[target_var]], dtype=np.float32)
        
        # Evaluate Model
        results = self._model.evaluate(x_test, y_test, batch_size=1)
        print('Results: ', results)
        
        # Make predictions on the test data
        predictions = self._model.predict(x_test)
        
        # Save the model
        save_model(self._model, 'LSTM')
    
    def predict(self, data, **kwargs):
        pass
