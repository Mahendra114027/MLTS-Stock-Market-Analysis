from keras.models import Sequential, load_model
from mlts.config import ModelParams, ModelPath
from sklearn.preprocessing import MinMaxScaler
from kerastuner.tuners import RandomSearch
from mlts.utils.data import split_data
from mlts.utils.save import save_model
from mlts.models import Model
import keras.layers as kl
import numpy as np


class LSTM(Model):
    """
    LSTM model class
    """
    
    def __init__(self):
        super().__init__()
        self._model = None
        self.x_train = None
        self.y_train = None
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
        self.y_train = np.array(train_data[[target_var]], dtype=np.float32)
        
        # Reshape the data for the LSTM model
        self.x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # Define model
        model = Sequential()
        model.add(
            kl.LSTM(
                units=hp.Int('input_unit', min_value=32, max_value=512, step=32),
                return_sequences=True,
                input_shape=(self.x_train.shape[1], 1)
            )
        )
        
        for i in range(hp.Int('n_layers', 1, 4)):
            model.add(
                kl.LSTM(
                    hp.Int(f'lstm_{i}_units', min_value=32, max_value=512, step=32),
                    return_sequences=True
                )
            )
        
        model.add(kl.LSTM(hp.Int('layer_2_neurons', min_value=32, max_value=512, step=32)))
        model.add(kl.Dropout(hp.Float('Dropout_rate', min_value=0, max_value=0.5, step=0.1)))
        model.add(kl.Dense(self.y_train.shape[1], activation=hp.Choice('dense_activation')))
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        # Model tuning
        tuner = RandomSearch(
            model,
            objective='loss',
            max_trials=2,
            executions_per_trial=1
        )
        
        tuner.search(
            x=self.x_train,
            y=self.y_train,
            epochs=20,
            batch_size=32
        )
        
        best_model = tuner.get_best_models(num_models=1)[0]
        best_model.summary()
        self._model = best_model
        
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
        """
        Predicts the next value in the sequence
        
        Args:
            data (object): Data to predict
            **kwargs: Keyword arguments

        Returns:
            object: Predicted value
        """
        
        try:
            if self._model is None:
                self._model = load_model(ModelPath.LSTM.value)
            
            if self._model:
                predictions = self._model.predict(data)
                
                return predictions
        
        except Exception as ex:
            raise Exception('Error predicting data: ', ex)
