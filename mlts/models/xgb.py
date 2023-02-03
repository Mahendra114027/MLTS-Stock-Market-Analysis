from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from mlts.utils.data import split_data
from xgboost import XGBRegressor
from mlts.models import Model
import pandas as pd
import math


class XGB(Model):
    """
    XGBoost Model
    """
    
    def fit(self, df, **kwargs):
        """
        Fit the model
        
        Args:
            df (pd.DataFrame): Data to fit the model
            **kwargs: Additional arguments
        """
        train_data, test_data = split_data(df)
        
        """Hyperparameter Tuning"""
        parameters = {
            # max_depth :Maximum tree depth for base learners
            'max_depth': range(2, 10, 1),
            
            # n_estimators: Number of boosted trees to fit
            'n_estimators': range(10, 250, 10),
            
            # learning_rate:Boosting learning rate (xgb’s “eta”)
            'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
            
            # min_child_weight : Minimum sum of instance weight(hessian) needed in a child
            'min_child_weight': range(1, 21, 2),
            
            # subsample : Subsample ratio of the training instance
            'subsample': [0, 0.2, 0.4, 0.6, 0.8, 1],
            
            # gamma : Minimum loss reduction required to make a further partition on a leaf node of the tree
            'gamma': [0, 0.2, 0.4, 0.6, 0.8, 1],
            
            # colsample_bytree :Subsample ratio of columns when constructing each tree
            'colsample_bytree': [0, 0.2, 0.4, 0.6, 0.8, 1],
            
            # colsample_bylevel :Subsample ratio of columns for each split, in each level
            'colsample_bylevel': [0, 0.2, 0.4, 0.6, 0.8, 1]
        }
        
        # Xgboost Model
        estimator = XGBRegressor(seed=42)
        
        # Grid search cross validation to get optimal parameters
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=parameters,
            scoring='neg_mean_squared_error'
        )
        
        tuned_model = grid_search.fit(X_train_scaled, y_train_scaled)
        return tuned_model
    
    def predict(self, data, **kwargs):
        """
        Predict the data

        Args:
            data:
            **kwargs:

        Returns:
            predictions (np.array): Predictions
        """
        if self._model is None:
            raise Exception('Model not trained')
        
        return self._model.predict(data)
    
    def normalize(self, row, mean, std):
        """
        Function to Normalize the column Values
        row=Row by Row Input Dataframe
        mean= Mean of the Feature
        std= Standard deviation of the feature

        Return
        Scaled Row
        """
        
        std = 0.001 if std == 0 else std
        scaRow = (row - mean) / std
        return scaRow
    
    def test_train_split(self, df, **kwargs):
        
        """
        This function is used to standarize the dataframe and split it into test and train.
        Input
        df Raw Stock dataframe

        Returns
        train - unscaled training data
        test - unscaled test data
        X_train_scaled - Scaled training data with input features
        y_train_scaled - Scaled target variable
        X_test_scaled - Scaled testing data with input features
        scaler_train.var_[0] - Standard Scaler variance
        scaler_train.mean_[0] - Standard scaler mean

        """
        df = self.featureEnginnering(df)
        test_size = 0.2
        num_test = int(test_size * len(df))
        num_train = len(df) - num_test
        
        # Split into train and test
        train = df[:num_train]
        test = df[num_train:]
        
        scaleColumns = ["adj_close"]
        
        for i in range(1, self.N + 1):
            scaleColumns.append("lag_" + str(i) + "_" + "highLowDiff")
            scaleColumns.append("lag_" + str(i) + "_" + "openCloseDiff")
            scaleColumns.append("lag_" + str(i) + "_" + "adj_close")
            scaleColumns.append("lag_" + str(i) + "_" + "volume")
        
        # Scaling train dataset
        scaler_train = StandardScaler()
        trainScaled = scaler_train.fit_transform(train[scaleColumns])
        
        # Convert the numpy array back into pandas dataframe
        trainScaled = pd.DataFrame(trainScaled, columns=scaleColumns)
        trainScaled[['date', 'month']] = train.reset_index()[['date', 'month']]
        
        # Scaling for test set
        testScaled = test[['date']]
        lagColumns = ['highLowDiff', 'openCloseDiff', 'volume', 'adj_close']
        for col in lagColumns:
            feat_list = ['lag_1_' + col, 'lag_2_' + col, 'lag_3_' + col]
            temp = test.apply(lambda row: self.normalize(row[feat_list], row[col + '_mean'], row[col + '_std']), axis=1)
            testScaled = pd.concat([testScaled, temp], axis=1)
        
        feat_Cols = []
        for i in range(1, self.N + 1):
            feat_Cols.append("lag_" + str(i) + "_" + "highLowDiff")
            feat_Cols.append("lag_" + str(i) + "_" + "openCloseDiff")
            feat_Cols.append("lag_" + str(i) + "_" + "adj_close")
            feat_Cols.append("lag_" + str(i) + "_" + "volume")
        
        tar_Col = "adj_close"
        
        # Split into X and y
        X_train_scaled = trainScaled[feat_Cols]
        y_train_scaled = trainScaled[tar_Col]
        X_test_scaled = testScaled[feat_Cols]
        
        return train, test, X_train_scaled, y_train_scaled, X_test_scaled, scaler_train.var_[0], scaler_train.mean_[0]
    
    def predict_train(self, xgboostModel, X_train_scaled, scalerVar, scalerMean, **kwargs):
        """
        This function do prediction on the training set.
        Input
        xgboostModel - xgBoost trainiend model
        X_train_scaled - Scaled training data with input features
        scalerVar - Standard Scaler variance
        scalerMean - Standard scaler mean

        Output
        Array of predictions
        """
        pred = xgboostModel.predict(X_train_scaled) * math.sqrt(scalerVar) + scalerMean
        return pred
    
    def predict_test(self, xgboostModel, X_test_scaled, test, **kwargs):
        """
        This function do prediction on the testing set.
        Input
        xgboostModel - xgBoost trainiend model
        X_test_scaled - Scaled testing data with input features

        Output
        Array of predictions
        """
        pred = xgboostModel.predict(X_test_scaled) * test['adj_close_std'] + test['adj_close_mean']
        return pred
