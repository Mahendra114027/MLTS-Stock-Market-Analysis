from mlts.utils.data import split_date, enrich_stock_features
from mlts.preprocessor import Preprocessor
from mlts.utils.save import save_data
from mlts.config import Preprocess


class StockPreprocessor(Preprocessor):
    """
    Stock Preprocessor
    """
    
    def preprocess(self, df, **kwargs):
        """
        Preprocess the data
        
        Args:
            df (pd.DataFrame): Dataframe to preprocess
            kwargs: Keyword arguments
            
        Returns:
            df (pd.DataFrame): Preprocessed dataframe
        """
        
        try:
            """Column Name Formatting"""
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            """Split date column into various date entities"""
            # year, month, day, quarter, is_month_start, is_month_end
            df = split_date(df, target_col='date')
            
            """Feature Engineering"""
            df = enrich_stock_features(df, num_days=5)
            
            # Drop features
            df = df.drop(columns=Preprocess.DROP_FEATURES.value)
            
            # Fill NaN values with 0
            df = df.fillna(0)
            
            # Set date as index
            df = df.reset_index(drop=True)
            df.set_index('date', inplace=True)
            
            # Parse the keyword arguments
            #  save (bool): Save the preprocessed data to disk
            #  dataset (str): Name of the dataset
            save = kwargs.get('save', False)
            dataset = kwargs.get('dataset', None)
            
            # Save the preprocessed data to disk
            if save:
                save_data(df, dataset)
            
            return df
        
        except Exception as ex:
            raise Exception(f"Preprocessing Failed {ex}")
