from mlts.utils.data import split_date, enrich_stock_features
from mlts.preprocessor import Preprocessor


class StockPreprocessor(Preprocessor):
    """
    Stock Preprocessor
    """
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def preprocess(self, df):
        """
        Preprocess the data
        
        Args:
            df (pd.DataFrame): Dataframe to preprocess

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
            df = df.drop(columns=['close'])
            
            # Set date as index
            df.set_index('date', inplace=True)
            
            print(df.head())
            print(df.columns)

            return df
        
        except Exception as ex:
            raise Exception(f"Preprocessing Failed {ex}")
