from mlts.models import Model


class LSTM(Model):
    """
    LSTM model class
    """
    
    def __init__(self):
        super().__init__()
        self._model = None
    
    def _design_model(self):
        raise NotImplementedError
    
    def fit(self, data, **kwargs):
        pass
    
    def predict(self, data, **kwargs):
        pass
