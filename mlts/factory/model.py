from mlts.factory import Factory
from mlts.models import LSTM


class ModelFactory(Factory):
    """
    Model Factory
    """
    
    def create(self, name, *args, **kwargs):
        
        if name == 'lstm':
            return LSTM(*args, **kwargs)
        else:
            raise ValueError('Unknown model name: {}'.format(name))
