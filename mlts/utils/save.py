from mlts.config import ModelPath
import os.path


def save_model(model, name):
    """
    Save model
    """
    
    if not os.path.isdir(os.path.dirname(ModelPath[name].value)):
        os.makedirs(os.path.dirname(ModelPath[name].value))
    
    if name in ['LSTM', 'XGB']:
        model.save(
            ModelPath[name].value,
            overwrite=True,
            include_optimizer=True,
            save_format='h5',
            save_traces=True,
        )
    elif name == 'ARIMA':
        raise NotImplementedError('ARIMA model saving not implemented yet.')
    else:
        raise ValueError('Unknown model name: {}'.format(name))
