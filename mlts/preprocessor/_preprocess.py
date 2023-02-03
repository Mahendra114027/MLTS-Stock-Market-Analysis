class Preprocessor:
    """
    Preprocessor
    
    An abstract class for doing preprocessing on data before modeling.
    """
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def preprocess(self):
        raise NotImplementedError
