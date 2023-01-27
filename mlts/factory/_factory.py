class Factory:
    
    def __init__(self):
        pass
    
    def create(self, name, *args, **kwargs):
        raise NotImplementedError
