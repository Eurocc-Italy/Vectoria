from abc import ABCMeta

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
    
    @classmethod
    def reset(cls):
        cls._instances = {}

class SingletonABC(ABCMeta, Singleton):
    pass