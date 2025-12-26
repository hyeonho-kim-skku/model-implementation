from .supervised_learning import *
from .rotnet import *
from .simclr import *

def load_method(method_name, model):
    if method_name == 'supervised':
        return SL(model)
    elif method_name == 'rotnet':
        return RotNet(model)
    elif method_name == 'simclr':
        return SimCLR(model)