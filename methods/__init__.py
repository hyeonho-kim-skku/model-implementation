from .supervised_learning import SL
from .rotnet import RotNet
from .simclr import SimCLR
from .moco import MoCo
from .byol import BYOL
from .simsiam import SimSiam

def load_method(method_name, model):
    if method_name == 'supervised':
        return SL(model)
    elif method_name == 'rotnet':
        return RotNet(model)
    elif method_name == 'simclr':
        return SimCLR(model)
    elif method_name == 'moco':
        return MoCo(model)
    elif method_name == 'byol':
        return BYOL(model)
    elif method_name == 'simsiam':
        return SimSiam(model)