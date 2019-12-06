from .radam import RAdam
from .lamb import LAMB
from .swa import SWA
from .lookahead import Lookahead
from .lars import LARS
from .yogi import Yogi

__all__ = ['RAdam', 'LAMB', 'SWA', 'Lookahead', 'LARS', 'Yogi', 'get_optimizer']


def get_optimizer(optimizer, **kwargs):
    assert optimizer.lower() in ['radam', 'lamb', 'lars', 'yogi'], ValueError('not a supported optimizer name')
    if optimizer.lower() == 'radam':
        return RAdam(**kwargs)
    if optimizer.lower() == 'lamb':
        return LAMB(**kwargs)
    if optimizer.lower() == 'lars':
        return LARS(**kwargs)
    if optimizer.lower() == 'yogi':
        return Yogi(**kwargs)
