from .model import Model
from .test_net import TestNet

implementations = (TestNet,)
models = {impl.__name__: impl for impl in implementations}
