from .dataset import Dataset
from .mnist import MNIST

implementations = (MNIST,)
datasets = {impl.__name__: impl for impl in implementations}
