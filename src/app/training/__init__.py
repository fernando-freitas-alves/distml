from .dist_training import DistTraining
from .synced_sgd import SyncedSGD

implementations = (SyncedSGD,)
training_methods = {impl.__name__: impl for impl in implementations}
