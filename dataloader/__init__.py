def get_dataloader(*args, **kwargs):
    # Standalone dataset imports do not need the application's local config.py.
    from .get_dataloader import get_dataloader as factory
    return factory(*args, **kwargs)

from .MNIST import MNISTDataset
from .MultiColorShapes import MultiColorShapesDataset
from .FaceDataset import FaceDataset
from .Malaria import MalariaCellDataset
from .Caltech101 import Caltech101Dataset
from .NonclassicFace import NonclassicFaceDataset
__all__ = ['get_dataloader', 
           'MNISTDataset', 'MultiColorShapesDataset', "FaceDataset",
           "MalariaCellDataset", "AnotherColored_MNIST", "Caltech101Dataset",
           "NonclassicFaceDataset"]
