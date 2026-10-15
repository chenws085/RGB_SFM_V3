from importlib import import_module


def __getattr__(name):
    # Import only the requested architecture; other models may need config.py.
    if name in __all__ or name == 'RGB_SFMCNN_V2':
        module = import_module(f'.{name}', __name__)
        globals()[name] = module
        return module
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

__all__ = ['CNN', 
           'SFMCNN', 
           'AlexNet', 
           "RGB_SFMCNN",
           "RGB_SFMCNN_V2",
           'ResNet',
           'GoogLeNet',
           'DenseNet',
           'Swin_tiny',
           'PVTv2',
           'VIT',
           'RGB_SFMCNN_V3',
           'MergingViT']
