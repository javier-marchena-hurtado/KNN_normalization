from importlib.metadata import version

from .knn_normalization import knn_normalize, normalize_with_neighbors

__all__ = ["knn_normalize", "normalize_with_neighbors"]

__version__ = version("KNN_normalization")
