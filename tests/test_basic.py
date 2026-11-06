import numpy as np
from anndata import AnnData
from mudata import MuData
from scipy.sparse import csr_matrix

import knn_normalization as knn
from knn_normalization._utils import _get_neighbors


def test_package_has_version():
    assert knn.__version__ is not None


def test_normalize_with_neighbors(adata):
    """This test tests whether the function normalize_with_neighbors works as intended."""
    original_data = adata.X.copy()
    neighbors = adata.obsp["connectivities"]
    knn.normalize_with_neighbors(adata, neighbors, log_transform=False)
    assert not np.allclose(adata.X, original_data)  # Data should be changed


def test_get_neighbors(adata, mdata):
    """This test tests whether the function _get_neighbors returns a KNN graph for each neighbor source."""
    protein = mdata["prot"]
    graphs = [
        _get_neighbors(adata, adata, "prot", n_neighbors=3),
        _get_neighbors(mdata, protein, "rna", n_neighbors=3),
        _get_neighbors(adata, adata, "use_existing_neighbors", n_neighbors=3),
    ]
    for graph in graphs:
        assert isinstance(graph, csr_matrix)
        assert graph.shape == (adata.n_obs, adata.n_obs)


def test_knn_normalize_anndata(adata):
    """This test tests whether the function knn_normalize works as intended, using an AnnData object as input."""
    knn.knn_normalize(adata, calculate_neighbors_from="prot", n_neighbors=3, inplace=True, log_transform=True)
    assert isinstance(adata, AnnData)
    assert "knn_normalization" in adata.uns


def test_knn_normalize_mudata_rna_neighbors(mdata):
    """This test tests whether the function knn_normalize works as intended, using an MuData object as input and with RNA neighbors."""
    result = knn.knn_normalize(mdata, calculate_neighbors_from="rna", n_neighbors=3, inplace=False)
    assert isinstance(result, MuData)


def test_knn_normalize_mudata_protein_neighbors(mdata):
    """This test tests whether the function knn_normalize works as intended, using an MuData object as input and with protein neighbors."""
    knn.knn_normalize(mdata, calculate_neighbors_from="prot", n_neighbors=3, inplace=True, log_transform=True)
    assert isinstance(mdata, MuData)
