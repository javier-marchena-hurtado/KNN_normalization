import numpy as np
from anndata import AnnData
from mudata import MuData
from scipy.sparse import csr_matrix

import knn_normalization as knn
from knn_normalization.pp.preprocessing import _compute_pca_and_neighbors


def test_package_has_version():
    assert knn.__version__ is not None


def test_normalize_with_neighbors(adata):
    """This test tests whether the function normalize_with_neighbors works as intended."""
    original_data = adata.X.copy()
    neighbors = adata.obsp["connectivities"]
    knn.tl.normalize_with_neighbors(adata, neighbors, log_transform=False)
    assert not np.allclose(adata.X, original_data)  # Data should be changed


def test_calculate_neighbors_from_protein(adata):
    """This test tests whether the function calculate_neighbors_from_protein works as intended."""
    connectivities = knn.pp.calculate_neighbors_from_protein(adata, n_neighbors=3, log_transform=True)
    _compute_pca_and_neighbors(adata, n_neighbors=3, use_pca=False, metric="cosine")
    graph = adata.obsp.get("connectivities")
    assert isinstance(connectivities, csr_matrix)
    assert graph is not None
    assert graph.shape == (adata.n_obs, adata.n_obs)


def test_knn_normalize_anndata(adata):
    """This test tests whether the function knn_normalize works as intended, using an AnnData object as input."""
    knn.tl.knn_normalize(adata, calculate_neighbors_from="prot", n_neighbors=3, inplace=True, log_transform=True)
    assert isinstance(adata, AnnData)
    assert "knn_normalization" in adata.uns


def test_knn_normalize_mudata_rna_neighbors(mdata):
    """This test tests whether the function knn_normalize works as intended, using an MuData object as input and with RNA neighbors."""
    result = knn.tl.knn_normalize(mdata, calculate_neighbors_from="rna", n_neighbors=3, inplace=False)
    assert isinstance(result, MuData)


def test_knn_normalize_mudata_protein_neighbors(mdata):
    """This test tests whether the function knn_normalize works as intended, using an MuData object as input and with protein neighbors."""
    knn.tl.knn_normalize(mdata, calculate_neighbors_from="prot", n_neighbors=3, inplace=True, log_transform=True)
    assert isinstance(mdata, MuData)
