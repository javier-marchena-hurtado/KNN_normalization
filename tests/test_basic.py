import numpy as np
from anndata import AnnData
from mudata import MuData

import knn_normalization


def test_package_has_version():
    assert knn_normalization.__version__ is not None


def test_retrieve_neighbors(adata):
    """This test tests whether the function retrieve_neighbors works as intended."""

    neighbor_pairs = np.where(
        adata.obsp["connectivities"].toarray() > 0
    )  # A pair of cells are neighbors if their connectivity is higher than 0.
    assert len(neighbor_pairs) == 2

    neighbors = knn_normalization.pp.retrieve_neighbors(adata)
    assert isinstance(neighbors, dict)
    assert len(neighbors) == adata.n_obs


def test_normalize_with_neighbors(adata):
    """This test tests whether the function _normalize_with_neighbors works as intended."""
    neighbors = knn_normalization.pp.retrieve_neighbors(adata)
    original_data = adata.X.copy()
    knn_normalization.tl._normalize_with_neighbors(adata, neighbors, log_transform=False, inplace=True)
    assert not np.allclose(adata.X, original_data)  # Data should be changed


def test_calculate_neighbors_from_protein(adata):
    """This test tests whether the function calculate_neighbors_from_protein works as intended."""
    neighbors = knn_normalization.pp.calculate_neighbors_from_protein(adata, n_neighbors=3, log_transform=True)
    assert isinstance(neighbors, dict)
    assert len(neighbors) > 0


def test_knn_normalize_protein_anndata(adata):
    """This test tests whether the function knn_normalize_protein works as intended, using an AnnData object as input."""
    knn_normalization.tl.knn_normalize_protein(
        adata, calculate_neighbors_from="prot", n_neighbors=3, inplace=True, log_transform=True
    )
    assert isinstance(adata, AnnData)


def test_knn_normalize_protein_mudata_rna_neighbors(mdata):
    """This test tests whether the function knn_normalize_protein works as intended, using an MuData object as input and with RNA neighbors."""
    result = knn_normalization.tl.knn_normalize_protein(
        mdata, calculate_neighbors_from="rna", n_neighbors=3, inplace=False
    )
    assert isinstance(result, MuData)


def test_knn_normalize_protein_mudata_protein_neighbors(mdata):
    """This test tests whether the function knn_normalize_protein works as intended, using an MuData object as input and with protein neighbors."""
    knn_normalization.tl.knn_normalize_protein(
        mdata, calculate_neighbors_from="prot", n_neighbors=3, inplace=True, log_transform=True
    )
    assert isinstance(mdata, MuData)
