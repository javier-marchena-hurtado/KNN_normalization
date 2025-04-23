import numpy as np
import pytest
from anndata import AnnData
from mudata import MuData

import knn_normalization


def test_package_has_version():
    assert knn_normalization.__version__ is not None


@pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_example():
    assert 1 == 0  # This test is designed to fail.


def test_retrieve_neighbors(adata):
    """This test tests whether the function retrieve_neighbors works as intended."""

    neighbor_pairs = np.where(
        adata.obsp["connectivities"].toarray() > 0
    )  # A pair of cells are neighbors if their connectivity is higher than 0.
    assert len(neighbor_pairs) == 2

    neighbors = knn_normalization.tl.retrieve_neighbors(adata)
    assert isinstance(neighbors, dict)
    assert len(neighbors) == adata.n_obs


def test_knn_normalization_basic(adata):
    neighbors = knn_normalization.tl.retrieve_neighbors(adata)
    original_data = adata.X.copy()
    knn_normalization(adata, neighbors, log_transform=False, inplace=True)
    assert not np.allclose(adata.X, original_data)  # Data should be changed


def test_calculate_neighbors_from_protein(adata):
    neighbors = knn_normalization.tl.calculate_neighbors_from_protein(adata, n_neighbors=3, log_transform=True)
    assert isinstance(neighbors, dict)
    assert len(neighbors) > 0


def test_knn_normalize_protein_return():
    np.random.seed(0)
    prot_data = AnnData(X=np.random.poisson(10, (10, 5)).astype(float))
    rna_data = AnnData(X=np.random.poisson(20, (10, 100)).astype(float))
    mdata = MuData({"prot": prot_data, "rna": rna_data})

    result = knn_normalization.tl.knn_normalize_protein(mdata, calculate_neighbors_from="rna", n_neighbors=3, inplace=False)
    assert isinstance(result, MuData)
