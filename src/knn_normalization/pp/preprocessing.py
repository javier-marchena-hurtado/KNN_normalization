import scanpy as sc
from anndata import AnnData

# Below this number of proteins, neighbors are computed directly on the protein
# data; at or above it, PCA is applied first.
PCA_PROTEIN_THRESHOLD = 70


def _compute_pca_and_neighbors(adata: AnnData, n_neighbors: int, use_pca: bool, metric="euclidean"):
    """Compute the PCA (if ``use_pca``) and KNN graph and return the KNN graph."""
    if use_pca:
        sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca" if use_pca else "X", metric=metric)
    return adata.obsp["connectivities"]


def calculate_neighbors_from_protein(protein_data, n_neighbors, log_transform=True, verbose=True):
    """
    Calculate the KNN graph from protein expression data.

    If the number of proteins is less than ``PCA_PROTEIN_THRESHOLD`` (70),
    neighbors are calculated directly on the protein data. Otherwise, PCA is
    applied first.

    Parameters
    ----------
    protein_data
        AnnData object with raw protein expression counts.
    n_neighbors
        Number of neighbors to compute.
    log_transform
        Whether to log-transform the protein data before computing neighbors.
    verbose
        Whether to print progress messages.

    Returns
    -------
    The KNN graph in scipy.sparse.csr_matrix format.
    """
    if verbose:
        print("Calculating KNN graph from protein data")

    protein = AnnData(protein_data.X.copy())
    if log_transform:
        sc.pp.log1p(protein)
    use_pca = protein.n_vars >= PCA_PROTEIN_THRESHOLD
    return _compute_pca_and_neighbors(protein, n_neighbors, use_pca=use_pca, metric="cosine")


def calculate_neighbors_from_rna(rna_data, n_neighbors, verbose=True):
    """
    Calculate the KNN graph from the RNA data.

    Parameters
    ----------
    rna_data
        AnnData object with RNA data. Expects raw counts if preprocess=True, and normalized if preprocess=False.
    n_neighbors
        Number of neighbors to use for the KNN graph.
    verbose
        Whether to print progress messages.

    Returns
    -------
    The KNN graph in scipy.sparse.csr_matrix format.
    """
    if verbose:
        print("Calculating KNN graph from RNA data")

    rna = AnnData(rna_data.X.copy())
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)

    return _compute_pca_and_neighbors(rna, n_neighbors, use_pca=True)
