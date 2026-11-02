import scanpy as sc

# Below this number of proteins, neighbors are computed directly on the protein
# data; at or above it, PCA is applied first.
PCA_PROTEIN_THRESHOLD = 70


def retrieve_neighbors(connectivities):
    """
    Convert the KNN graph into a dictionary mapping each cell to its neighbors.

    Parameters
    ----------
    connectivities
        KNN graph in the format of ``.obsp["connectivities"]``.

    Returns
    -------
    dict
        Dictionary mapping each cell index to a list of its neighbor indices.
    """
    row_indices, col_indices = connectivities.nonzero()

    neighbors = {}
    for src, tgt in zip(row_indices, col_indices, strict=True):
        neighbors.setdefault(src, []).append(tgt)

    return neighbors


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

    adata = protein_data.copy()
    if log_transform:
        sc.pp.log1p(adata)

    if adata.n_vars < PCA_PROTEIN_THRESHOLD:
        sc.pp.neighbors(adata, use_rep="X", metric="cosine", n_neighbors=n_neighbors)
    else:
        sc.pp.pca(adata)
        sc.pp.neighbors(adata, metric="cosine", n_neighbors=n_neighbors)

    return adata.obsp["connectivities"]


def calculate_neighbors_from_rna(rna_data, n_neighbors, preprocess=True, verbose=True):
    """
    Calculate the KNN graph from the RNA data.

    Parameters
    ----------
    rna_data
        AnnData object with RNA data
    """
    data_for_neighbors = rna_data.copy()
    if verbose:
        print("Calculating KNN graph from RNA data")

    if preprocess:
        sc.pp.normalize_total(data_for_neighbors)
        sc.pp.log1p(data_for_neighbors)
    sc.pp.pca(data_for_neighbors)
    sc.pp.neighbors(data_for_neighbors, n_neighbors=n_neighbors)

    return data_for_neighbors.obsp["connectivities"]
