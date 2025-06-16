import scanpy as sc


def retrieve_neighbors(neighbors):
    """Convert the KNN graph into a dictionary mapping each cell to its neighbor cells."""
    row_indices, col_indices = neighbors.nonzero()
    neighbors = {}
    for src, tgt in zip(row_indices, col_indices, strict=False):
        neighbors.setdefault(src, []).append(tgt)

    return neighbors


# def retrieve_neighbors(data):
#     """Convert the KNN graph into a dictionary mapping each cell to its neighbor cells."""
#     connectivity = data.obsp["connectivities"].toarray()
#     row_indices, col_indices = np.where(connectivity > 0)

#     neighbors = {}
#     for src, tgt in zip(row_indices, col_indices, strict=False):
#         neighbors.setdefault(src, []).append(tgt)

#     return neighbors


def calculate_neighbors_from_protein(protein_data, n_neighbors, log_transform=True):
    """Calculates neighbors (the KNN graph) from protein data."""
    data_for_neighbors = protein_data.copy()
    if log_transform:
        sc.pp.log1p(data_for_neighbors)

    n_proteins = data_for_neighbors.n_vars
    if (
        n_proteins < 70
    ):  # If the number of proteins is less than 70, calculate neighbors on the protein data without performing PCA first.
        sc.pp.neighbors(data_for_neighbors, use_rep="X", metric="cosine", n_neighbors=n_neighbors)
    else:  # If the number of proteins is more than 70, calculate neighbors on the PCA results of the protein data.
        sc.pp.pca(data_for_neighbors)
        sc.pp.neighbors(data_for_neighbors, metric="cosine", n_neighbors=n_neighbors)
    neighbors = data_for_neighbors.obsp["connectivities"]

    return neighbors
