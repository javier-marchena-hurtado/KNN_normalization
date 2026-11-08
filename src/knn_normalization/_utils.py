import os
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral, Real
from warnings import warn

import numpy as np
import scanpy as sc
from anndata import AnnData
from mudata import MuData
from scipy.sparse import csr_matrix, issparse

# Below this number of proteins, neighbors are computed directly on the protein
# data; at or above it, PCA is applied first.
PCA_PROTEIN_THRESHOLD = 70

_NEIGHBOR_SOURCES = ("prot", "rna", "use_existing_neighbors")
_N_CELLS_TO_CHECK = 100  # only the first cells are inspected when validating the values


def _check_positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _check_values(x, name):
    """Check the first cells of ``x``: raise on NaN or infinite values, warn on negative values.

    Returns True if all checked values are integers.
    """
    head = x[:_N_CELLS_TO_CHECK]
    values = head.data if issparse(head) else np.asarray(head)  # sparse matrices only store their non-zeros
    if not np.isfinite(values).all():
        raise ValueError(f"The {name} data contains NaN or infinite values.")
    if values.size and values.min() < 0:
        warn(f"The {name} data contains negative values; raw (non-negative) counts are expected.", stacklevel=3)
    return bool(np.all(values == np.round(values)))


def _validate_inputs(
    data,
    calculate_neighbors_from,
    log_transform,
    n_neighbors,
    pseudocount,
    max_iterations,
    inplace,
    save_size_factors,
):
    """Raise a clear error (or warn) if the arguments or data cannot be normalized."""
    # Check arguments of knn_normalize()
    if not isinstance(data, (AnnData, MuData)):
        raise TypeError(f"data must be an AnnData or MuData object, got {type(data).__name__}.")
    if not isinstance(calculate_neighbors_from, str) or calculate_neighbors_from not in _NEIGHBOR_SOURCES:
        raise ValueError(
            f"calculate_neighbors_from must be one of {_NEIGHBOR_SOURCES}, got {calculate_neighbors_from!r}."
        )
    _check_positive_int(max_iterations, "max_iterations")
    if n_neighbors is not None:
        _check_positive_int(n_neighbors, "n_neighbors")
    if isinstance(pseudocount, bool) or not isinstance(pseudocount, Real) or not 0 <= pseudocount < np.inf:
        raise ValueError(f"pseudocount must be a finite number >= 0, got {pseudocount!r}.")

    # Check the structure of the input
    if isinstance(data, MuData) and "prot" not in data.mod:
        raise ValueError("A MuData input needs a 'prot' modality.")
    if calculate_neighbors_from == "rna":
        if not isinstance(data, MuData):
            raise TypeError("calculate_neighbors_from='rna' requires a MuData object with 'prot' and 'rna' modalities.")
        if not {"prot", "rna"} <= data.mod.keys():
            raise ValueError("calculate_neighbors_from='rna' requires both a 'prot' and an 'rna' modality.")
        if not data["rna"].obs_names.equals(data["prot"].obs_names):
            raise ValueError(
                "The 'rna' and 'prot' modalities must contain the same cells in the same order. "
                "Subset them first, e.g. with mudata.pp.intersect_obs(mdata)."
            )

    protein = _get_protein_adata(data)
    if protein.isbacked or (inplace and protein.is_view):
        raise ValueError("The data is a view or backed on disk. Use .copy() (or .to_memory()) before normalizing.")
    if n_neighbors is not None and n_neighbors >= protein.n_obs:
        raise ValueError(f"n_neighbors ({n_neighbors}) must be smaller than the number of cells ({protein.n_obs}).")
    if protein.n_obs <= _default_n_neighbors(protein.n_obs):
        raise ValueError(f"Too few cells ({protein.n_obs}) to compute a KNN graph.")

    # Check the neighbor source
    if calculate_neighbors_from == "use_existing_neighbors":
        graph = protein.obsp.get("connectivities")
        if graph is None:
            raise ValueError(
                "calculate_neighbors_from='use_existing_neighbors' requires "
                "protein.obsp['connectivities']. Run sc.pp.neighbors on the protein data first."
            )
        if graph.shape != (protein.n_obs, protein.n_obs):
            raise ValueError(
                f"protein.obsp['connectivities'] has shape {graph.shape}, expected {(protein.n_obs,) * 2}."
            )

    # Check protein values (first cells only)
    if not _check_values(protein.X, "protein"):
        warn("The protein data is not integer-valued. KNN normalization expects raw counts.", stacklevel=3)
    if log_transform and "log1p" in protein.uns:
        warn("The protein data might already be log-transformed, consider using log_transform=False", stacklevel=3)
    if "knn_normalization" in protein.uns:
        warn(
            "This protein data has already been KNN-normalized; normalizing again is possibly a mistake.", stacklevel=3
        )
    if save_size_factors and "size_factor" in protein.obs:
        warn("protein.obs['size_factor'] already exists and will be overwritten.", stacklevel=3)

    # Check RNA values (first cells only)
    if calculate_neighbors_from == "rna" and not _check_values(data["rna"].X, "RNA"):
        warn(
            "The RNA data is not integer-valued. KNN normalization expects raw RNA counts "
            "and will library-size normalize and log-transform them.",
            stacklevel=3,
        )


def _get_protein_adata(data):
    """Return the protein AnnData: the ``prot`` modality of a MuData, or the AnnData itself."""
    return data["prot"] if isinstance(data, MuData) else data


def _densify(x):
    """Return a dense float copy of x."""
    return x.toarray().astype(float, copy=False) if issparse(x) else np.array(x, dtype=float)


def _default_n_neighbors(n_cells):
    return max(15, min(round(n_cells / 20), 300))


def _get_neighbors(data, protein, calculate_neighbors_from, n_neighbors, log_transform=True, verbose=True):
    """
    Get the KNN graph (sparse matrix of connectivities) from the protein or RNA data, or reuse the existing one.

    For protein data, PCA is applied first if there are at least ``PCA_PROTEIN_THRESHOLD`` proteins, and the
    cosine distance is used. RNA data is library-size normalized and log-transformed, and always uses PCA.
    """
    if calculate_neighbors_from == "use_existing_neighbors":
        return protein.obsp["connectivities"]

    if calculate_neighbors_from == "prot":
        if verbose:
            print("Calculating KNN graph from protein data")
        adata = AnnData(protein.X.copy())
        if log_transform:
            sc.pp.log1p(adata)
        use_pca = adata.n_vars >= PCA_PROTEIN_THRESHOLD
        metric = "cosine"
    else:
        if verbose:
            print("Calculating KNN graph from RNA data")
        adata = AnnData(data["rna"].X.copy())
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        use_pca = True
        metric = "euclidean"

    if use_pca:
        sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca" if use_pca else "X", metric=metric)
    return adata.obsp["connectivities"]


def _neighbor_pairs(connectivities: csr_matrix):
    """
    Return (target_cells, neighbor_cells, neighbors_per_cell) of the KNN graph.

    ``target_cells[i]`` and ``neighbor_cells[i]`` form the i-th (cell, neighbor) pair.
    """
    knn_graph = connectivities.tocsr(copy=True)
    knn_graph.eliminate_zeros()
    neighbors_per_cell = np.diff(knn_graph.indptr)
    target_cells = np.repeat(np.arange(knn_graph.shape[0]), neighbors_per_cell)
    return target_cells, knn_graph.indices, neighbors_per_cell


def _n_available_cpus():
    """Number of CPUs that this process is allowed to use (this respects, for example, a SLURM allocation)."""
    if hasattr(os, "sched_getaffinity"):  # Not available on macOS and Windows.
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


def _pairwise_median_ratios(x, target_cells, neighbor_cells, chunk_size=20_000):
    """
    Median over proteins of ``neighbor / cell`` expression, for every (cell, neighbor) pair.

    This is the expensive step of the normalization. It only has to be computed once: afterwards the
    ratios can be updated from the size factors alone (see ``normalize_with_neighbors``).

    The pairs are split into chunks. The chunks do not depend on each other, so several chunks are processed
    at the same time, each in its own thread, using all CPUs that are available.

    Parameters
    ----------
    x
        Dense array of protein data (pseudocount already added).
    target_cells
        For each (cell, neighbor) pair, the index of the target cell.
    neighbor_cells
        For each (cell, neighbor) pair, the index of the neighbor cell.
    chunk_size
        Number of (cell, neighbor) pairs in one chunk. Every thread holds one chunk in memory at a time,
        so a smaller chunk size means less memory use.

    Returns
    -------
    np.array with one median ratio per (cell, neighbor) pair.
    """
    n_pairs = len(target_cells)
    median_ratios = np.empty(n_pairs)

    def fill_in_chunk(first_pair):
        """Calculate the median ratios of one chunk of pairs and store them in ``median_ratios``."""
        chunk = slice(first_pair, first_pair + chunk_size)
        ratios = x[neighbor_cells[chunk]] / x[target_cells[chunk]]
        median_ratios[chunk] = np.median(ratios, axis=1)

    first_pair_of_each_chunk = range(0, n_pairs, chunk_size)
    with ThreadPoolExecutor(max_workers=_n_available_cpus()) as executor:
        list(executor.map(fill_in_chunk, first_pair_of_each_chunk))  # list() makes errors in a thread show up here
    return median_ratios


def _size_factors_from_ratios(median_ratios, target_cells, neighbors_per_cell):
    """
    Calculate the size factor of each cell.

    The size factor of a cell is the average of the median ratios of all of its neighbors. A cell
    without neighbors gets a neutral size factor of 1.

    Parameters
    ----------
    median_ratios
        One median ratio per (cell, neighbor) pair.
    target_cells
        For each (cell, neighbor) pair, the index of the cell the pair belongs to.
    neighbors_per_cell
        Number of neighbors of each cell.

    Returns
    -------
    np.array with one size factor per cell.
    """
    n_cells = len(neighbors_per_cell)
    has_neighbors = neighbors_per_cell > 0

    # Add up the median ratios of all pairs that belong to the same cell.
    ratio_sum_per_cell = np.bincount(target_cells, weights=median_ratios, minlength=n_cells)

    # Average them. Cells without neighbors keep the neutral value 1.
    size_factors = np.ones(n_cells)
    size_factors[has_neighbors] = ratio_sum_per_cell[has_neighbors] / neighbors_per_cell[has_neighbors]
    return size_factors


def _save_size_factors(protein_anndata, size_factor_history, connectivities):
    """Saves size factors and connectivities to the resulting AnnData object."""
    history = np.array(size_factor_history)
    protein_anndata.obs["size_factor"] = history.prod(axis=0)  # cumulative factor over all iterations
    protein_anndata.obsm["size_factor_history"] = history.T
    protein_anndata.obsp["connectivities_KNN_normalization"] = connectivities
