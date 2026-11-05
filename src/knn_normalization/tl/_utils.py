from numbers import Integral, Real
from warnings import warn

import numpy as np
from anndata import AnnData
from mudata import MuData
from scipy.sparse import csr_matrix, issparse

from knn_normalization.pp.preprocessing import (
    calculate_neighbors_from_protein,
    calculate_neighbors_from_rna,
)

_NEIGHBOR_SOURCES = ("prot", "rna", "use_existing_neighbors")
_N_CELLS_TO_CHECK = 100  # only the first cells are inspected when validating the values


def _sample_values(x, n_cells=_N_CELLS_TO_CHECK):
    """Return the stored values of the first ``n_cells`` rows of ``x`` (non-zeros only if sparse)."""
    head = x[:n_cells]
    return head.data if issparse(head) else np.asarray(head)


def _check_values(values, name):
    """Raise if ``values`` contain NaN, infinite or negative numbers; return True if integer-valued."""
    if not np.isfinite(values).all():
        raise ValueError(f"The {name} data contains NaN or infinite values.")
    if values.size and values.min() < 0:
        raise ValueError(f"The {name} data contains negative values; raw (non-negative) counts are expected.")
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
    if (
        isinstance(pseudocount, bool)
        or not isinstance(pseudocount, Real)
        or not np.isfinite(pseudocount)
        or pseudocount < 0
    ):
        raise ValueError(f"pseudocount must be a positive number or 0, got {pseudocount!r}.")

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
    if not _check_values(_sample_values(protein.X), "protein"):
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
    if calculate_neighbors_from == "rna" and not _check_values(_sample_values(data["rna"].X), "RNA"):
        warn(
            "The RNA data is not integer-valued. KNN normalization expects raw RNA counts "
            "and will library-size normalize and log-transform them.",
            stacklevel=3,
        )


def _check_positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _get_protein_adata(data):
    """
    Get protein data from the AnnData or MuData input. Also check that the input data type is correct.

    Parameters
    ----------
    data
        AnnData or MuData object with the protein data.

    Returns
    -------
    AnnData object with the protein data.
    """
    if isinstance(data, AnnData):
        return data
    if isinstance(data, MuData):
        return data["prot"]
    raise TypeError(f"Expected AnnData or MuData, got {type(data).__name__}.")


def _densify(x):
    """Return a dense float copy of x."""
    return x.toarray().astype(float, copy=False) if issparse(x) else np.array(x, dtype=float)


def _get_neighbors(data, protein, calculate_neighbors_from, n_neighbors, log_transform=True, verbose=True):
    """
    Get neighbors (KNN graph) from the desired data modality.

    Parameters
    ----------
    data
        AnnData or MuData object.
    protein
        AnnData object with the protein data.
    calculate_neighbors_from
        Calculate neighbors from protein, RNA or use existing neighbors.
    n_neighbors
        Number of neighbors for the KNN graph.
    log_transform
        Whether to log-transform the data.
    verbose
        Whether to print progress messages.

    Returns
    -------
    The KNN graph in scipy.sparse.csr_matrix format.
    """
    if calculate_neighbors_from == "prot":
        return calculate_neighbors_from_protein(protein, n_neighbors, log_transform, verbose)
    if calculate_neighbors_from == "rna":
        return calculate_neighbors_from_rna(data["rna"], n_neighbors, verbose)
    if "connectivities" not in protein.obsp:
        raise ValueError(
            "calculate_neighbors_from='use_existing_neighbors' requires "
            "protein.obsp['connectivities']. Run sc.pp.neighbors on the protein data first."
        )
    return protein.obsp["connectivities"]


def _default_n_neighbors(n_cells):
    return max(15, min(round(n_cells / 20), 300))


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


def _calculate_size_factors(x, target_cells, neighbor_cells, neighbors_per_cell, chunk_size=50_000):
    """
    Calculate the size factor of each cell.

    For every (cell, neighbor) pair, take the median over proteins of
    ``neighbor / cell`` expression. A cell's size factor is the mean of
    these medians over all of its neighbors.

    Parameters
    ----------
    x
        Dense array of protein data (pseudocount already added).
    target_cells
        For each (cell, neighbor) pair, the index of the target cell.
    neighbor_cells
        For each (cell, neighbor) pair, the index of the neighbor cell.
    neighbors_per_cell
        Number of neighbors of each cell in the KNN graph.
    chunk_size
        Number of (cell, neighbor) pairs processed at once, to bound memory use.

    Returns
    -------
    np.array of size factors
    """
    n_cells = x.shape[0]
    n_pairs = len(target_cells)
    median_ratios = np.empty(n_pairs)

    for start in range(0, n_pairs, chunk_size):
        chunk = slice(start, start + chunk_size)
        ratios = x[neighbor_cells[chunk]] / x[target_cells[chunk]]
        median_ratios[chunk] = np.median(ratios, axis=1)

    ratio_sums = np.bincount(target_cells, weights=median_ratios, minlength=n_cells)
    # Cells without neighbors keep a neutral size factor of 1.
    size_factors = np.divide(
        ratio_sums,
        neighbors_per_cell,
        out=np.ones(n_cells),
        where=neighbors_per_cell > 0,
    )
    return size_factors


def _save_size_factors(protein_anndata, size_factor_history, connectivities):
    """Saves size factors and connectivities to the resulting AnnData object."""
    history = np.array(size_factor_history)
    protein_anndata.obs["size_factor"] = history.prod(axis=0)  # cumulative factor over all iterations
    protein_anndata.obsm["size_factor_history"] = history.T
    protein_anndata.obsp["connectivities_KNN_normalization"] = connectivities
