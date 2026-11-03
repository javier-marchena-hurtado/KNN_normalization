from numbers import Integral, Real
from typing import Literal
from warnings import warn

import mudata
import numpy as np
from anndata import AnnData
from mudata import MuData
from scipy.sparse import csr_matrix, issparse

from knn_normalization.pp.preprocessing import (
    calculate_neighbors_from_protein,
    calculate_neighbors_from_rna,
)

if hasattr(mudata, "settings"):
    mudata.settings.pull_on_update = False
else:
    mudata.set_options(pull_on_update=False)

_NEIGHBOR_SOURCES = ("prot", "rna", "use_existing_neighbors")


def knn_normalize(
    data: MuData | AnnData,
    calculate_neighbors_from: Literal["prot", "rna", "use_existing_neighbors"] = "rna",
    preprocess_rna: bool = True,
    log_transform: bool = True,
    n_neighbors: Integral | None = None,
    pseudocount: Real = 5,
    max_iterations: Integral = 25,
    inplace: bool = True,
    save_size_factors: bool = False,
    verbose: bool = True,
):
    """
    Normalize protein expression with KNN normalization.

    Parameters
    ----------
    data
        MuData object with ``prot`` and ``rna`` modalities or
        AnnData object with protein expression counts.
    calculate_neighbors_from
        Whether to use the ``prot`` or the ``rna`` modality to calculate neighbor
        cells. If ``use_existing_neighbors``, the neighbors already present in the
        protein data will be used.
    preprocess_rna
        If using RNA to calculate neighbors, whether to preprocess the RNA data
        with library size normalization and log-transformation.
    n_neighbors
        Number of neighbors. If ``None``, calculated automatically as
        ``max(15, min(round(n_cells / 20), 300))``.
    log_transform
        Whether to log-transform the protein data.
    pseudocount
        Pseudocount to add before normalization to avoid zero-division errors.
    max_iterations
        Maximum number of iterations.
    inplace
        Whether to update the object in place or return a copy.
    save_size_factors
        If ``True``, saves the final size factors to ``data.obs['size_factor']``
        and the size factor history to ``data.obsm['size_factor_history']``.
    verbose
        Whether to print progress messages.

    Returns
    -------
    Normalized data will be written to ``data.mod['prot']`` (if it is a MuData object) or to
    ``data`` (if it is an AnnData object) as an X matrix. If inplace is False, returns
    a new MuData object (if input is MuData) or a new AnnData object (if input is AnnData)
    with the normalized data.
    """
    if calculate_neighbors_from not in _NEIGHBOR_SOURCES:
        raise ValueError(f"calculate_neighbors_from must be one of {_NEIGHBOR_SOURCES}.")
    if isinstance(data, MuData) and not {"prot", "rna"} <= data.mod.keys():
        raise ValueError("A MuData input needs both a 'prot' and an 'rna' modality.")
    if calculate_neighbors_from == "rna" and isinstance(data, AnnData):
        raise ValueError("calculate_neighbors_from='rna' requires a MuData object with 'prot' and 'rna' modalities.")
    protein = _get_protein_adata(data)
    if not inplace:
        data = data.copy()
    if ("log1p" in protein.uns) and log_transform:
        warn("The protein data might already be log-transformed, consider using log_transform=False", stacklevel=2)
    if n_neighbors is None:
        n_neighbors = _default_n_neighbors(data.n_obs)
    neighbors = _get_neighbors(
        data,
        protein,
        calculate_neighbors_from,
        n_neighbors,
        preprocess_rna=preprocess_rna,
        log_transform=log_transform,
        verbose=verbose,
    )
    normalize_with_neighbors(
        protein,
        neighbors,
        log_transform=log_transform,
        pseudocount=pseudocount,
        max_iterations=max_iterations,
        verbose=verbose,
        save_size_factors=save_size_factors,
    )

    return None if inplace else data


def _densify(x):
    """Return a dense float copy of x."""
    return x.toarray().astype(float, copy=False) if issparse(x) else np.array(x, dtype=float)


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


def _get_neighbors(
    data, protein, calculate_neighbors_from, n_neighbors, preprocess_rna, log_transform=True, verbose=True
):
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
    preprocess_rna
        If the KNN graph is calculated from RNA, whether to normalize and log-transform the RNA data.
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
        return calculate_neighbors_from_rna(data["rna"], n_neighbors, preprocess_rna, verbose)
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


def normalize_with_neighbors(
    protein_anndata: AnnData,
    connectivities: csr_matrix,
    log_transform=True,
    save_size_factors=False,
    pseudocount=5,
    max_iterations=25,
    change_for_stop=0.0005,
    verbose=True,
):
    """
    Apply KNN normalization given precomputed neighbors.

    Parameters
    ----------
    protein_anndata
        AnnData object with protein expression data.
    connectivities
        KNN graph containing neighbor cells, in the format of
        ``.obsp["connectivities"]``.
    log_transform
        If ``True``, log-transforms the data after normalization.
    save_size_factors
        If ``True``, saves the final size factors to
        ``protein_anndata.obs["size_factor"]`` and the size factor history
        to ``protein_anndata.obsm["size_factor_history"]``.
    pseudocount
        Pseudocount added to avoid zero-division errors.
    max_iterations
        Maximum number of iterations.
    change_for_stop
        Convergence criterion. The algorithm stops when the change in size
        factor between iterations is smaller than this value.
    verbose
        Whether to print progress messages.
    """
    x = _densify(protein_anndata.X)
    x += pseudocount
    total_counts_before = x.sum()
    size_factor_history = []
    target_cells, neighbor_cells, neighbors_per_cell = _neighbor_pairs(connectivities)

    for iteration in range(max_iterations):
        size_factors = _calculate_size_factors(x, target_cells, neighbor_cells, neighbors_per_cell)
        # Rescale so that the total counts in the matrix stay constant.
        size_factors /= (size_factors @ x.sum(axis=1)) / total_counts_before
        x *= size_factors[:, None]
        size_factor_history.append(size_factors)

        if verbose:
            print(f"Iteration {iteration + 1}")
        if iteration > 0:
            change = np.max(np.abs(size_factors - size_factor_history[-2]))
            if verbose:
                print(f"Change wrt previous iteration: {change}")
            if change < change_for_stop:
                break
    if log_transform:
        np.log(x, out=x)
    if save_size_factors:
        _save_size_factors(protein_anndata, size_factor_history, connectivities)

    protein_anndata.X = x
