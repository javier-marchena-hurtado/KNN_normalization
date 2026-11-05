from numbers import Integral, Real
from typing import Literal

import mudata
import numpy as np
from anndata import AnnData
from mudata import MuData
from scipy.sparse import csr_matrix

from knn_normalization.tl._utils import (
    _calculate_size_factors,
    _default_n_neighbors,
    _densify,
    _get_neighbors,
    _get_protein_adata,
    _neighbor_pairs,
    _save_size_factors,
    _validate_inputs,
)

if hasattr(mudata, "settings"):
    mudata.settings.pull_on_update = False
else:
    mudata.set_options(pull_on_update=False)


def knn_normalize(
    data: MuData | AnnData,
    calculate_neighbors_from: Literal["prot", "rna", "use_existing_neighbors"] = "rna",
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
        AnnData object with protein expression counts. The function expects
        raw counts.
    calculate_neighbors_from
        Whether to calculate neighbors from protein or RNA modalities.
        ``"rna"`` requires a MuData with ``prot`` and ``rna`` modalities. The RNA data
        in ``.X`` must be **raw counts**: they are library-size normalized and
        log-transformed internally. ``"prot"`` works for MuData and AnnData.
        ``"use_existing_neighbors"`` uses ``protein.obsp["connectivities"]``,
        which must exist and be a square graph over all cells.
    n_neighbors
        Number of neighbors. If ``None``, calculated automatically as
        ``max(15, min(round(n_cells / 20), 300))``.
    log_transform
        Whether to log-transform the protein data.
    pseudocount
        Pseudocount to add before normalization to avoid zero-division errors.
        If you don't want to use pseudocounts, because you have no zeroes in the
        protein data, pass pseudocount=0.
    max_iterations
        Maximum number of iterations. If KNN normalization takes long to run,
        consider lowering the max_iterations to 10.
    inplace
        Whether to update the object in place or return a copy.
    save_size_factors
        If ``True``, saves the final size factors to the protein AnnData as
        ``.obs['size_factor']`` and the size factor history as
        ``.obsm['size_factor_history']``.
    verbose
        Whether to print progress messages.

    Returns
    -------
    ``None`` if ``inplace=True``: the object is modified and nothing is returned.
      If ``inplace=False``, returns a normalized copy and
      the input is left untouched.

    Notes
    -----
    Counts must be stored in ``.X`` (not in layers or ``.raw``).
    """
    _validate_inputs(
        data,
        calculate_neighbors_from,
        log_transform,
        n_neighbors,
        pseudocount,
        max_iterations,
        inplace,
        save_size_factors,
    )
    if not inplace:
        data = data.copy()
    protein = _get_protein_adata(data)
    if n_neighbors is None:
        n_neighbors = _default_n_neighbors(data.n_obs)
    neighbors = _get_neighbors(
        data,
        protein,
        calculate_neighbors_from,
        n_neighbors,
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

    if verbose:
        print("Running KNN normalization")

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
    protein_anndata.uns["knn_normalization"] = {"pseudocount": pseudocount, "log_transform": log_transform}
