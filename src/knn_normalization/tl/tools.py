from numbers import Integral
from typing import Literal
from warnings import warn

import mudata
import numpy as np
from anndata import AnnData
from mudata import MuData
from scipy.sparse import csr_matrix

from knn_normalization.pp.preprocessing import (
    calculate_neighbors_from_protein,
    calculate_neighbors_from_rna,
    retrieve_neighbors,
)

if hasattr(mudata, "settings"):
    mudata.settings.pull_on_update = False
else:
    mudata.set_options(pull_on_update=False)

_NEIGHBOR_SOURCES = ("prot", "rna", "use_existing_neighbors")


def knn_normalize(
    data: AnnData | MuData,
    calculate_neighbors_from: Literal["prot", "rna", "use_existing_neighbors"] = "prot",
    preprocess_rna: bool = True,
    log_transform: bool = True,
    n_neighbors: Integral | None = None,
    pseudocount: Integral = 5,
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
        AnnData object with protein expression counts or MuData object with
        ``prot`` and ``rna`` modalities.
    calculate_neighbors_from
        Whether to use the ``prot`` or the ``rna`` modality to calculate neighbor
        cells. If ``use_existing_neighbors``, the neighbors already present in the
        protein data will be used.
    preprocess_rna
        If using RNA to calculate neighbors, whether to preprocess the RNA data
        with library size normalization and log-transformation first.
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
    Normalized data will be written to ``data`` (if it is an AnnData object) or
    ``data.mod['prot']`` (if it is a MuData object) as an X matrix. If inplace is False,
    returns a new AnnData object (if input is AnnData) or a new MuData object (if input is MuData)
    with the normalized data.
    """
    if calculate_neighbors_from not in _NEIGHBOR_SOURCES:
        raise ValueError(f"calculate_neighbors_from must be one of {_NEIGHBOR_SOURCES}.")
    if isinstance(data, AnnData) and calculate_neighbors_from == "rna":
        raise ValueError("calculate_neighbors_from='rna' requires a MuData object with 'prot' and 'rna'.")
    if "rna" not in data.mod and calculate_neighbors_from == "rna":
        raise ValueError("The option calculate_neighbors_from == 'rna' requires a modality called 'rna'.")

    protein = _get_protein_adata(data)
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
    result = _normalize_with_neighbors(
        protein,
        neighbors,
        log_transform=log_transform,
        pseudocount=pseudocount,
        max_iterations=max_iterations,
        verbose=verbose,
        save_size_factors=save_size_factors,
        inplace=inplace,
    )
    if inplace:
        return None
    if isinstance(data, AnnData):
        return result
    elif isinstance(data, MuData):
        return MuData({"rna": data["rna"], "prot": result})


def _get_protein_adata(data):

    if isinstance(data, AnnData):
        return data
    elif isinstance(data, MuData):
        if "prot" not in data.mod:
            raise ValueError("The MuData object needs a modality called 'prot'.")
        return data["prot"]
    raise TypeError(f"Expected AnnData or MuData, got {type(data).__name__}.")


def _get_neighbors(
    data, protein, calculate_neighbors_from, n_neighbors, preprocess_rna, log_transform=True, verbose=True
):

    if calculate_neighbors_from == "prot":
        neighbors = calculate_neighbors_from_protein(protein, n_neighbors, log_transform, verbose)
    elif calculate_neighbors_from == "rna":
        neighbors = calculate_neighbors_from_rna(data["rna"], n_neighbors, preprocess_rna, verbose)
    elif calculate_neighbors_from == "use_existing_neighbors":
        neighbors = protein.obsp["connectivities"]

    return neighbors


def _default_n_neighbors(n_cells):
    return max(15, min(round(n_cells / 20), 300))


def _normalize_with_neighbors(
    protein_anndata: AnnData,
    connectivities: csr_matrix,
    log_transform=True,
    save_size_factors=False,
    pseudocount=5,
    max_iterations=25,
    change_for_stop=0.0005,
    verbose=True,
    inplace: bool = True,
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
    inplace
        Whether to update the AnnData object in place or return a copy.

    Returns
    -------
        Normalized data will be written to ``protein_anndata.X``. If inplace is False,
        returns a new AnnData object with the normalized data.
    """
    neighbors = retrieve_neighbors(
        connectivities
    )  # Converts the format of the KNN graph into a dictionary mapping each cell to its neighbor cells.

    if not inplace:
        protein_anndata = protein_anndata.copy()

    # TODO: FUNCTIONS IN CASE THE DATA IS SPARSE.

    x = protein_anndata.X + pseudocount

    total_sums_before = x.sum()

    num_cells = x.shape[0]
    size_factor_history = []

    if verbose:
        print("Running KNN normalization")

    # KNN normalization.
    for iteration in range(max_iterations):
        size_factors = np.zeros(num_cells)

        for target_cell, neighbor_list in neighbors.items():
            neighbor_indices = np.array(neighbor_list)
            target_indices = np.full(len(neighbor_list), target_cell)

            ratios = x[neighbor_indices] / x[target_indices]
            proto_size_factors = np.median(ratios, axis=1)

            # After having collected the ratios for between the neighbor cells and the target cell, we calculate the average of those ratios. That will be the cell-specific size factor.
            size_factor = np.mean(proto_size_factors)
            size_factors[target_cell] = size_factor

        # Now, we multiply the protein expression of each cell by its cell-specific factor.
        x *= size_factors[:, None]

        # We preserve total counts, that is, the sum of all total counts is the same before and after normalization.
        total_sums_after_iteration = x.sum()
        ratio_of_total_counts = total_sums_after_iteration / total_sums_before
        x = x / ratio_of_total_counts
        size_factors = size_factors / ratio_of_total_counts

        # Save this iteration's size factors. This is done mainly to compare with the previous iteration for the stopping criterion.
        size_factor_history.append(size_factors)
        if verbose:
            print("Iteration ", iteration + 1)

        # Unless it's the first iteration, check the algorithm stopping criterion: if all changes of size_factors are smaller than the "change_for_stop" value with respect to the previous iteration.
        if iteration > 0:
            biggest_size_factor_change = np.max(np.abs(size_factor_history[-1] - size_factor_history[-2]))
            if verbose:
                print("Change wrt previous iteration:", biggest_size_factor_change)
            if biggest_size_factor_change < change_for_stop:
                break

    if log_transform:
        x = np.log(x)

    if save_size_factors:
        history = np.array(size_factor_history)
        protein_anndata.obs["size_factor"] = history.prod(
            axis=0
        )  # Multiplication of the size factors across all iterations.
        protein_anndata.obsm["size_factor_history"] = history.T
        protein_anndata.obsp["connectivities_KNN_normalization"] = connectivities

    protein_anndata.X = x

    return None if inplace else protein_anndata
