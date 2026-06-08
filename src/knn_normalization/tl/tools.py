from numbers import Integral
from typing import Literal
from warnings import warn

import mudata
import numpy as np
import scanpy as sc
from anndata import AnnData
from mudata import MuData
from scipy import stats

from knn_normalization.pp.preprocessing import calculate_neighbors_from_protein, retrieve_neighbors

mudata.set_options(pull_on_update=False)


def knn_normalize(
    data: AnnData | MuData,
    calculate_neighbors_from: Literal["prot", "rna", "use_existing_neighbors"] = "prot",
    preprocess_rna: bool = True,
    log_transform: bool = True,
    n_neighbors: Integral | None = None,
    pseudocount: Integral = 5,
    max_iterations: Integral = 25,
    mean: Literal["average", "geom_mean", "trimmed_mean"] = "average",
    inplace: bool = True,
    save_size_factors: bool = False,
    verbose: bool = True,
    preserve_total_counts=True,
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
    mean
        Type of mean to use: ``'average'``, ``'geom_mean'``, or ``'trimmed_mean'``.
    inplace
        Whether to update the object in place or return a copy.
    save_size_factors
        If ``True``, saves the final size factors to ``data.obs['size_factor']``
        and the size factor history to ``data.obsm['size_factor_history']``.
    verbose
        Whether to print progress messages.
    preserve_total_counts
        Whether to preserve total counts across iterations.

    Returns
    -------
    Normalized data will be written to ``data`` (if it is an AnnData object) or
    ``data.mod['prot']`` (if it is a MuData object) as an X matrix. If inplace is False,
    returns a new AnnData object (if input is AnnData) or a new MuData object (if input is MuData)
    with the normalized data.
    """
    toreturn = None

    assert calculate_neighbors_from in ["prot", "rna", "use_existing_neighbors"], (
        "the argument calculate_neighbors_from must be prot, rna or use_existing_neighbors."
    )

    if isinstance(data, AnnData):
        if ("log1p" in data.uns) and log_transform:
            warn("The protein data might already be log-transformed.", stacklevel=2)

        if n_neighbors is None:
            n_cells = data.n_obs
            n_neighbors = max(15, min(round(n_cells / 20), 300))

        if calculate_neighbors_from == "prot":
            neighbors = calculate_neighbors_from_protein(data, n_neighbors=n_neighbors, log_transform=log_transform)
        elif calculate_neighbors_from == "use_existing_neighbors":
            neighbors = data.obsp["connectivities"]
        assert calculate_neighbors_from != "rna", (
            "If an AnnData object with the protein data is provided, ``calculate_neighbors_from`` cannot be ``rna``. If calculate_neighbors_from = rna is desired, please provide a MuData object with the protein and the rna data."
        )

        if inplace:
            _normalize_with_neighbors(
                data,
                neighbors,
                log_transform=log_transform,
                pseudocount=pseudocount,
                max_iterations=max_iterations,
                verbose=verbose,
                save_size_factors=save_size_factors,
                inplace=True,
                mean=mean,
                preserve_total_counts=preserve_total_counts,
            )
        else:
            knn_normalized_protein = _normalize_with_neighbors(
                data,
                neighbors,
                log_transform=log_transform,
                pseudocount=pseudocount,
                max_iterations=max_iterations,
                verbose=verbose,
                save_size_factors=save_size_factors,
                inplace=False,
                mean=mean,
                preserve_total_counts=preserve_total_counts,
            )
            toreturn = knn_normalized_protein

    elif isinstance(data, MuData):
        assert "prot" in data.mod, (
            "The MuData object does not have a modality called ``prot``, please add a modality called ``prot`` with the protein data."
        )

        if ("log1p" in data["prot"].uns) and log_transform:
            warn("The protein data might already be log-transformed.", stacklevel=2)

        if n_neighbors is None:
            n_cells = data.n_obs
            n_neighbors = max(15, min(round(n_cells / 20), 300))

        if calculate_neighbors_from == "prot":
            neighbors = calculate_neighbors_from_protein(
                data["prot"], n_neighbors=n_neighbors, log_transform=log_transform
            )
        elif calculate_neighbors_from == "rna":
            assert "rna" in data.mod, (
                "The MuData object does not have a modality called ``rna``, please add a modality called ``rna`` in order to calculate neighbors from the RNA data."
            )
            data_for_neighbors = data["rna"].copy()
            if preprocess_rna:
                sc.pp.normalize_total(data_for_neighbors)
                sc.pp.log1p(data_for_neighbors)
            sc.pp.pca(data_for_neighbors)
            sc.pp.neighbors(data_for_neighbors, n_neighbors=n_neighbors)
            neighbors = data_for_neighbors.obsp["connectivities"]

        elif calculate_neighbors_from == "use_existing_neighbors":
            neighbors = data["prot"].obsp["connectivities"]

        if inplace:
            _normalize_with_neighbors(
                data["prot"],
                neighbors,
                log_transform=log_transform,
                pseudocount=pseudocount,
                max_iterations=max_iterations,
                verbose=verbose,
                save_size_factors=save_size_factors,
                inplace=True,
                mean=mean,
                preserve_total_counts=preserve_total_counts,
            )
        else:
            knn_normalized_protein = _normalize_with_neighbors(
                data["prot"],
                neighbors,
                log_transform=log_transform,
                pseudocount=pseudocount,
                max_iterations=max_iterations,
                verbose=verbose,
                save_size_factors=save_size_factors,
                inplace=False,
                mean=mean,
                preserve_total_counts=preserve_total_counts,
            )
            new_mdata = MuData({"rna": data["rna"], "prot": knn_normalized_protein})
            toreturn = new_mdata

    return toreturn


def _normalize_with_neighbors(
    protein_anndata,
    connectivities,
    log_transform=True,
    log_transform_before=False,
    save_size_factors=False,
    pseudocount=5,
    max_iterations=25,
    change_for_stop=0.0005,
    verbose=True,
    inplace: bool = True,
    mean="average",
    preserve_total_counts=True,
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
    log_transform_before
        If ``True``, log-transforms the data before normalization.
        Cannot be ``True`` at the same time as ``log_transform``.
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
    mean
        Type of mean to use: ``'average'``, ``'geom_mean'``, or
        ``'trimmed_mean'``.
    preserve_total_counts
        Whether to preserve total counts across iterations.

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

    x = protein_anndata.X
    x += pseudocount  # To avoid zero-division, we add a pseudocount.

    assert not (log_transform_before and log_transform), "log_transform and log_transform_before cannot be both True"
    if log_transform_before:
        x = np.log(x)
    total_sums_before = x.sum()

    num_cells = x.shape[0]
    size_factor_history = []

    # KNN normalization.
    for iteration in range(max_iterations):
        size_factors = np.zeros(num_cells)

        for target_cell, neighbor_list in neighbors.items():
            neighbor_indices = np.array(neighbor_list)
            target_indices = np.full(len(neighbor_list), target_cell)

            ratios = x[neighbor_indices] / x[target_indices]
            proto_size_factors = np.median(ratios, axis=1)

            # After having collected the ratios for between the neighbor cells and the target cell, we calculate the average of those ratios. That will be the cell-specific size factor.
            if mean == "average":
                size_factor = np.mean(proto_size_factors)
            elif mean == "trimmed_mean":
                size_factor = stats.trim_mean(proto_size_factors, 0.1)
            else:
                size_factor = stats.gmean(proto_size_factors)

            size_factors[target_cell] = size_factor

        # Now, we multiply the protein expression of each cell by its cell-specific factor.
        x *= size_factors[:, None]

        if preserve_total_counts:
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
        total_size_factors = np.prod(
            np.array(size_factor_history), axis=0
        )  # Multiplication of the size factors across all iterations.
        protein_anndata.obs["size_factor"] = total_size_factors
        size_factor_history = np.array(size_factor_history).T
        protein_anndata.obsm["size_factor_history"] = size_factor_history
        protein_anndata.obsp["connectivities_KNN_normalization"] = connectivities

    protein_anndata.X = x

    return None if inplace else protein_anndata


# def _normalize_with_neighbors(
#     protein_anndata,
#     connectivities,
#     log_transform=True,
#     log_transform_before=False,
#     save_size_factors=False,
#     pseudocount=5,
#     max_iterations=25,
#     change_for_stop=0.0005,
#     verbose=True,
#     inplace: bool = True,
#     mean="average",
# ):
#     """
#     Applies KNN normalization given precomputed neighbors.

#     protein_data: an AnnData object with the protein data in CITE-seq.
#     connectivities:  The KNN graph containing neighbor cells. It expects the format from .obsp["connectivities"].
#     log_transform: if True, takes the logarithm of the data.
#     save_size_factors: if True, the final size factors are saved to protein_anndata.obs["size_factor"] and the size factor history (all size factors across iterations) is saved to protein_anndata.obsm["size_factor_history"].
#     pseudocount: adds pseudocounts to the data to avoid ZeroDivision errors. This argument also determines the value of the pseudocount (5 by default).
#     max_iteration: maximum number of iterations.
#     change_for_stop: the algorithm stops when the change in size factor is smaller than this value (convergence criterion).
#     verbose: whether you want to print guidance information when running the function.
#     """
#     neighbors = retrieve_neighbors(
#         connectivities
#     )  # Converts the format of the KNN graph into a dictionary mapping each cell to its neighbor cells.

#     if not inplace:
#         protein_anndata = protein_anndata.copy()

#     # TODO: FUNCTIONS IN CASE THE DATA IS SPARSE.

#     x = protein_anndata.X
#     x += pseudocount  # To avoid zero-division, we add a pseudocount.

#     assert not (log_transform_before and log_transform), "log_transform and log_transform_before cannot be both True"
#     if log_transform_before:
#         x = np.log(x)

#     num_cells = x.shape[0]
#     size_factor_history = []

#     # KNN normalization.
#     for iteration in range(max_iterations):
#         size_factors = np.zeros(num_cells)

#         for target_cell, neighbor_list in neighbors.items():
#             neighbor_indices = np.array(neighbor_list)
#             target_indices = np.full(len(neighbor_list), target_cell)
#             ratios = x[neighbor_indices] / x[target_indices]
#             proto_size_factors = np.median(ratios, axis=1)

#             # After having collected the ratios for between the neighbor cells and the target cell, we calculate the average of those ratios. That will be the cell-specific size factor.
#             if mean == "average":
#                 size_factor = np.mean(proto_size_factors)
#             else:
#                 size_factor = stats.gmean(proto_size_factors)
#             size_factors[target_cell] = size_factor

#         # Now, we multiply the protein expression of each cell by its cell-specific factor.
#         x *= size_factors[:, None]

#         # Save this iteration's size factors. This is done mainly to compare with the previous iteration for the stopping criterion.
#         size_factor_history.append(size_factors)
#         if verbose:
#             print("Iteration ", iteration + 1)

#         # Unless it's the first iteration, check the algorithm stopping criterion: if all changes of size_factors are smaller than the "change_for_stop" value with respect to the previous iteration.

#         if iteration > 0:
#             biggest_size_factor_change = np.max(np.abs(size_factor_history[-1] - size_factor_history[-2]))
#             if verbose:
#                 print("Change wrt previous iteration:", biggest_size_factor_change)
#             if biggest_size_factor_change < change_for_stop:
#                 break

#     if log_transform:
#         x = np.log(x)

#     if save_size_factors:
#         total_size_factors = np.prod(
#             np.array(size_factor_history), axis=0
#         )  # Multiplication of the size factors across all iterations.
#         protein_anndata.obs["size_factor"] = total_size_factors
#         size_factor_history = np.array(size_factor_history).T
#         protein_anndata.obsm["size_factor_history"] = size_factor_history
#         protein_anndata.obsp["connectivities_KNN_normalization"] = connectivities

#     protein_anndata.X = x

#     return None if inplace else protein_anndata


# def _normalize_with_neighbors(
#     protein_anndata,
#     neighbors,
#     log_transform=True,
#     log_transform_before=False,
#     save_size_factors=False,
#     pseudocount=5,
#     max_iterations=25,
#     change_for_stop=0.0005,
#     verbose=True,
#     inplace: bool = True,
#     mean="average",
# ):
#     """
#     Applies KNN normalization given precomputed neighbors.

#     protein_data: an AnnData object with the protein data in CITE-seq.
#     neighbors:  Neighbor cells. These neighbors are retrieved with the "retrieve_neighbors" function. The expected format a is dictionary of lists indicating which cells are neighbors.
#     log_transform: if True, takes the logarithm of the data.
#     save_size_factors: if True, the final size factors are saved to protein_anndata.obs["size_factor"] and the size factor history (all size factors across iterations) is saved to protein_anndata.obsm["size_factor_history"].
#     pseudocount: adds pseudocounts to the data to avoid ZeroDivision errors. This argument also determines the value of the pseudocount (5 by default).
#     max_iteration: maximum number of iterations.
#     change_for_stop: the algorithm stops when the change in size factor is smaller than this value (convergence criterion).
#     verbose: whether you want to print guidance information when running the function.
#     """
#     if not inplace:
#         protein_anndata = protein_anndata.copy()

#     # TODO: FUNCTIONS IN CASE THE DATA IS SPARSE.

#     x = protein_anndata.X
#     x += pseudocount  # To avoid zero-division, we add a pseudocount.

#     assert not (log_transform_before and log_transform), "log_transform and log_transform_before cannot be both True"
#     if log_transform_before:
#         x = np.log(x)

#     num_cells = x.shape[0]
#     size_factor_history = []

#     # KNN normalization.
#     for iteration in range(max_iterations):
#         size_factors = np.zeros(num_cells)

#         for target_cell, neighbor_list in neighbors.items():
#             neighbor_indices = np.array(neighbor_list)
#             target_indices = np.full(len(neighbor_list), target_cell)
#             ratios = x[neighbor_indices] / x[target_indices]
#             proto_size_factors = np.median(ratios, axis=1)

#             # After having collected the ratios for between the neighbor cells and the target cell, we calculate the average of those ratios. That will be the cell-specific size factor.
#             if mean == "average":
#                 size_factor = np.mean(proto_size_factors)
#             else:
#                 size_factor = stats.gmean(proto_size_factors)
#             size_factors[target_cell] = size_factor

#         # Now, we multiply the protein expression of each cell by its cell-specific factor.
#         x *= size_factors[:, None]

#         # Save this iteration's size factors. This is done mainly to compare with the previous iteration for the stopping criterion.
#         size_factor_history.append(size_factors)
#         if verbose:
#             print("Iteration ", iteration + 1)

#         # Unless it's the first iteration, check the algorithm stopping criterion: if all changes of size_factors are smaller than the "change_for_stop" value with respect to the previous iteration.

#         if iteration > 0:
#             biggest_size_factor_change = np.max(np.abs(size_factor_history[-1] - size_factor_history[-2]))
#             if verbose:
#                 print("Change wrt previous iteration:", biggest_size_factor_change)
#             if biggest_size_factor_change < change_for_stop:
#                 break

#     if log_transform:
#         x = np.log(x)

#     if save_size_factors:
#         total_size_factors = np.prod(
#             np.array(size_factor_history), axis=0
#         )  # Multiplication of the size factors across all iterations.
#         protein_anndata.obs["size_factor"] = total_size_factors
#         size_factor_history = np.array(size_factor_history).T
#         protein_anndata.obsm["size_factor_history"] = size_factor_history

#     protein_anndata.X = x

#     return None if inplace else protein_anndata
