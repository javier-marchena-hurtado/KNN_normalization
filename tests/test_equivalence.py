"""Check that the fast normalize_with_neighbors gives the same results as the original (slow) algorithm."""

import numpy as np
import pytest

from knn_normalization import normalize_with_neighbors
from knn_normalization._utils import _densify, _neighbor_pairs


def _calculate_size_factors(x, target_cells, neighbor_cells, neighbors_per_cell, chunk_size=50_000):
    """Original implementation: median ratios of all (cell, neighbor) pairs, recomputed from the data."""
    n_cells = x.shape[0]
    median_ratios = np.empty(len(target_cells))
    for start in range(0, len(target_cells), chunk_size):
        chunk = slice(start, start + chunk_size)
        ratios = x[neighbor_cells[chunk]] / x[target_cells[chunk]]
        median_ratios[chunk] = np.median(ratios, axis=1)
    ratio_sums = np.bincount(target_cells, weights=median_ratios, minlength=n_cells)
    return np.divide(ratio_sums, neighbors_per_cell, out=np.ones(n_cells), where=neighbors_per_cell > 0)


def _normalize_reference(
    adata, connectivities, log_transform, pseudocount=5, max_iterations=25, change_for_stop=0.0005
):
    """Original implementation: recomputes the median ratios of every (cell, neighbor) pair in each iteration."""
    x = _densify(adata.X)
    x += pseudocount
    total_counts_before = x.sum()
    size_factor_history = []
    target_cells, neighbor_cells, neighbors_per_cell = _neighbor_pairs(connectivities)

    for iteration in range(max_iterations):
        size_factors = _calculate_size_factors(x, target_cells, neighbor_cells, neighbors_per_cell)
        size_factors /= (size_factors @ x.sum(axis=1)) / total_counts_before
        x *= size_factors[:, None]
        size_factor_history.append(size_factors)
        if iteration > 0 and np.max(np.abs(size_factor_history[-1] - size_factor_history[-2])) < change_for_stop:
            break

    if log_transform:
        np.log(x, out=x)
    return x, np.array(size_factor_history)


def _isolate_first_cell(connectivities):
    """Return a copy of the graph in which the first cell has no neighbors."""
    graph = connectivities.tolil()
    graph[0, :] = 0
    return graph.tocsr()


@pytest.mark.parametrize("log_transform", [True, False])
@pytest.mark.parametrize("change_for_stop", [0.0005, 0.0])  # 0.0 never stops early: runs all 25 iterations
@pytest.mark.parametrize("isolated_cell", [False, True])
def test_fast_normalization_matches_reference(adata, log_transform, change_for_stop, isolated_cell):
    """The fast version must reach the same normalized data and size factors as the original algorithm."""
    connectivities = adata.obsp["connectivities"]
    if isolated_cell:
        connectivities = _isolate_first_cell(connectivities)
    expected_x, expected_history = _normalize_reference(
        adata, connectivities, log_transform=log_transform, change_for_stop=change_for_stop
    )

    normalize_with_neighbors(
        adata,
        connectivities,
        log_transform=log_transform,
        change_for_stop=change_for_stop,
        save_size_factors=True,
        verbose=False,
    )

    np.testing.assert_allclose(adata.X, expected_x, rtol=1e-7)
    np.testing.assert_allclose(adata.obsm["size_factor_history"], expected_history.T, rtol=1e-7)
    np.testing.assert_allclose(adata.obs["size_factor"], expected_history.prod(axis=0), rtol=1e-7)
