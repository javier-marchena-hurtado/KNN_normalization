import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData


@pytest.fixture
def adata():
    np.random.seed(0)
    data = np.random.poisson(10, (10, 5)).astype(float)  # 10 cells, 5 proteins
    adata = AnnData(X=data)
    sc.pp.log1p(adata)
    sc.pp.neighbors(adata)

    return adata
