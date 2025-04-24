import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from mudata import MuData


@pytest.fixture
def adata():
    np.random.seed(0)
    data = np.random.poisson(10, (100, 20)).astype(float)  # 100 cells, 5 proteins
    adata = AnnData(X=data)
    sc.pp.neighbors(adata)

    return adata

@pytest.fixture
def mdata():
    np.random.seed(0)
    prot_data = AnnData(X=np.random.poisson(10, (100, 20)).astype(float))
    rna_data = AnnData(X=np.random.poisson(5, (100, 100)).astype(float))
    mdata = MuData({"prot": prot_data, "rna": rna_data})
    return mdata
