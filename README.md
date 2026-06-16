# KNN normalization

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/javier-marchena-hurtado/KNN_normalization/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/KNN_normalization

<img src="https://raw.githubusercontent.com/javier-marchena-hurtado/KNN_normalization/main/images/KNN_normalization_logo.png" width="250">

## Background and motivation

KNN normalization is a normalization method for protein counts in CITE-seq data. KNN normalization learns from neighbor cells in a KNN graph in order to estimate the appropriate total protein counts in each cell, accurately estimating total protein counts while preserving biological information.

## Installation

Install the latest release of `KNN_normalization` from [PyPI][]:

```bash
pip install KNN_normalization
```

## Basic usage

```python
import scanpy as sc
import knn_normalization as knn

# Load your CITE-seq data
adata = sc.read_h5ad("path/to/your/data.h5ad")

# Run KNN normalization (modifies adata in place)
knn.tl.knn_normalize(adata)
```

## Documentation

Please refer to the [documentation][],
in particular, the [documentation of the knn_normalize() function][].

## Installing the Development Version (optional)

If you need the latest unreleased features or bug fixes:

```bash
pip install git+https://github.com/javier-marchena-hurtado/KNN_normalization.git@main
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, please open a [discussion][] on GitHub.
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[issue tracker]: https://github.com/javier-marchena-hurtado/KNN_normalization/issues
[tests]: https://github.com/javier-marchena-hurtado/KNN_normalization/actions/workflows/test.yaml
[documentation]: https://knn-normalization.readthedocs.io
[changelog]: https://knn-normalization.readthedocs.io/en/latest/changelog.html
[documentation of the knn_normalize() function]: https://knn-normalization.readthedocs.io/en/latest/generated/knn_normalization.tl.knn_normalize.html#knn_normalization.tl.knn_normalize
[pypi]: https://pypi.org/project/KNN_normalization
[discussion]: https://github.com/javier-marchena-hurtado/KNN_normalization/discussions
