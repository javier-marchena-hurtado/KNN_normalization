# KNN_normalization

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/javier-marchena-hurtado/KNN_normalization/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/KNN_normalization

<img src="images/KNN_normalization_logo.png" width="250">

## Background and motivation

KNN normalization is a normalization method for protein counts in CITE-seq data. KNN normalization learns from neighbor cells in a KNN graph in order to estimate the appropriate total protein counts in each cell. KNN normalization accurately estimates total protein counts while preserving biological information.

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation


<!--
1) Install the latest release of `KNN_normalization` from [PyPI][]:

```bash
pip install KNN_normalization
```
-->
 Install the latest development version:

```bash
pip install git+https://github.com/javier-marchena-hurtado/KNN_normalization.git@main
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in .
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/javier-marchena-hurtado/KNN_normalization/issues
[tests]: https://github.com/javier-marchena-hurtado/KNN_normalization/actions/workflows/test.yaml
[documentation]: https://KNN_normalization.readthedocs.io
[changelog]: https://KNN_normalization.readthedocs.io/en/latest/changelog.html
[api documentation]: https://KNN_normalization.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/KNN_normalization
