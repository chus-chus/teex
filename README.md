# teex

A Python **T**oolbox for the **E**valuation of machine learning **Ex**planations.

This project aims to provide a simple way of evaluating all kinds of individual black box explanations. Moreover, it contains a collection
of easy-to-access datasets with available ground truth explanations.

## Installation

The teex package is on [PyPI](https://pypi.org/project/teex/). To install it, simply run

```shell
pip install teex
```

Note that Python >= 3.5 is required.

## Tutorials and API

The full API documentation can be found on [Read The Docs](https://teex.readthedocs.io).

Here are some sample notebooks on basic usages and examples:

- [Generating image data with g.t. saliency map explanations](https://github.com/chus-chus/teex/blob/main/docs/_demos/notebooks/gen_saliency_maps_seneca.ipynb)

### Datasets

To use a dataset, simply search the one you want in the API documentation and:

```python
from teex import datasets

data = datasets.Kahikatea()
X, y, explanations = data[:100]
```

## Contributing

Before contributing to teex, please take a moment to read the [manual]((https://github.com/chus-chus/teex/blob/main/CONTRIBUTING.md)).
