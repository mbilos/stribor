# Normalizing flows

A normalizing flow library - invertible neural networks that define complicated high-dimensional densities as transformations of random variables.

## Install package and dependencies

```
pip install -r requirements.txt
pip install -e .
```

## Generate synthetic 2D data

```
python nf/gen/toy.py --N --seed
```
Default `N=10000` and `seed=123`. Creates 10 datasets in `data/`.

## Run tests

```
pytest
```

## Usage

To define a normalizing flow, define a base distribution and a series of transformations, e.g.:
```
import nf
import torch
model = nf.Flow(torch.distributions.Normal(0, 1), [nf.Identity()])
```
```
>> model.forward(torch.Tensor([1])) # Returns y and log_jac_diag
(tensor([1.]), tensor([0.]))
>> model.sample(5) # Output will differ every time
tensor([0.1695, 1.9026, 0.4640, 0.7100, 0.2773])
```

Example runnable notebook can be found in [`notebooks/example_spline_coupling.ipynb`](notebooks/example_spline_coupling.ipynb).
