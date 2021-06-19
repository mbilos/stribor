# Normalizing flows

A normalizing flow library - invertible neural networks that define complicated high-dimensional densities as transformations of random variables.

## Install package and dependencies

```
pip install -r requirements.txt
pip install -e .
```

## Generate synthetic 2D data

```
python stribor/gen/toy.py --N 10000 --seed 123
```
Creates 10 datasets in `data/`.

## Run tests

```
pytest
```

## Usage

To define a normalizing flow, define a base distribution and a series of transformations, e.g.:
```
import stribor as st
import torch
model = st.Flow(st.Normal(0, 1), [st.Identity()])
```
```
>> model.forward(torch.Tensor([1])) # Returns y and log_jac_diag
(tensor([1.]), tensor([0.]))
>> model.sample(5) # Output will differ every time
tensor([0.1695, 1.9026, 0.4640, 0.7100, 0.2773])
```

Example runnable notebook can be found in [`notebooks/example_spline_coupling.ipynb`](notebooks/example_spline_coupling.ipynb).
