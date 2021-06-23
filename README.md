# Stibor

Package to easily define normalizing flows and neural flows for Pytorch.

- Normalizing flows define complicated high-dimensional densities as transformations of random variables.
- Neural flows define continuous time dynamics with invertible neural networks.

## Install package and dependencies

```
pip install git+https://github.com/mbilos/stribor.git
```

## Normalizing flows

### Base densities

- Normal `st.Normal` and `st.UnitNormal` and `st.MultivariateNormal`
- Uniform `st.UnitUniform`
- Other distributions from `torch.distributions`

### Invertible transformations

- Activation functions
    - ELU `st.ELU`
    - Leaky ReLU `st.LeakyReLU`
- Affine
    - Element-wise transformation `st.Affine`
    - Fixed (non-learnable) element-wise transformation `st.AffineFixed`
    - Linear layer with PLU factorization `st.AffinePLU`
    - Matrix exponential `st.MatrixExponential`
- Coupling layer that can be combined with any element-wise transformation `st.Coupling`
- Continuous normalizing flows `st.ContinuousNormalizingFlow`
    - Differential equations with stochastic trace estimation:
        - `st.net.DiffeqMLP`
        - `st.net.DiffeqDeepset`
        - `st.net.DiffeqSelfAttention`
    - Differential equations with fixed zero trace:
        - `st.net.DiffeqZeroTraceMLP`
        - `st.net.DiffeqZeroTraceDeepSet`
        - `st.net.DiffeqZeroTraceAttention`
    - Differential equations with exact trace computation:
        - `st.net.DiffeqExactTraceMLP`
        - `st.net.DiffeqExactTraceDeepSet`
        - `st.net.DiffeqExactTraceAttention`
- Cummulative sum `st.Cumsum` and difference `st.Diff`
    - Across single column `st.CumsumColumn` and `st.DiffColumn`
- Permutations
    - Flipping the indices `st.Flip`
    - Random permutation of indices `st.Permute`
- Sigmoid `st.Sigmoid` and logit `st.Logit` function
- Spline (quadratic or cubic) element-wise transformation `st.Spline`


### Example

To define a normalizing flow, define a base distribution and a series of transformations, e.g.:
```py
import stribor as st
import torch

dim = 2
base_dist = st.UnitNormal(dim)

transforms = [
    st.Coupling(
        flow=st.Affine(dim, latent_net=st.net.MLP(dim, [64], dim)),
        mask='ordered_right_half'
    ),
    st.ContinuousNormalizingFlow(
        dim,
        net=st.net.DiffeqMLP(dim + 1, [64], dim)
    )
]

flow = st.Flow(base_dist, transforms)

x = torch.rand(1, dim)
y, ljd = flow(x)
y_inv, ljd_inv = flow.inverse(y)
```

## Run tests

```
pytest --pyargs stribor
```
