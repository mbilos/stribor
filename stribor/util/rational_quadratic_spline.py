# https://github.com/bayesiains/nsf/blob/master/nde/transforms/splines/rational_quadratic.py

import stribor as st
import torch
from torch.nn import functional as F
import numpy as np

__all__ = ['rational_quadratic_spline', 'unconstrained_rational_quadratic_spline']

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def unconstrained_rational_quadratic_spline(inputs,
                                            unnorm_widths,
                                            unnorm_heights,
                                            unnorm_derivatives,
                                            inverse=False,
                                            lower=-1,
                                            upper=1,
                                            left=None,
                                            right=None,
                                            bottom=None,
                                            top=None,
                                            tails='linear',
                                            min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                                            min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                                            min_derivative=DEFAULT_MIN_DERIVATIVE):
    """
    Takes inputs and unnormalized parameters for width, height and
    derivatives of spline bins. Normalizes parameters and applies quadratic spline.

    Args:
        inputs: (..., dim)
        unnorm_widths: (..., dim, n_bins)
        unnorm_heights: (..., dim, n_bins)
        unnorm_derivatives: (..., dim, n_bins - 1) or (..., dim, n_bins + 1)

    Returns:
        outputs: (..., dim)
        logabsdet: (..., dim)
    """

    # Check if all boundaries are defined
    if all(x is not None for x in [left, right, top, bottom]):
        if inverse:
            lower = bottom
            upper = top
        else:
            lower = left
            upper = right
    else:
        left = bottom = lower
        right = top = upper

    # Inside/outside spline window
    unnorm_widths = unnorm_widths.expand(*inputs.shape, -1)
    unnorm_heights = unnorm_heights.expand(*inputs.shape, -1)
    unnorm_derivatives = unnorm_derivatives.expand(*inputs.shape, -1)

    inside_interval = (inputs >= lower) & (inputs <= upper)
    outside_interval = ~inside_interval

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    # If edge derivatives not parametrized
    if unnorm_derivatives.shape[-1] == unnorm_widths.shape[-1] - 1:
        unnorm_derivatives = F.pad(unnorm_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnorm_derivatives[..., 0] = constant
        unnorm_derivatives[..., -1] = constant

    # Tails
    if tails == 'linear':
        outputs[outside_interval] = inputs[outside_interval]
        logabsdet[outside_interval] = 0
    else:
        raise RuntimeError(f'"{tails}" tails are not implemented.')

    # If nothing inside interval -> return unchanged
    if not inside_interval.any():
        return outputs, logabsdet

    # Go from unconstrained to actual values
    num_bins = unnorm_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = F.softmax(unnorm_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths

    heights = F.softmax(unnorm_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights

    derivatives = min_derivative + F.softplus(unnorm_derivatives)

    # Rational spline
    outputs, logabsdet = rational_quadratic_spline(
        inputs=inputs,
        widths=widths,
        heights=heights,
        derivatives=derivatives,
        inside_interval=inside_interval,
        initial_outputs=outputs,
        initial_logabsdet=logabsdet,
        inverse=inverse,
        left=left,
        right=right,
        bottom=bottom,
        top=top
    )

    return outputs, logabsdet


def rational_quadratic_spline(inputs,
                              widths,
                              heights,
                              derivatives,
                              inside_interval,
                              inverse=False,
                              initial_outputs=None,
                              initial_logabsdet=None,
                              left=None,
                              right=None,
                              bottom=None,
                              top=None):
    """
    Args:
        inputs: (..., dim)
        widths: (..., dim, n_bins)
        heights: (..., dim, n_bins)
        derivatives: (..., dim, n_bins + 1)
        inside_interval: Bool mask, (..., dim)
        inverse: Whether to do inverse calculation
        initial_outputs: Can be initialized to zero, or same as inputs, (..., dim)
        initial_logabsdet: Same as initial_outputs
        left: Left boundary, either (float) or (tensor) with same shape as inputs
        right: Right boundary, same type as left boundary
        bottom: Bottom boundary, same type as left boundary
        top: Top boundary, same type as left boundary
    Outputs:
        outputs: (..., dim)
        logabsdet: (..., dim)
    """

    # Take only inside interval
    inputs = inputs[inside_interval]
    widths = widths[inside_interval, :]
    heights = heights[inside_interval, :]
    derivatives = derivatives[inside_interval, :]

    # If boundaries not tensor, convert to one
    def boundary_to_tensor(b):
        return b[inside_interval].unsqueeze(-1) if torch.is_tensor(b) else torch.ones(inputs.shape[0], 1) * b
    left = boundary_to_tensor(left)
    right = boundary_to_tensor(right)
    top = boundary_to_tensor(top)
    bottom = boundary_to_tensor(bottom)

    if inverse and ((inputs < bottom).any() or (inputs > top).any()):
        raise ValueError('Inverse input is outside of domain')
    elif not inverse and ((inputs < left).any() or (inputs > right).any()):
        raise ValueError('Input is outside of domain')

    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0, None] = left
    cumwidths[..., -1, None] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0, None] = bottom
    cumheights[..., -1, None] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = st.util.searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = st.util.searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (((inputs - input_cumheights) * (input_derivatives
                                             + input_derivatives_plus_one
                                             - 2 * input_delta)
              + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives
             - (inputs - input_cumheights) * (input_derivatives
                                              + input_derivatives_plus_one
                                              - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * root.pow(2)
                                                     + 2 * input_delta * theta_one_minus_theta
                                                     + input_derivatives * (1 - root).pow(2))
        logabsdet = -torch.log(derivative_numerator) + 2 * torch.log(denominator) # Note the sign change
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2)
                                     + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2)
                                                     + 2 * input_delta * theta_one_minus_theta
                                                     + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    initial_outputs[inside_interval], initial_logabsdet[inside_interval] = outputs, logabsdet
    return initial_outputs, initial_logabsdet
