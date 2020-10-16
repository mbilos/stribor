# https://github.com/bayesiains/nsf/blob/master/nde/transforms/splines/rational_quadratic.py

import nf
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
                                            tails='linear',
                                            min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                                            min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                                            min_derivative=DEFAULT_MIN_DERIVATIVE):

    unnorm_widths = unnorm_widths.expand(*inputs.shape, -1)
    unnorm_heights = unnorm_heights.expand(*inputs.shape, -1)
    unnorm_derivatives = unnorm_derivatives.expand(*inputs.shape, -1)

    inside_interval = (inputs >= lower) & (inputs <= upper)
    outside_interval = ~inside_interval

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == 'linear':
        unnorm_derivatives = F.pad(unnorm_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnorm_derivatives[..., 0] = constant
        unnorm_derivatives[..., -1] = constant

        outputs[outside_interval] = inputs[outside_interval]
        logabsdet[outside_interval] = 0
    else:
        raise RuntimeError('{} tails are not implemented.'.format(tails))

    if not inside_interval.any():
        return outputs, logabsdet

    outputs[inside_interval], logabsdet[inside_interval] = rational_quadratic_spline(
        inputs=inputs[inside_interval],
        unnorm_widths=unnorm_widths[inside_interval, :],
        unnorm_heights=unnorm_heights[inside_interval, :],
        unnorm_derivatives=unnorm_derivatives[inside_interval, :],
        inverse=inverse,
        left=lower, right=upper, bottom=lower, top=upper,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )

    return outputs, logabsdet

def rational_quadratic_spline(inputs,
                              unnorm_widths,
                              unnorm_heights,
                              unnorm_derivatives,
                              inverse=False,
                              left=None, right=None, bottom=None, top=None,
                              min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                              min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                              min_derivative=DEFAULT_MIN_DERIVATIVE):

    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError('Output is outside of domain')

    num_bins = unnorm_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = F.softmax(unnorm_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnorm_derivatives)

    heights = F.softmax(unnorm_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = nf.util.searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = nf.util.searchsorted(cumwidths, inputs)[..., None]

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
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
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

        return outputs, logabsdet
