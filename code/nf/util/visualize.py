import nf
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('whitegrid', { 'axes.grid': False })

__all__ = ['plot_synthetic_datasets', 'plot_density']


def plot_synthetic_datasets():
    datasets = [
        'cos',
        'checkerboard',
        'circles',
        'line',
        'pinwheel',
        'rings',
        'swissroll',
        'moons',
        'spirals',
        'gaussians'
    ]

    N = len(datasets)
    _, axes = plt.subplots(figsize=(12, 5), nrows=2, ncols=5)
    axes = axes.reshape(-1)

    for i, name in enumerate(datasets):
        data = nf.load_dataset(name)
        data = data[np.random.randint(0, len(data), 1000)]
        axes[i].scatter(data[:,0], data[:,1], alpha=1, marker='.')
        axes[i].set_title(name.capitalize())

    plt.tight_layout()
    plt.show()

def plot_density(model, a=-4, b=4, num_samples=256, **kwargs):
    model.eval()

    X, Y = np.mgrid[a:b:200j, a:b:200j]
    xy = torch.Tensor(np.vstack((Y.flatten(), X.flatten())).T)

    log_prob = model.log_prob(xy, **kwargs)

    prob = log_prob.exp().view(*X.shape).detach().cpu().numpy()
    print(f'Area under curve {prob.sum() / 200**2 * (b - a)**2:.4f}')

    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    ax[0].set_aspect('equal')
    ax[0].imshow(prob, origin='lower', interpolation='none', extent=[a, b, a, b])
    ax[0].set_title('Learned density')
    ax[0].set_xlim([a, b])
    ax[0].set_ylim([a, b])

    samples = model.sample(num_samples, **kwargs).detach().cpu().numpy()
    ax[1].set_aspect('equal')
    ax[1].scatter(samples[:,0], samples[:,1], alpha=0.2, marker='.')
    ax[1].set_title('Samples')

    plt.tight_layout()
    plt.show()
    model.train()
