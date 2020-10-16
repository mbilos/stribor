# Adapted from https://github.com/rtqichen/ffjord/blob/master/lib/toy_data.py

import numpy as np
import argparse
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle
import pathlib


dataset_dir = pathlib.Path(__file__).parents[3] / 'data'
dataset_dir.mkdir(parents=True, exist_ok=True)


def swissroll(N=10000, seed=1):
    """ Generates N elements of swissroll dataset with set seed. """
    np.random.seed(seed)
    data = sklearn.datasets.make_swiss_roll(n_samples=N, noise=1.0)[0]
    data = data.astype('float32')[:, [0, 2]]
    data /= 5
    return data

def circles(N=10000, seed=1):
    """ Generates N elements of circles dataset with set seed. """
    np.random.seed(seed)
    data = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=0.08)[0]
    data = data.astype('float32')
    data *= 3
    return data

def rings(N=10000, seed=1):
    """ Generates N elements of rings dataset with set seed. """
    np.random.seed(seed)
    n_samples4 = n_samples3 = n_samples2 = N // 4
    n_samples1 = N - n_samples4 - n_samples3 - n_samples2

    # so as not to have the first point = last point, we set endpoint=False
    linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
    linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
    linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
    linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

    circ4_x = np.cos(linspace4)
    circ4_y = np.sin(linspace4)
    circ3_x = np.cos(linspace4) * 0.75
    circ3_y = np.sin(linspace3) * 0.75
    circ2_x = np.cos(linspace2) * 0.5
    circ2_y = np.sin(linspace2) * 0.5
    circ1_x = np.cos(linspace1) * 0.25
    circ1_y = np.sin(linspace1) * 0.25

    X = np.vstack([
        np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
        np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
    ]).T * 3.0
    X = util_shuffle(X)

    # Add noise
    X = X + np.random.normal(scale=0.08, size=X.shape)

    return X.astype('float32')

def moons(N=10000, seed=1):
    """ Generates N elements of moons dataset with set seed. """
    np.random.seed(seed)
    data = sklearn.datasets.make_moons(n_samples=N, noise=0.1)[0]
    data = data.astype('float32')
    data = data * 2 + np.array([-1, -0.2])
    return data

def gaussians(N=10000, seed=1):
    """ Generates N elements of gaussians dataset with set seed. """
    np.random.seed(seed)
    scale = 4.
    centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                        1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
    centers = [(scale * x, scale * y) for x, y in centers]

    dataset = []
    for i in range(N):
        point = np.random.randn(2) * 0.5
        idx = np.random.randint(8)
        center = centers[idx]
        point[0] += center[0]
        point[1] += center[1]
        dataset.append(point)
    dataset = np.array(dataset, dtype='float32')
    dataset /= 1.414
    return dataset

def pinwheel(N=10000, seed=1):
    """ Generates N elements of pinwheel dataset with set seed. """
    np.random.seed(seed)
    radial_std = 0.3
    tangential_std = 0.1
    num_classes = 5
    num_per_class = N // 5
    rate = 0.25
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

    features = np.random.randn(num_classes*num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:, 0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    return 2 * np.random.permutation(np.einsum('ti,tij->tj', features, rotations))

def spirals(N=10000, seed=1):
    """ Generates N elements of spirals dataset with set seed. """
    np.random.seed(seed)
    n = np.sqrt(np.random.rand(N // 2, 1)) * 540 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(N // 2, 1) * 0.5
    d1y = np.sin(n) * n + np.random.rand(N // 2, 1) * 0.5
    x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
    x += np.random.randn(*x.shape) * 0.1
    return x

def checkerboard(N=10000, seed=1):
    """ Generates N elements of checkerboard dataset with set seed. """
    np.random.seed(seed)
    x1 = np.random.rand(N) * 4 - 2
    x2_ = np.random.rand(N) - np.random.randint(0, 2, N) * 2
    x2 = x2_ + (np.floor(x1) % 2)
    return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

def line(N=10000, seed=1):
    """ Generates N elements of line dataset with set seed. """
    np.random.seed(seed)
    x = np.random.rand(N) * 5 - 2.5
    y = x
    return np.stack((x, y), 1)

def cos(N=10000, seed=1):
    """ Generates N elements of cos dataset with set seed. """
    np.random.seed(seed)
    x = np.random.rand(N) * 5 - 2.5
    y = np.sin(x) * 2.5
    return np.stack((x, y), 1)



parser = argparse.ArgumentParser('Toy data generation')
parser.add_argument('--N', type=int, default=10000, help='Number of elements to generate')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
args = parser.parse_args()

if __name__ == '__main__':
    print(f'Generating datasets with {args.N} elements and random seed {args.seed}')

    funcs = [swissroll, circles, rings, moons, gaussians, pinwheel, spirals, checkerboard, line, cos]

    for func in funcs:
        data = func(N=args.N, seed=args.seed)
        np.savetxt(dataset_dir / f'{func.__name__}.txt', data)
