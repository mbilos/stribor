import os
import numpy as np
import torch
import torch.utils.data as data_utils
from pathlib import Path

dataset_dir = Path(__file__).parents[2] / 'data'

__all__ = ['list_datasets', 'load_dataset']


def list_datasets():
    check = lambda x: x.is_file() and x.suffix == '.txt'
    file_list = [x.stem for x in (dataset_dir).iterdir() if check(x)]
    return file_list

def load_dataset(name):
    if not name.endswith('.txt'):
        name += '.txt'
    return np.loadtxt(dataset_dir / name)
