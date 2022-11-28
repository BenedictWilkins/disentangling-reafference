#!/usr/bin/env python
# -*- coding: utf-8 -*-


import webdataset
import pathlib
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, ConcatDataset

from ...data import iterators

__all__ = ("load", "make_dataset")

def load(*paths):
    paths = [pathlib.Path(p).expanduser().resolve() for p in paths]
    for p in paths:
        assert p.exists()
        assert p.suffix in [".tar", ".tar.gz"]
    paths = [str(p) for p in paths]
    dataset = webdataset.WebDataset(paths).compose(iterators.decode)
    dataset = dataset.compose(lambda source : iterators.keep(source, ["state", "action", "rotation"]))
    dataset = dataset.compose(iterators.to_tuple)
    return dataset

def make_dataset(*paths):
    def episodes():
        for path in tqdm(paths):
            episode = load(path)
            x, a, _ = zip(*episode)
            x, a = np.stack(x), np.stack(a)
            x, a = torch.from_numpy(x), torch.from_numpy(a)
            x1, x2, a = x[:-1], x[1:], a[:-1]
            a = torch.eye(3)[a.long()] # one hot, remember [0,1,0] is null-action
            yield TensorDataset(x1, x2, a)
    return ConcatDataset(list(episodes()))