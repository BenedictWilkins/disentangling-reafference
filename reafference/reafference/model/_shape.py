#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import numpy as np

__all__ = ("as_shape", "View", "Flatten")

def as_shape(shape):
    if isinstance(shape, int):
        return (shape,)
    else:
        return tuple(shape)

class View(nn.Module):
    """ A Module that creates a view of an input tensor. """

    def __init__(self, input_shape, output_shape):
        super(View, self).__init__()
        def infer_shape(x, y): # x contains a -1
            assert not -1 in y
            t_y, t_x = np.prod(y), - np.prod(x)
            assert t_y % t_x == 0 # shapes are incompatible...
            x = list(x)
            x[x.index(-1)] = t_y // t_x
            return as_shape(x)

        self.input_shape = as_shape(input_shape)
        self.output_shape = as_shape(output_shape)

        # infer -1 in shape
        if -1 in self.output_shape:
            self.output_shape = infer_shape(self.output_shape, self.input_shape)
        if -1 in self.input_shape:
            self.input_shape = infer_shape(self.input_shape, self.output_shape)

    def forward(self, x):
        return x.view(x.shape[0], *self.output_shape)

    def __str__(self):
        attrbs = "{0}->{1}".format(self.input_shape, self.output_shape)
        return "{0}({1})".format(self.__class__.__name__, attrbs)

    def __repr__(self):
        return str(self)

    def shape(self, *args, **kwargs):
        return self.output_shape

    def inverse(self, **kwargs):
        return View(self.output_shape, self.input_shape)

class Flatten(View):
    """ A Module that creates a flat view of a tensor."""

    def __init__(self, input_shape):
        super().__init__(input_shape, (-1,))