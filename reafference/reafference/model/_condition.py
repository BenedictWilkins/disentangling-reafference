#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from ._shape import as_shape, Flatten 

__all__ = ("DiagLinear",)

class DiagLinear(nn.Module):

    def __init__(self, input_shape, action_shape, latent_shape=(1024,)):
        super().__init__()
        self.input_shape = as_shape(input_shape)
        self.action_shape = as_shape(action_shape)
        self.latent_shape = as_shape(latent_shape)
        assert len(self.action_shape) == 1

        if len(input_shape) > 1:
            self.flatten_in = Flatten(input_shape)
            self.action_in = nn.Linear(self.action_shape[0], self.latent_shape[0])
            self.state_out = nn.Linear(self.latent_shape[0], self.flatten_in.output_shape[0])
            self.state_in = nn.Linear(self.flatten_in.output_shape[0], self.latent_shape[0])
            self.flatten_out = self.flatten_in.inverse()
            self.forward = self._forward_flatten
        else:
            self.action_in = nn.Linear(self.action_shape[0], self.latent_shape[0])
            self.state_out = nn.Linear(self.latent_shape[0], self.input_shape[0])
            self.state_in = nn.Linear(self.input_shape[0], self.latent_shape[0])
            self.forward = self._forward    

    def _forward(self, x, a):
        x = self.state_in(x) * self.action_in(a)
        return self.state_out(x)

    def _forward_flatten(self, x, a):
        x = self.flatten_in(x)
        y = self._forward(x, a)
        y = self.flatten_out(y)
        return y