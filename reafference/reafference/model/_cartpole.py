#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

__all__ = ("CartPoleNet", )

class CartPoleNet(torch.nn.Module):
    
    def __init__(self, input_shape=(4,), action_shape=(3,), hidden_shape=(256,)):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(input_shape[0] + action_shape[0], hidden_shape[0]), 
            nn.Tanh(),
            nn.Linear(hidden_shape[0], hidden_shape[0]),
            nn.Tanh(),
            nn.Linear(hidden_shape[0], hidden_shape[0]),
            nn.Tanh(),
            nn.Linear(hidden_shape[0], input_shape[0]))
            
    def forward(self, x, a):
        z = torch.cat([x,a], dim=-1)
        return self.l1(z)