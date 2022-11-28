#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils.data import TensorDataset, ConcatDataset

from ...data.iterators import gym_iterator

__all__ = ("make_dataset", "make_episode", "ground_truth")

def ground_truth(env, states, ram_states):
    def _get():
        for ram_state in ram_states[:-1]:
            env.ale.restoreState(ram_state)
            yield env.step(0)[0]
    s = states
    sn = np.stack([s for s in _get()])
    print(s.shape, sn.shape)
    gt_effect = s[1:] - s[:-1]      # T(X_t, A_t) - X_t
    gt_re = (s[1:] - sn)            # T(X_t, A_t) - T(X_t, 0)
    gt_ex = gt_effect - gt_re       
    return gt_effect, gt_re, gt_ex

def make_episode(env, policy=None, max_length=1000):
    iterator = gym_iterator(env, policy=policy, max_length=max_length)
    state, action, reward, done, info = zip(*iterator)
    state, action, = np.stack(state), np.stack(action)
    _info = {}
    for k in info[0].keys():
        _info[k] = np.stack([x[k] for x in info])
    return state, action, _info

def make_dataset(env, num_episodes=1000, max_episode_length=100, device="cpu"):
    onehot = lambda x: torch.nn.functional.one_hot(x, env.action_space.n).float()
    datasets = []
    for i in range(num_episodes):
        state, action, _ = make_episode(env, max_length=max_episode_length)
        state = torch.from_numpy(state).to(device)
        action = onehot(torch.from_numpy(action)).to(device)
        datasets.append(TensorDataset(state[:-1], state[1:], action[:-1]))
    return ConcatDataset(datasets)
        
        