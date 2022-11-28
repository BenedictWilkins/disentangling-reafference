#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import seaborn as sb
import gym
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, ConcatDataset
from ...data.iterators import gym_iterator

__all__ = ("environment_step", "ground_truth", "get_image", "get_images", "plot", "make_episode", "make_dataset")

def environment_step(env, x1, a):
    if len(x1.shape) == 1:
        x1 = x1.view(1, -1)
    if isinstance(a, (int, float)):
        a = np.array([a])
    if len(a.shape) == 1:
        a = a.view(1, -1)

    assert x1.shape[0] == a.shape[0]
    assert len(a.shape) == 2
    assert len(x1.shape) == 2

    def _step(x1, a):
        if torch.is_tensor(x1):
            x1 = x1.cpu().numpy()
        if torch.is_tensor(a):
            a = a.cpu().numpy()
        if a.shape[-1] > 1: # probably onehot
            a = a.argmax(-1)
        env.reset()
        x2 = np.empty_like(x1)
        for i in range(x1.shape[0]):
            env.unwrapped.state = x1[i]
            x2[i], *_ = env.step(a[i])
        return x2

    x2 = _step(x1, a)
    if torch.is_tensor(x1):
        x2 = torch.from_numpy(x2).to(x1.device)
    return x1, x2, a

def ground_truth(env, state):
    """ Gets ground truth effects (reafference and exafference) given `states` that have been generated from the given `env`. 

    Args:
        env (gym.Env): environment used to gather states should be a CartPole environment.
        state (numpy.ndarray): states 

    Returns:
        tuple(numpy.ndarray): ground truth (total_effect, reafferent_effect, exafferent_effect)
    """
    if env.action_space.n == 3: 
        return ground_truth_noop(env, state)
    raise ValueError(f"Invalid environment {env}, requires 3 actions (noop, left, right).")


def ground_truth_noop(env, state):
    assert hasattr(env.unwrapped, "state") # invalid environment, expected a CartPole environment
    def get_noop_states(state):
        def _get():
            for x1 in state[:-1]:
                env.unwrapped.state = x1
                x2, *_ = env.step(0) # take null action
                yield x2
        state_noop = np.stack([x for x in _get()])
        return state_noop
    s = state
    sn = get_noop_states(s)
    gt_effect = s[1:] - s[:-1]      # T(X_t, A_t) - X_t
    gt_re = (s[1:] - sn)            # T(X_t, A_t) - T(X_t, 0)
    gt_ex = gt_effect - gt_re       
    return gt_effect, gt_re, gt_ex


def make_dataset(env, num_episodes=1000, max_episode_length=100, device="cpu"):
    onehot = lambda x: torch.nn.functional.one_hot(x, env.action_space.n).float()
    datasets = []
    for i in range(num_episodes):
        state, action, _ = make_episode(env, max_length=max_episode_length)
        state = torch.from_numpy(state).to(device)
        action = onehot(torch.from_numpy(action)).to(device)
        datasets.append(TensorDataset(state[:-1], state[1:], action[:-1]))
    return ConcatDataset(datasets)
        
    #dataset = gymu.data.dataset(lambda : env, None, mode=gymu.mode.sad, max_episode_length=max_episode_length, num_episodes=num_episodes)
    
    #dataset = dataset.to_dict().window(window_size=2)
    #dataset = dataset.to_numpy().to_tensor()
    #dataset = dataset.map_dict(action=onehot)
    #dataset = dataset.to_tuple("state", "action")
    #dataset = dataset.to_tensor_dataset()
    #dataset.tensors = [x.to(device) for x in dataset.tensors]
    #return dataset

def make_episode(env, policy=None, max_length=1000):
    iterator = gym_iterator(env, policy=policy, max_length=max_length)
    state, action, reward, done, info = zip(*iterator)
    state, action, = np.stack(state), np.stack(action)
    _info = {}
    for k in info[0].keys():
        _info[k] = np.stack([x[k] for x in info])
    return state, action, _info

def get_image(env, state, action):
    env.unwrapped.state = state
    nstate, reward, done, info = env.step(action)
    return nstate, info['img']

def get_images(env, state, action):
    if torch.is_tensor(state):
        state = state.detach().cpu().numpy()
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()

    if len(state.shape) == 3:
        x1, x2 = state[:,0], state[:,1]
    elif len(state.shape) == 2:
        x1, x2 = state, None
    else:
        x1, x2 = state.view(1, -1), None

    if len(action.shape) == 3:
        action = action[:,0]
    if len(action.shape) == 2:
        action = action.argmax(-1)
    a = action

    env.reset()
    test_data = [get_image(env, _x1, _a) for _x1,_a in zip(x1, a)]
    _x2, imgs = zip(*test_data)
    if x2 is not None:
        # check nothing funnny has happened (some precision is lost)
        np.testing.assert_almost_equal(x2, np.stack(_x2), decimal=3)
    return np.stack(imgs)

def plot(state, style="-", label_prefix="", action=None, ax=None, fig=None, title=None):
    assert len(state.shape) == 2 # [batch_size, state_size]
    assert state.shape[1] == 4 # state size
    if torch.is_tensor(state):
        state = state.detach().cpu().numpy()
    
    if fig is None:
        fig = plt.figure(figsize=(12,4))
        ax = plt.gca()
    if title is not None:
        ax.set_title(title)
    X = np.arange(state.shape[0])
    ax1 = sb.lineplot(x=X, y=state[:,0], label=f"{label_prefix}Position",          linestyle=style, ax=ax)
    ax2 = sb.lineplot(x=X, y=state[:,1], label=f"{label_prefix}Velocity",          linestyle=style, ax=ax)
    ax3 = sb.lineplot(x=X, y=state[:,2], label=f"{label_prefix}Angle",             linestyle=style, ax=ax)
    ax4 = sb.lineplot(x=X, y=state[:,3], label=f"{label_prefix}Angular Velocity",   linestyle=style, ax=ax)
    axes = [ax1, ax2, ax3, ax4]
    if action is not None:
        # TODO validate action... should be of shape [batch_size]
        sb.scatterplot(x=X, y = action, label="Action", ax=ax)
    return axes

def state_labels():
    return ["Position", "Velocity", "Angle", "AngularVelocity"]

