#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import io
import gym
import sys

from typing import Callable, Dict, Any, List, Iterable, Union

__all__ = ("decode", "keep", "to_tuple", "gym_iterator")



_RESET_COMPAT_ERROR_MSG = "You are using the old gym API `state = env.reset()` please use the new one `state, info = env.reset()` or wrap your environment with a `gymu.wrappers.InfoResetWrapper` to ensure compatability."

class gym_iterator(Iterable):

    def __init__(self, env : Union[str, Callable, gym.Env, gym.Wrapper], 
                    policy : Callable = None,  
                    max_length : int = sys.maxsize):
      
        if isinstance(env, str):
            env = gym.make(env)
        elif not isinstance(env, gym.Env) and callable(env):
            env = env()
        self.max_length = max_length
        self.env = env
        if policy is None:
            policy = lambda *args: env.action_space.sample()
        self.policy = policy

    def __iter__(self):
        stateinfo = self.env.reset()
        try: # validate API...
            if len(stateinfo) != 2: 
                raise ValueError(_RESET_COMPAT_ERROR_MSG)
        except TypeError: 
            raise ValueError(_RESET_COMPAT_ERROR_MSG)
        state, info = stateinfo
        done = False
        i = 0
        while not done:
            action = self.policy(state)
            # state, action, reward, next_state, done, info
            next_state, reward, done, next_info = self.env.step(action)
            i += 1
            done = done or i >= self.max_length
            if done:
                assert not 'terminal_state' in info
                assert not 'terminal_info' in info
                info['terminal_state'] = next_state
                info['terminal_info'] = next_info
            yield (state, action, reward, done, info) # S_t, A_t, R_{t+1}, done_{t+1}, info_t
            state = next_state
            info = next_info


def decode(source, keep_meta=False):
    """Decode .tar dataset.

    Args:
        source (iterable): dataset iterable (see webdataset)
        keep_meta (bool, optional): whether to keep dataset meta data. Defaults to False.
    """
    def _decode_keep_meta(source):
        for data in source:
            x = {k:v for k,v in  np.load(io.BytesIO(data['npz']), allow_pickle=True).items()}
            del data['npz']
            x.update(data)
            yield x
    def _decode(source):
        for data in source:
            yield {k:v for k,v in  np.load(io.BytesIO(data['npz']), allow_pickle=True).items()}
    if keep_meta:
        yield from _decode_keep_meta(source)
    else:
        yield from _decode(source)

def keep(source : Iterable, keys : List[Any]):
    """ Keep the specified keys, discard the rest.
    Args:
        source (Iterable): source iterable.
        keys (List[Any], optional): keys to keep.
    Yields:
        dict: dictionary containing only the specified keys and their associated values.
    """
    for x in source:
        yield {k:x[k] for k in keys} # TODO missing error handling?

def to_tuple(source, *keys):
    """Convert a dict iterable to a tuple iterable, values appear in the order of 'keys'

    Args:
        source (Iterabe): iterable of dicts

    Yields:
        tuple: iterable of tuples
    """
    if len(keys) == 0:
        for data in source:
            yield tuple(data.values())
    else:
        for data in source:
            #print([x.shape for x in data.values()])
            yield tuple([data[k] for k in keys])
