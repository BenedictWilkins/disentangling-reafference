#!/usr/bin/env python
# -*- coding: utf-8 -*-


def register_entry_point(): # entry point hook for openai gym
    from gym.envs import register
    register(id="CartPole-v0", entry_point="reafference.environment.cartpole:make",  kwargs=dict(no_gravity=False))
    register(id="CartPole-v1", entry_point="reafference.environment.cartpole:make", kwargs=dict(no_gravity=True))
    register(id="Freeway-v0", entry_point="reafference.environment.freeway:make")
