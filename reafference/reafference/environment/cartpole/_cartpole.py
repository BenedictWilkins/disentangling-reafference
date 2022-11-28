#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import numpy as np

__all__ = ("CartPoleRenderWrapper", "CartPoleNoopWrapper", "make")

def make(render=False, angle_threshold=90, euler=True, no_gravity=False, pole_length=0.5):
    env = gym.make("CartPole-v1")
    #env = InfoResetWrapper(env)
    if no_gravity:
        env = CartPoleNoGravity(env, angle_threshold=angle_threshold, euler=euler, pole_length=pole_length)
    else:
        env = CartPoleNoopWrapper(env, angle_threshold=angle_threshold, euler=euler, pole_length=pole_length)
    if render:
        env = CartPoleRenderWrapper(env)
    return env        

class CartPoleNoGravity(gym.Wrapper):

    def __init__(self, env, angle_threshold=90, euler=True, pole_length=0.5):
        super().__init__(env)
        self.env.unwrapped.tau = 0.02 # same as original? 
        self.env.unwrapped.theta_threshold_radians = angle_threshold * 2 * np.pi / 360
        self.env.unwrapped.kinematics_integrator = "euler" if euler else "non-euler"
        self._gravity = self.env.unwrapped.gravity 

        self.env.unwrapped.length = pole_length
        self.env.unwrapped.polemass_length = self.env.unwrapped.masspole * self.env.unwrapped.length


    def step(self, action):
        x, x_dot, theta, theta_dot = self.env.unwrapped.state
        # intervene on velocities to remove effect of previous actions
        self.env.unwrapped.gravity = 0
        self.env.unwrapped.state = [x, 0, theta, 0]
        iobs, *_ = super().step(action)

        self.env.unwrapped.gravity = self._gravity
        self.env.unwrapped.state = [x, x_dot, theta, theta_dot]
        obs, reward, done, info = super().step(action)
        info['is1'] = [x, 0, theta, 0]
        info['is2'] = iobs
        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        info = dict()
        info['is1'] = obs
        info['is2'] = obs
        return obs, info # conform to new gym API

class CartPoleRenderWrapper(gym.Wrapper):
    
    def __init__(self, env,  size=(100,150)):
        super().__init__(env)
        #assert env.unwrapped.screen is None
        #env.unwrapped.screen_width = size[0]    # TODO doesnt work?
        #env.unwrapped.screen_height = size[1]   # TODO doesnt work?
        #env.unwrapped.render_mode = "rgb_array" 

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info['img'] = self.get_screen()
        return obs, reward, done, info

    def reset(self):
        obs, info = super().reset()
        info['img'] = self.get_screen()
        return obs, info
    
    def get_screen(self):
        try: 
            screen = self.env.unwrapped.render(mode='rgb_array')
            screen = screen.transpose((2, 0, 1))
        except AttributeError as e:
            raise ImportError("Install pygame to render Cartpole.")
        return np.ascontiguousarray(screen, dtype=np.float32) / 255.
    
class CartPoleNoopWrapper(gym.Wrapper):
    
    def __init__(self, env, angle_threshold=90, euler=True, pole_length=0.5):
        super().__init__(env)
        self.env.unwrapped.tau = 0.02 # same as original? 
        self.env.unwrapped.theta_threshold_radians = angle_threshold * 2 * np.pi / 360
        self.env.unwrapped.kinematics_integrator = "euler" if euler else "non-euler"
        # 0 = Noop
        # 1 = Push left
        # 2 = Push right
        self.action_space = gym.spaces.Discrete(3) # [0,1,2]
        self._original_force_mag = env.unwrapped.force_mag

        self.env.unwrapped.length = pole_length
        self.env.unwrapped.polemass_length = self.env.unwrapped.masspole * self.env.unwrapped.length
        
    def step(self, action):
        noop = (action == 0)
        if noop:
            self.env.unwrapped.force_mag = 0 # perform a noop
        action = max(0, action-1)
        info = self.get_info(action)
        result = super().step(action)
        result[-1].update(info)
        if noop:
            self.env.unwrapped.force_mag = self._original_force_mag
        return result

    def get_info(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.env.unwrapped.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        acc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        return dict(acc=acc, thetaacc = thetaacc)

    def reset(self):
        obs = super().reset()
        return obs, dict(acc=0, thetaacc=0) # conform to new gym API