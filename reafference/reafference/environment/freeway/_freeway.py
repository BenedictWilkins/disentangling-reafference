import gym
import numpy as np

__all__ = ("make", )

def make(noop_range=(1,40)):
    assert noop_range[0] <= noop_range[1]
    assert noop_range[0] >= 1
    env = gym.make("FreewayDeterministic-v4")
    env = FreewayWrapper(env)
    return env

class FreewayWrapper(gym.Wrapper):

    def __init__(self, env, noop_range=(1,40)):
        """ Please only use this with the FreewayDeterministic-v4 environment... """
        env = NoopResetWrapper(env, 0, range=noop_range) # take a random number of noops at the start of each episode
        super().__init__(env)
        assert env.observation_space.shape == (210, 160, 3)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,84,84)) 

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info['ram'] = self.env.ale.cloneState()
        return self._state_transform(obs), reward, done, info

    def reset(self):
        obs, info = super().reset()
        info['ram'] = self.env.ale.cloneState()
        return self._state_transform(obs), info # comply with new gym API

    def _state_transform(self, x):
        x = x[111:195,15:99,:] 
        x = x.transpose((2,0,1)) # final shape (2,3,84,84)
        x = (x.astype(np.float32) / 255.)
        # binarise, TODO create wrapper instead...?
        # x[x < threshold] = 0.
        # x[x > threshold] = 1.
        # i = (x.prod(1, keepdim=True) != 1).repeat(1,3,1,1)
        # x[i] = 0        
        return x

class NoopResetWrapper(gym.Wrapper):

    def __init__(self, env, noop, range=(0,100), sampler=None):
        super().__init__(env)
        self.noop = noop
        self.range = range
        assert range[0] >= 0 and range[1] >= range[0]
        if range[1] != range[0]:
            self.sampler = sampler if sampler is not None else NoopResetWrapper.UniformSampler
        else:
            self.sampler = lambda x: x[0]
            
    def reset(self):
        obs = super().reset()
        n = self.sampler(self.range)
        for _ in range(n):
            obs, reward, done, info = self.step(self.noop) # take n noop steps
            assert not done # failed to do a noop start... this should not happen.
        return obs, dict()

    @staticmethod
    def UniformSampler(range):
        return np.random.randint(range[0], range[1])