import collections

import numpy as np

from .basics import convert


class Driver:

  _CONVERSION = {
      np.floating: np.float32,
      np.signedinteger: np.int32,
      np.uint8: np.uint8,
      bool: bool,
  }

  def __init__(self, env, **kwargs):
    assert len(env) > 0
    self._env = env
    self._kwargs = kwargs
    self._on_steps = []
    self._on_episodes = []
    self.reset()

  def reset(self):
    self._acts = {
        k: convert(np.zeros((len(self._env),) + v.shape, v.dtype))
        for k, v in self._env.act_space.items()}
    self._acts['reset'] = np.ones(len(self._env), bool)
    self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
    self._state = None

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode)

  def _step(self, policy, step, episode):
    # Assertion: Check that the lengths of all actions are consistent with the number of environments
    assert all(len(x) == len(self._env) for x in self._acts.values())
    
    # Action Processing: Filter actions and perform an action step in the environment
    acts = {k: v for k, v in self._acts.items() if not k.startswith('log_')}
    obs = self._env.step(acts)
    obs = {k: convert(v) for k, v in obs.items()}
    assert all(len(x) == len(self._env) for x in obs.values()), obs
    
    # Policy Execution: Get actions from the policy function based on the observations and state
    acts, self._state = policy(obs, self._state, **self._kwargs)
    acts = {k: convert(v) for k, v in acts.items()}
    
    # Handling 'is_last' Observations: Adjust actions for terminated environments
    if obs['is_last'].any():
        mask = 1 - obs['is_last']
        acts = {k: v * self._expand(mask, len(v.shape)) for k, v in acts.items()}
    acts['reset'] = obs['is_last'].copy()
    self._acts = acts
    
    # Transition Processing: Merge observations and actions into a transition dictionary
    trns = {**obs, **acts}
    
    # Handling 'is_first' Observations: Clear episode dictionaries for new episodes
    if obs['is_first'].any():
        for i, first in enumerate(obs['is_first']):
            if first:
                self._eps[i].clear()
    
    # Environment-wise Processing: Process transitions for each environment
    for i in range(len(self._env)):
        trn = {k: v[i] for k, v in trns.items()}
        [self._eps[i][k].append(v) for k, v in trn.items()]
        [fn(trn, i, **self._kwargs) for fn in self._on_steps]
        step += 1
    
    # Handling 'is_last' Observations (Episode Completion): Call episode callbacks for completed episodes
    if obs['is_last'].any():
        for i, done in enumerate(obs['is_last']):
            if done:
                ep = {k: convert(v) for k, v in self._eps[i].items()}
                [fn(ep.copy(), i, **self._kwargs) for fn in self._on_episodes]
                episode += 1
    
    # Return updated step and episode counters
    return step, episode


  def _expand(self, value, dims):
    while len(value.shape) < dims:
      value = value[..., None]
    return value
