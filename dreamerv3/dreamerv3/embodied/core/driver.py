import collections
import numpy as np
from .basics import convert

# class responsible for interacting with the environment using a given policy
class Driver:
  # Dictionary mapping data types to their corresponding conversion functions
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
    self._on_steps = []       # List to store step callback functions
    self._on_episodes = []    # List to store episode callback functions
    self.reset()              # Call the reset method to initialize the driver

  def reset(self):
    # Initialize the action dictionary with zero-filled arrays and a reset flag
    self._acts = {
        k: convert(np.zeros((len(self._env),) + v.shape, v.dtype))
        for k, v in self._env.act_space.items()}
    self._acts['reset'] = np.ones(len(self._env), bool)
    
    # Create a list of defaultdicts to store episode data for each environment instance
    self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
    self._state = None        # Initialize the internal state to None

  def on_step(self, callback):
    self._on_steps.append(callback)  # Add a callback function to the list of step callbacks

  def on_episode(self, callback):
    self._on_episodes.append(callback)  # Add a callback function to the list of episode callbacks

  def __call__(self, policy, steps=0, episodes=0, total_step=0):
    step, episode = 0, 0
    # Continue the interaction loop until reaching the desired number of steps or episodes
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode, total_step)

  # performs a single step in the interaction loop. It receives observations from the environment, 
  # determines actions using the policy, updates internal state, and stores transition data
  def _step(self, policy, step, episode, total_step):
    # Assertion: Check that the lengths of all actions are consistent with the number of environments
    assert all(len(x) == len(self._env) for x in self._acts.values())

    # Action Processing: Filter actions and perform an action step in the environment
    # Prepare actions for the environment based on the current policy
    acts = {k: v for k, v in self._acts.items() if not k.startswith('log_')}
    obs = self._env.step(acts)  # Interact with the environment using the prepared actions
    obs = {k: convert(v) for k, v in obs.items()}  # Convert observation data to appropriate data types

    # RANDOM OBSERVATION USED FOR EXPERIMENT ONLY

    ACTIVATED = True
    THRESHOLD_STEPS = 150000
    NOISY = 0.1
    # NOISY = min((total_step - THRESHOLD_STEPS) * 0.001, 1)

    if ACTIVATED:
        def generate_random_dictionary():
            random_dict = {
                # 'state_vec': np.array([[np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()]]), # CARTPOLE
                # 'state_vec': np.array([[np.random.rand()*2 -1, np.random.rand()*2-1, np.random.rand()*16-8]]), # PENDULUM
                # 'state_vec': np.array([[np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1),
                #                 np.random.uniform(-1, 1), np.random.uniform(-4*np.pi, 4*np.pi),
                #                 np.random.uniform(-9*np.pi, 9*np.pi)]]), # ACROBOT
                'state_vec': np.array([[np.random.rand()*1.8 -1.2, np.random.rand()*0.14-0.07]]), # MOUNTAIN CAR CONT
                'reward': np.array([np.random.choice([0, 1])], dtype=np.float32),
                'is_first': np.array([np.random.choice([True, False])]),
                'is_last': np.array([np.random.choice([True, False])]),
                'is_terminal': np.array([np.random.choice([True, False])])
            }
            return random_dict

        random_obs = generate_random_dictionary()
        random_obs["reward"] = obs["reward"]
        random_obs["is_first"] = obs["is_first"]
        random_obs["is_last"] = obs["is_last"]

        if total_step > THRESHOLD_STEPS:
            if NOISY == 1:
                obs = random_obs
            elif NOISY == 0:
                pass
            else:
                obs["state_vec"] = (1 - NOISY) * obs["state_vec"] + NOISY * random_obs["state_vec"]

    assert all(len(x) == len(self._env) for x in obs.values()), obs
    
    # Policy Execution: Get actions from the policy function based on the observations and state
    acts, self._state = policy(obs, self._state, **self._kwargs)  # Determine new actions based on the observed states
    acts = {k: convert(v) for k, v in acts.items()}  # Convert the action data to appropriate data types
    
    # Handling 'is_last' Observations: Adjust actions for terminated environments
    if obs['is_last'].any():
        mask = 1 - obs['is_last']
        acts = {k: v * self._expand(mask, len(v.shape)) for k, v in acts.items()}  # Mask actions if an episode is done
    acts['reset'] = obs['is_last'].copy()  # Include a reset flag for the environment if an episode is done
    self._acts = acts
    
    # Transition Processing: Merge observations and actions into a transition dictionary
    trns = {**obs, **acts}  # Combine observation and action data
    
    # Handling 'is_first' Observations: Clear episode dictionaries for new episodes
    if obs['is_first'].any():
        for i, first in enumerate(obs['is_first']):
            if first:
                self._eps[i].clear()  # Clear episode data if a new episode has started
    
    # Environment-wise Processing: Process transitions for each environment
    for i in range(len(self._env)):
        trn = {k: v[i] for k, v in trns.items()}  # Retrieve data specific to each environment instance
        [self._eps[i][k].append(v) for k, v in trn.items()]  # Store transition data for each environment instance
        [fn(trn, i, **self._kwargs) for fn in self._on_steps]  # Call step callbacks for each environment instance
        step += 1
    
    # Handling 'is_last' Observations (Episode Completion): Call episode callbacks for completed episodes
    if obs['is_last'].any():
        for i, done in enumerate(obs['is_last']):
            if done:
                ep = {k: convert(v) for k, v in self._eps[i].items()}  # Convert episode data to appropriate data types
                [fn(ep.copy(), i, **self._kwargs) for fn in self._on_episodes]  # Call episode callbacks for each environment instance
                episode += 1
    
    # Return updated step and episode counters
    return step, episode


  def _expand(self, value, dims):
    while len(value.shape) < dims:
      value = value[..., None]
    return value
