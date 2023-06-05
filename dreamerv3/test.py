
import warnings
import dreamerv3
from dreamerv3 import embodied
import gym
import time
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')  # Ignoring certain types of warnings
from dreamerv3.embodied.core.basics import convert
import numpy as np

def _expand(self, value, dims):
    while len(value.shape) < dims:
      value = value[..., None]
    return value

# See configs.yaml for all options.
config = embodied.Config(dreamerv3.configs['defaults'])  # Loading default configuration
config = config.update(dreamerv3.configs['medium'])  # Updating with medium configuration
config = config.update({  # Further customizing configuration
    'logdir': f'logdir/{int(time.time())}',  # Directory for logs. Created fresh each time
    'run.train_ratio': 64,
    'run.log_every': 30,  # Logging frequency (in seconds)
    'batch_size': 16,  # Batch size for training
    'jax.prealloc': False,
    'encoder.mlp_keys': '.*',
    'decoder.mlp_keys': '.*',
    'encoder.cnn_keys': '$^',
    'decoder.cnn_keys': '$^',
    'jax.platform': 'cpu',
})
config = embodied.Flags(config).parse()  # Parsing final configuration

logdir = embodied.Path(config.logdir)  # Path to log directory
step = embodied.Counter()  # Step counter
# Logger for different output formats
logger = embodied.Logger(step, [
    embodied.logger.TerminalOutput(),  # Logging to terminal
    embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),  # Logging metrics in JSONL format
    embodied.logger.TensorBoardOutput(logdir),  # TensorBoard output
    # embodied.logger.WandBOutput(logdir.name, config),  # WandB output (commented out)
    # embodied.logger.MLFlowOutput(logdir.name),  # MLFlow output (commented out)
])

from dreamerv3.embodied.envs import from_gym    
env = gym.make('CartPole-v0')  # Creating Gym environment
env = from_gym.FromGym(env, obs_key='state_vec')  # Wrapping gym environment to use state vector as observation
env = dreamerv3.wrap_env(env, config)  # Wrapping the environment with DreamerV3 specifics
env = embodied.BatchEnv([env], parallel=False)  # Creating a batched version of the environment

agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)  # Creating DreamerV3 agent
replay = embodied.replay.Uniform(
    config.batch_length, config.replay_size, logdir / 'replay')  # Replay buffer for experience replay
args = embodied.Config(
    **config.run, logdir=config.logdir,
    batch_steps=config.batch_size * config.batch_length)  # Arguments for training
# embodied.run.train(agent, env, replay, logger, args)  # Starting the training process
# embodied.run.eval_only(agent, env, logger, args)  # Evaluation mode
# embodied.run.train_eval(agent, env, env, replay, replay, logger, args)  # Training and evaluation mode

print(agent)

policy = lambda *args: agent.policy(*args, mode='train') # mode = train so that we have access to action entropy
step = 0
episode = 0

mode = "train"
imagine = agent.agent.wm.imagine

# _acts = {
#         k: convert(np.zeros((len(env),) + v.shape, v.dtype))
#         for k, v in env.act_space.items()}
_acts = {'action': array([[0.5, 0.5]], dtype=float32), 'reset': array([False])}
print("_acts", _acts)
_state = None

assert all(len(x) == len(env) for x in _acts.values())

# Action Processing: Filter actions and perform an action step in the environment
# Prepare actions for the environment based on the current policy

# IF TRAIN THEN NO CHANGE:
if mode == "train" or mode == "expl" or step == 0:
    if step == 0:
        print("\tSTEP == 0")
    else:
        print("\tTRAIN OR EXPL")
    acts = {k: v for k, v in _acts.items() if not k.startswith('log_')}
    obs = env.step(acts)  # Interact with the environment using the prepared actions
    prev_state = obs
    obs = {k: convert(v) for k, v in obs.items()}  # Convert observation data to appropriate data types
    assert all(len(x) == len(env) for x in obs.values()), obs
    
    # Policy Execution: Get actions from the policy function based on the observations and state
    acts, _state = policy(obs, _state)  # Determine new actions based on the observed states
    acts = {k: convert(v) for k, v in acts.items()}  # Convert the action data to appropriate data types
    
    # Handling 'is_last' Observations: Adjust actions for terminated environments
    if obs['is_last'].any():
        mask = 1 - obs['is_last']
        acts = {k: v * _expand(mask, len(v.shape)) for k, v in acts.items()}  # Mask actions if an episode is done
    acts['reset'] = obs['is_last'].copy()  # Include a reset flag for the environment if an episode is done
    _acts = acts
    
    # Transition Processing: Merge observations and actions into a transition dictionary
    trns = {**obs, **acts}  # Combine observation and action data

# IF EVAL THEN SHOULD OBSERVE WORLD LESS:
if mode == "eval":
    print("\tEVAL")
    acts = {k: v for k, v in _acts.items() if not k.startswith('log_')}
    if step % 5 != 0:
        print("\t\tSTEP != 5*t")
        obs = imagine(policy, prev_state, 1)
    else:
        print("\t\tSTEP == 5*t")
        obs = env.step(acts)  # Interact with the environment using the prepared actions
    prev_state = obs
    obs = {k: convert(v) for k, v in obs.items()}
    assert all(len(x) == len(env) for x in obs.values()), obs
    
    # Policy Execution: Get actions from the policy function based on the observations and state
    acts, _state = policy(obs, _state)
    acts = {k: convert(v) for k, v in acts.items()}
    
    # Handling 'is_last' Observations: Adjust actions for terminated environments
    if obs['is_last'].any():
        mask = 1 - obs['is_last']
        acts = {k: v * _expand(mask, len(v.shape)) for k, v in acts.items()}
    acts['reset'] = obs['is_last'].copy()
    _acts = acts
    
    # Transition Processing: Merge observations and actions into a transition dictionary
    trns = {**obs, **acts} 



