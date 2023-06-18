import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from cartpole import CartpoleEnv
from wrapper import MeasureWrapper

obs_cost = 0.4
obs_flag = 1
vanilla = 0
total_steps = 200000
log_dir = "logdir"

# env = make_env("Pendulum-v1", obs_cost, obs_flag, vanilla)
cartpole_env = CartpoleEnv()
env = MeasureWrapper(cartpole_env, obs_cost=obs_cost, unknown_state='LAST_MEASURED', obs_flag=obs_flag)

# Set the seed for reproducibility
env.seed(0)

# Define the number of actions and states
nb_actions = 2
nb_states = env.observation_space.shape[0]

# Define the neural network model
model = Sequential()
model.add(Dense(16, input_shape=(nb_states,), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

# Define the Q-learning agent
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=10000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)

# Compile the agent
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Train the agent
dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)

# Test the agent
dqn.test(env, nb_episodes=5, visualize=True)
