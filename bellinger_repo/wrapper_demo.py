from stable_baselines3 import PPO
from wrapper import make_env

obs_cost = 0
obs_flag = 0
vanilla = 1

env = make_env("CartPole-v1", obs_cost, obs_flag, vanilla)
print(type(env))

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)

observation = env.reset()

for _ in range(10000):
  env.render()
  action, _states = model.predict(observation)
  observation, reward, terminated, info = env.step(action)
 
  if terminated:
    observation = env.reset()
env.close()