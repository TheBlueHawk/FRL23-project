import gym
from stable_baselines3 import PPO
from wrapper import make_env

obs_cost = 0.1
obs_flag = 1
vanilla = 0

env = make_env("CartPole-v1", obs_cost, obs_flag, vanilla)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

observation = env.reset()
for _ in range(100):
    # env.render()
    action, _ = model.predict(observation)
    observation, reward, done, info = env.step(action)
    print(done)
    if done:
        observation = env.reset()
        print("..........")

env.close()
