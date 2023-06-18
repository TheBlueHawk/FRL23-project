import gym
from stable_baselines3 import PPO, DDPG
from wrapper import make_env
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import torch
import env

obs_cost = 0.4
obs_flag = 1
vanilla = 0
total_steps = 200000
log_dir = "logdir"

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

#env = make_env("InvertedPendulum-v2", obs_cost, obs_flag, vanilla)
env = make_env("MBRLCartpole-v0", obs_cost, obs_flag, vanilla)

class PrintEpisodeRewardCallback(BaseCallback):
    """
    Callback for printing the reward per training episode and saving the data to a pandas DataFrame.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose: int = 1):
        super(PrintEpisodeRewardCallback, self).__init__(verbose)
        self.episode_data = {'Episode': [], 'Reward': [], 'Length': [], 'TotalSteps': []}

    def _on_step(self) -> bool:
        # print(f"Step: {self.num_timesteps}")
        if len(self.model.ep_info_buffer) > 0:
            rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
            lengths = [ep_info['l'] for ep_info in self.model.ep_info_buffer]
            total_steps = self.num_timesteps
            for episode, reward in enumerate(rewards):
                episode_num = len(self.episode_data['Episode'])
                self.episode_data['Episode'].append(episode_num)
                self.episode_data['Reward'].append(reward)
                self.episode_data['Length'].append(lengths[episode])
                self.episode_data['TotalSteps'].append(total_steps)
                if self.verbose > 0:
                    print(f"Episode: {episode_num}, Reward: {reward}, Length: {lengths[episode]}, Total Steps: {total_steps}")
            self.model.ep_info_buffer.clear()  # Clear the episode information buffer after printing
        return True

model = PPO("MlpPolicy", env, device=device, verbose=1)  # Set the device parameter

callback = PrintEpisodeRewardCallback(verbose=0)  # Create the callback instance
model.learn(total_timesteps=total_steps, callback=callback)

env.close()

# Convert episode data to pandas DataFrame
episode_df = pd.DataFrame(callback.episode_data)

import time
import os
from datetime import datetime
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
file_name = f"{env.spec.id}_{total_steps}__{timestamp}.csv"
file_path = os.path.join(log_dir, file_name)
episode_df.to_csv(file_path, index=False)

print(episode_df.describe())
