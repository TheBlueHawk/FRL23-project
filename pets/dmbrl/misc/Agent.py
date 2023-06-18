from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from gym.monitoring import VideoRecorder
from dotmap import DotMap

import time


class Agent:
    """An general class for RL agents.
    """
    def __init__(self, params):
        """Initializes an agent.

        Arguments:
            params: (DotMap) A DotMap of agent parameters.
                .env: (OpenAI gym environment) The environment for this agent.
                .noisy_actions: (bool) Indicates whether random Gaussian noise will 
                    be added to the actions of this agent.
                .noise_stddev: (float) The standard deviation to be used for the 
                    action noise if params.noisy_actions is True.
        """
        self.env = params.env
        self.noise_stddev = params.noise_stddev if params.get("noisy_actions", False) else None

        if isinstance(self.env, DotMap):
            raise ValueError("Environment must be provided to the agent at initialization.")
        if (not isinstance(self.noise_stddev, float)) and params.get("noisy_actions", False):
            raise ValueError("Must provide standard deviation for noise for noisy actions.")

        if self.noise_stddev is not None:
            self.dU = self.env.action_space.shape[0]

    def sample(self, horizon, policy, record_fname=None, imagine=False):
        """Samples a rollout from the agent.

        Arguments: 
            horizon: (int) The length of the rollout to generate from the agent.
            policy: (policy) The policy that the agent will use for actions.
            record_fname: (str/None) The name of the file to which a recording of the rollout
                will be saved. If None, the rollout will not be recorded.

        Returns: (dict) A dictionary containing data from the rollout.
            The keys of the dictionary are 'obs', 'ac', and 'reward_sum'.
        """
        video_record = record_fname is not None
        recorder = None if not video_record else VideoRecorder(self.env, record_fname)

        times, rewards = [], []
        O, A, reward_sum, done = [self.env.reset()], [], 0, False

        policy.reset()
        t = 0
        while t < horizon and not done:
            if video_record:
                recorder.capture_frame()
            start = time.time()
            acs, pred_trajs, pred_trajs_var, full_acs = policy.act(O[t], t)
            times.append(time.time() - start)

            # if pred_trajs is not False:
            #     print("pred_trajs ", pred_trajs.shape)
            #     woah = np.squeeze(pred_trajs, axis=1) # (26, 20, 4)
            #     jeez = np.var(woah, axis=1) # (26, 4)
            #     print("variance ", jeez)
            
            if imagine:
                i=0
                pred_trajs = np.squeeze(pred_trajs, axis=1) # (26, 20, 4)
                variance = np.var(pred_trajs, axis=1) # (26, 4)
                weights = np.array([0.4487, 0.6450, 0.1020, 0.0319])
                avg_variance = np.average(variance, axis=1, weights=weights) # (26,)
                mean = np.mean(pred_trajs, axis=1) # (26, 4)
                full_acs = full_acs.reshape(-1, 1)

                mask = avg_variance > 0.0005 # 0.01 0.1 -> nul
                index = np.argmax(mask)
                # print(avg_variance)
                # print("index ", index)
                # index = 3 # 3, 5 -> nul

                if index == 0:
                    index = avg_variance.shape[0] - 1

                while i < index and not done:
                    A.append(full_acs[i])
                
                    if self.noise_stddev is None:
                        obs, reward, done, info = self.env.step(full_acs[i])
                    else:
                        action = full_acs[i] + np.random.normal(loc=0, scale=self.noise_stddev, size=[self.dU])
                        action = np.minimum(np.maximum(action, self.env.action_space.low), self.env.action_space.high)
                        obs, reward, done, info = self.env.step(action)
                    O.append(obs)
                    if i == 0:
                        reward = reward - 0.4
                    reward_sum += reward
                    rewards.append(reward)
                    i+=1
                    t+=1
            else:
                A.append(acs)
            
                if self.noise_stddev is None and not done:
                    obs, reward, done, info = self.env.step(A[t])
                else:
                    action = A[t] + np.random.normal(loc=0, scale=self.noise_stddev, size=[self.dU])
                    action = np.minimum(np.maximum(action, self.env.action_space.low), self.env.action_space.high)
                    obs, reward, done, info = self.env.step(action)
                O.append(obs)
                reward = reward - 0.4
                reward_sum += reward
                rewards.append(reward)
                
                t+=1

        if video_record:
            recorder.capture_frame()
            recorder.close()

        print("Average action selection time: ", np.mean(times))
        print("Rollout length: ", len(A))

        return {
            "obs": np.array(O),
            "ac": np.array(A),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
        }
