import matplotlib
from gym import spaces
from gym.utils import seeding

matplotlib.use("Agg")
import gym
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D

SMALL_SIZE = 7
MEDIUM_SIZE = 9
BIGGER_SIZE = 11

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def fig2im(fig):
    width, height = fig.get_size_inches() * fig.get_dpi()
    canvas = FigureCanvas(fig)
    canvas.draw()
    return np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(
        int(height), int(width), 3
    )


class CustomHikeEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array", "3dfig"]}

    def __init__(
        self,
        obs_cost=-0.5,  # for runs on 0403 turned off obs cost, -0.2,
        goal_reward=None,
        transition_std=0.025,
        observation_std=0.0,
        goal_end=False,
        goal_position=[0.7, 0.5],
        goal_radius=0.1,
        outside_box_cost=-1.5,
        starting_position=[-0.85, -0.85],
        starting_std=0.1,
        action_cost_factor=0.1,
        max_action_value=0.05,
        max_time=100,
        box_scale=10,
        shaping_power=4,
        hill_height=4,
    ):
        self.obs_cost = obs_cost
        self.box_scale = box_scale
        # self.box_mean = (self.box_max - self.box_min)/2.
        # self.box_size = self.box_max - self.box_min
        self.max_time = max_time
        self.action_cost_factor = action_cost_factor
        self.max_action_value = max_action_value
        self.shaping_power = shaping_power
        self.hill_height = hill_height

        # self.obs_pos = []
        # self.no_pos = []
        # self.steps = 0
        self.action_space = spaces.Box(
            low=-max_action_value, high=max_action_value, shape=(3,)
        )
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(2,))
        self.viewer = None

        self.transition_std = transition_std
        self.observation_std = observation_std
        self.total_reward = 0
        self.goal_reward = goal_reward
        self.goal_position = np.array(goal_position)
        self.goal_radius = goal_radius
        self.goal_end = goal_end

        self.outside_box_cost = outside_box_cost

        self.seed()
        self.starting_position = np.array(starting_position)
        self.starting_std = starting_std

    def reached_goal(self):
        return np.linalg.norm(self.position - self.goal_position) < self.goal_radius

    def outside_box(self):
        x, y = self.position
        if (x < -1) or (x > 1) or (y < -1) or (y > 1):
            return True
        else:
            return False

    def done(self):
        return (self.goal_end and self.reached_goal()) or self.time <= 0

    def resize(self, pos):
        return pos * self.box_scale

    def observation(self):
        rand_obs = self.np_random.multivariate_normal(
            self.position, self.observation_std**2 * np.eye(2)
        )
        self.last_observation = rand_obs
        resized_obs = self.resize(rand_obs)
        return resized_obs

    def get_reward(self, X, Y):

        Z = np.zeros(X.shape)
        means = [np.array([0.5, 0.5]), np.array([0, 0.2]), np.array([-0.375, -0.5])]
        covs = [
            np.array([[0.75, 0], [0, 0.1]]),
            np.array([[0.1, 0], [0.0, 0.75]]),
            np.array([[0.75, 0], [0, 0.1]]),
        ]

        factors = [1, 0.8, 0.55]

        num_mixtures = len(means)
        for mean, cov, factor in zip(means, covs, factors):
            # Z += factor * mlab.bivariate_normal(X, Y, cov[0, 0], cov[1, 1], mean[0], mean[1], cov[0, 1]) / num_mixtures
            Z = np.maximum(
                Z,
                factor
                * mlab.bivariate_normal(
                    X, Y, cov[0, 0], cov[1, 1], mean[0], mean[1], cov[0, 1]
                )
                / num_mixtures,
            )

        # Z = Z - (X + Y) * (1-Z)
        # Z = (X + Y + 2) / 4 * Z
        Z = 1 - np.power(1 - Z, self.shaping_power)
        Z *= self.hill_height
        Z -= 0.5 + self.hill_height
        Z += (X + Y) / 4

        return Z

    def _step(self, _action):
        # observed is either [0 = not obs, 1 = obs]
        action, observed = _action[:2], _action[-1]
        self.time = self.time - 1
        if self._done or self.time < 0:
            return None

        action_penalty = self.action_cost_factor * np.linalg.norm(action)
        if np.linalg.norm(action) > self.max_action_value:
            action = action / np.linalg.norm(action) * self.max_action_value
        # action = np.clip(action, -self.max_action_value, self.max_action_value)

        self.position = self.np_random.multivariate_normal(
            self.position + action, self.transition_std**2 * np.eye(2)
        )
        # self.steps += 1
        self._done = self.done()
        if self.reached_goal() and self.goal_reward is not None:
            reward = self.goal_reward
        elif self.outside_box():
            reward = self.outside_box_cost * self.hill_height
        else:
            reward = self.get_reward(*self.position)
        observation = self.observation()
        total_reward = reward - action_penalty

        if observed < 1:  # if missing obs
            observation = np.zeros((self.observation_space.shape), dtype=float)
            # self.no_pos.append(self.position)
        else:
            # self.obs_pos.append(self.position)
            total_reward += self.obs_cost
        self.total_reward += total_reward
        info = {
            "observed": observed,
            "true state": self.position.copy(),
            "total_reward": self.total_reward,
            "reward": reward,
            "action_penalty": action_penalty,
            "true_position": self.resize(self.position),
        }
        # if self.steps == 75:
        #    fig = self._render(mode='2dfig')
        #    fig.savefig('obs_dvrl.png')
        return observation, total_reward, self._done, info

    def _reset(self):
        self.total_reward = 0.0
        self.position = self.np_random.multivariate_normal(
            self.starting_position, self.starting_std**2 * np.eye(2)
        )
        self.time = self.max_time
        self._done = False
        return self.observation()

    def _get_XYR(self):
        n = 256
        X, Y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
        R = self.get_reward(X, Y)
        X = self.resize(X)
        Y = self.resize(Y)
        return X, Y, R

    def _get_contour(self, alpha=0.8, figsize=(2.1, 2), cmap=cm.gnuplot):
        X, Y, R = self._get_XYR()
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        ax.contourf(X, Y, R, cmap=cm.hot)
        contour = ax.contourf(
            X,
            Y,
            R,
            cmap=cmap,
            alpha=alpha,
            levels=np.linspace(-5, 0, 51),
            antialiased=True,
        )
        return fig, ax, contour

    def _render(self, mode="human", close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if mode == "rgb_array":
            # fig, ax = plt.subplots(1, 1)
            # fig.set_size_inches(3, 3)
            # fig.set_dpi(200)
            # ax.scatter(self.position[0], self.position[1], color='white', label='true position')
            # im = ax.imshow(R, interpolation='bilinear', extent=(self.box_min, self.box_max, self.box_min, self.box_max), origin='lower')
            # fig.colorbar(im, shrink=0.7)
            # ax.legend(bbox_to_anchor=(1, 1), loc='upper right')
            # fig.tight_layout()
            # plt.close(fig)
            return fig2im(fig)
        elif mode == "2dfig":
            fig, ax, contour = self._get_contour()
            x = np.array(self.obs_pos)[:, 0] * 10.0
            y = np.array(self.obs_pos)[:, 1] * 10.0
            ax.scatter(x, y, marker="^", s=5, c="mediumblue")
            x = np.array(self.no_pos)[:, 0] * 10.0
            y = np.array(self.no_pos)[:, 1] * 10.0
            ax.scatter(x, y, marker="o", s=5, c="black")
            fig.colorbar(contour, shrink=0.7)
            return fig
        elif mode == "3dfig":
            X, Y, R = self._get_XYR()
            fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={"projection": "3d"})
            ax.plot_surface(X, Y, R, cmap=cm.hot, linewidth=0, antialiased=False)
            return fig
        elif mode == "human":
            from gym.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(fig2im(fig))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
