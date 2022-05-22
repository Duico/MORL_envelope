from math import inf, pi
import gym
import numpy as np


class CartPoleV1AngleEnergyRewardWrapper_adapter(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        # required for compatibility with the Yang repo
        self.state_spec = [
            ["continuous", 1, [-4.8, 4.8]],
            ["continuous", 1, [-inf, inf]],
            ["continuous", 1, [-0.418, 0.418]],
            ["continuous", 1, [-inf, inf]],
        ]
        self.action_spec = ["discrete", 1, [0, 2]]

        # reward specification: 2-dimensional reward
        # 1st: [abs(angle), -energy]
        self.reward_spec = [[0, 2 * pi], [-10000, 0]]

        self.current_state = np.array([0, 0, 0, 0])
        self.terminal = False

    def reward(self, reward):
        return self.env.reward(reward)[0]

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward), done
