""" Utilities for State class"""

import logging

import numpy as np

from common.common_utils import CommonInfo, flatten


logger = logging.getLogger(__name__)


class Episode(CommonInfo):
    def reset(self, algo, train):
        self._loc = np.zeros(2, dtype=int)
        self._v = np.zeros(2, dtype=int)
        self.action = np.random.randint(5)
        self._kinetic = 0
        self._episode_reward = 0  # current episode reward
        self._discount = 1
        self.step = 0
        self.cur_reward = 0
        self._cur_drift = np.zeros(2, dtype=int)
        self.episode_end = False
        self.is_training = train
        if train:
            self._init_state_action(algo)

    def _init_state_action(self, algo):
        state, action = algo.initial_state_action()
        self._loc = state[:2]
        self._v = state[2:]
        self.action = action

    def update(self, algo):
        algo.update(self)
    
    def update_action(self, action):
        self.action = action

    def update_episode(self):
        # Update v and cost.
        self._v += self._control_unit[self.action]
        cost = max(sum(self._v**2) - self._kinetic, 0) + self._time_cost
        self._v[self._v > self._speed_limit] = self._speed_limit
        self._v[self._v < -self._speed_limit] = -self._speed_limit
        self._kinetic = sum(self._v**2)
        # Update loc.
        self._cur_drift = self.drift_effect
        self._loc += self._v + self._cur_drift
        self._loc %= self._size
        self.step += 1
        if tuple(self._loc) in self._endzone:
            self.episode_end = True
        # Update reward.
        self._discount *= self._discount_factor
        if self.step >= self._step_limit:
            self.cur_reward -= self._punitive_cost
            self.episode_end = True
            logger.warning('exceeds step limit')
        else:
            self.cur_reward = self.random_reward
        self._episode_reward += self.cur_reward * self._discount - cost

    @property
    def state(self):
        return tuple(flatten(self._loc, self._v))

    @property
    def state_action(self):
        return tuple(flatten(self._loc, self._v, self.action))

    @property
    def reward(self):
        return self._episode_reward

    @property
    def drift_effect(self):
        loc = tuple(self._loc)
        if loc not in self._drift_effect:
            return np.zeros(2, dtype=int)
        if not self._drift_effect[loc]:
            self._drift_effect[loc] = self._torus_map.drift_effect(
                loc, self._buffer)
        return self._drift_effect[loc].pop()

    @property
    def random_reward(self):
        loc = tuple(self._loc)
        if loc not in self._random_reward:
            return 0
        if not self._random_reward[loc]:
            self._random_reward[loc] = self._torus_map.random_reward(
                loc, self._buffer)
        return self._random_reward[loc].pop()
