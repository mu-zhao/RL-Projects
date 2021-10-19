""" Utilities for State class"""

import logging

import numpy as np

from common.common_utils import CommonInfo, flatten


logger = logging.getLogger(__name__)


class Episode(CommonInfo):
    def reset(self, algo, train):
        self._loc = np.zeros(2, dtype=int)
        self._v = np.zeros(2, dtype=int)
        self._action = np.random.randint(5)
        self._kinetic = 0
        self._reward = 0  # current episode reward
        self._discount = 1
        self._step = 0
        self._cur_reward = 0
        self._cur_drift = np.zeros(2, dtype=int)
        self.episode_end = False
        if train:
            self._init_state_action(algo)

    def _init_state_action(self, algo):
        state, action = algo.initial_state_action()
        self._loc = state[:2]
        self._v = state[2:]
        self._action = action

    def update(self, algo, train=True):
        cur_state_action = self.state_action
        cost = self._update_v_and_cost()
        self._update_loc()
        self._update_reward(cost)
        if train:
            algo.update(cur_state_action, self._cur_reward, self._cur_drift,
                        self._step, self.episode_end)
        self._action = algo.control(self.state)

    def _update_loc(self):
        self._cur_drift = self.drift_effect
        self._loc += self._v + self._cur_drift
        self._loc %= self._size
        self._step += 1
        if tuple(self._loc) in self._endzone:
            self.episode_end = True

    def _update_v_and_cost(self):
        self._v += self._control_unit[self._action]
        cost = max(sum(self._v**2) - self._kinetic, 0) + self._time_cost
        self._v[self._v > self._speed_limit] = self._speed_limit
        self._v[self._v < -self._speed_limit] = -self._speed_limit
        self._kinetic = sum(self._v**2)
        return cost

    def _update_reward(self, cost):
        self._discount *= self._discount_factor
        if self._step >= self._step_limit:
            self._cur_reward -= self._punitive_cost
            self.episode_end = True
            logger.warning('exceeds step limit')
        else:
            self._cur_reward = self.random_reward
        self._reward += self._cur_reward * self._discount - cost

    @property
    def state(self):
        return flatten(self._loc, self._v)

    @property
    def state_action(self):
        return flatten(self._loc, self._v, [self._action])

    @property
    def reward(self):
        return self._reward

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
