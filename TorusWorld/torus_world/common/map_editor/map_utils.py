"""Utilities of the map class"""
import logging

import numpy as np

from common.common_utils import CommonUtils, change_keys

logger = logging.getLogger(__name__)


def _filter(size, end_loc):
    endzone = []
    for pos in end_loc:
        if len(pos) != 2:
            logger.warning(f"loc {pos} not 2d coordinates")
        if not(isinstance(pos[0], int) and isinstance(pos[1], int)) or \
                pos[0] >= size[0] or pos[1] >= size[1]:
            logger.warning(f"loc {pos} not in the world")
        else:
            endzone.append(tuple(pos))
    assert endzone, "valid endzone is empty!"
    return endzone


class TorusMap(CommonUtils):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._convert(False)

    def generate_drift(self, drift_config):
        drift_limit = drift_config[0]
        self._drift_unit = list(range(-drift_limit, drift_limit+1))
        drift_prob = drift_config[1]
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if np.random.uniform() < drift_prob:
                    local_drift = np.random.uniform(size=2 * drift_limit + 1)
                    local_drift_dist = local_drift / sum(local_drift)
                    self._drift[(i, j)] = list(local_drift_dist)

    def generate_reward(self, reward_config):
        mu, var_limit, reward_prob = reward_config
        assert var_limit > 0, "variance is not positive."
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if np.random.uniform() < reward_prob:
                    self._reward[(i, j)] = [np.random.normal(mu, 1),
                                            np.random.uniform(0, var_limit)]

    @property
    def size(self):
        return self._size

    @property
    def map_id(self):
        return self._map_id

    @property
    def endzone(self):
        return self._endzone

    @endzone.setter
    def endzone(self, new_endzone):
        self._endzone = _filter(self._size, new_endzone)

    def drift_effect(self, loc, size):
        assert tuple(loc) in self._drift
        return list(np.random.choice(
            self._drift_unit, size=size, p=self._drift[tuple(loc)]))

    def random_reward(self, loc, size):
        assert tuple(loc) in self._reward
        mu, sig = self._reward[tuple(loc)]
        return list(np.random.normal(mu, sig, size=size))

    def get_endzone(self):
        return {tuple(loc) for loc in self._endzone}

    def save(self, path, overwrite):
        self._convert()
        super().save(path, overwrite)
        logger.info(f"Flat Torus {self._map_id} Saved!")

    def _convert(self, to_int=True):
        if hasattr(self, '_reward'):
            self._reward = change_keys(self._reward, self._size, to_int)
        if hasattr(self, '_drift'):
            self._drift = change_keys(self._drift, self._size, to_int)

    @property
    def driftzone(self):
        return set(self._drift.keys())

    @property
    def rewardzone(self):
        return set(self._reward.keys())


def random_generate_map(map_id, size, endzone, reward_config, drift_config,
                        path):
    map = TorusMap()
    map._map_id = map_id
    map._size = size
    map._endzone = _filter(size, endzone)
    map._reward = {}
    map._drift = {}
    map.generate_drift(drift_config)
    map.generate_reward(reward_config)
    logger.warning('successfully generate map')

    map.save(path, overwrite=False)
