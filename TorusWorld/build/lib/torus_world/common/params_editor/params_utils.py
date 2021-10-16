"""Utilities of the Params class"""
import logging

from common.common_utils import CommonUtils

logger = logging.getLogger(__name__)


class Params(CommonUtils):
    def __init__(self, **kwargs):
        if 'path' in kwargs:
            super().__init__(kwargs['path'])
        else:
            self._params_id = kwargs['params_id']
            self._episodes = kwargs["episodes"]
            self.discount_factor = kwargs["discount_factor"]
            self._step_limit = kwargs["step_limit"]
            self.time_cost = kwargs["time_cost"]
            self._punitive_cost = kwargs["punitive_cost"]

    @property
    def time_cost(self):
        return self._time_cost

    @time_cost.setter
    def time_cost(self, time_cost):
        assert time_cost > 0, 'time_cost is not positive'
        self._time_cost = time_cost

    @property
    def step_limit(self):
        return self._step_limit

    @property
    def params_id(self):
        return self._params_id

    @property
    def episodes(self):
        return self._episodes

    @episodes.setter
    def episodes(self, num):
        assert self._episodes <= num
        self._episodes = num

    @property
    def discount_factor(self):
        return self._discount_factor

    @discount_factor.setter
    def discount_factor(self, discount_factor):
        assert 0 <= discount_factor <= 1, 'discount factor not between 0 and 1'
        self._discount_factor = discount_factor

    def save(self, path, overwrite):
        super().save(path, overwrite)
        logger.info(f"Parameters {self._params_id} Saved!")


def generate_params(params_id, episodes, discount_factor, step_limit,
                    time_cost, punitive_cost, path):
    params = Params(params_id=params_id, episodes=episodes,
                    discount_factor=discount_factor, step_limit=step_limit,
                    time_cost=time_cost, punitive_cost=punitive_cost)
    logger.info(f"successfully generate parameters {params_id}")
    params.save(path, overwrite=False)
