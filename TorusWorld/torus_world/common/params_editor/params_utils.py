"""Utilities of the Params class"""
import logging

from common.common_utils import CommonUtils

logger = logging.getLogger(__name__)


class Params(CommonUtils):

    @property
    def time_cost(self):
        return self._time_cost

    @time_cost.setter
    def time_cost(self, time_cost):
        assert time_cost > 0, 'time_cost is not positive'
        self._time_cost = time_cost

    @property
    def speed_limit(self):
        return self._speed_limit

    @speed_limit.setter
    def speed_limit(self, speed):
        assert isinstance(speed, int) and speed > 0
        self._speed_limit = speed

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

    @property
    def punitive_cost(self):
        return self._punitive_cost

    @punitive_cost.setter
    def punitive_cost(self, cost):
        assert cost > 0
        self._punitive_cost = cost

    def save(self, path, overwrite):
        super().save(path, overwrite)
        logger.info(f"Parameters {self._params_id} Saved!")


def generate_params(path, **kwargs):
    params = Params()
    params.__dict__.update(kwargs)
    logger.info(f"successfully generate parameters {params.params_id}")
    params.save(path, overwrite=False)
