""" This is utilities for algorithms"""

import numpy as np

from common.common_utils import CommonUtils, CommonInfo, gen_path


class HyperParameter(CommonUtils):
    def save(self, dir, overwrite):
        hyper_params_path = gen_path('hyper_parameter', dir, 0)
        super().save(hyper_params_path, overwrite)

    def load(self, dir):
        hyper_params_path = gen_path('hyper_parameter', dir, 0)
        super().load(hyper_params_path)


class AlgoParameter(CommonUtils):
    pass


class Algo(CommonInfo):
    """ Algo base class
        hyper trained_episode
    """
    def __init__(self, params, torus_map, hyper_parameter):
        super().__init__(params, torus_map)
        self.algo_parameters = AlgoParameter()
        self.hyper_parameters = hyper_parameter

    def save(self, dir, overwrite=True):
        self.hyper_parameters.save(dir, overwrite)
        self.algo_parameters.save(dir, overwrite)

    def load(self, dir):
        self.algo_parameters.load(dir)

    def initial_state_action(self):
        raise NotImplementedError('Must be implemented by subclasses')

    def initial_state(self, random=True):
        if not random:
            return np.zeros(4, dtype=int)
        x_size, y_size = self._size
        x_loc = np.random.randint(x_size)
        y_loc = np.random.randint(y_size)
        v = np.random.randint(self._speed_limit, size=2)
        return np.array([x_loc, y_loc] + list(v), dtype=int)

    def initial_action(self, state, random_start=True):
        if not random_start:
            return self.algo_parameters.state_action.decision(state)
        return np.random.randint(5)

    def control(self, state):
        return self.algo_parameters.state_action.decision(state)

    def update(self, episode_info):
        raise NotImplementedError('Must be implemented by subclasses')
