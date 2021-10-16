""" This is utilities for algorithms"""


import numpy as np

from common.common_utils import CommonUtils, CommonInfo, gen_path


def generate_policy(self, state_action_value, epsilon):
    policy = np.zeros_like(state_action_value) + epsilon
    x, y, vx, vy = np.shape(state_action_value)[:4]
    for i in range(x):
        for j in range(y):
            for k in range(vx):
                for m in range(vy):
                    action_values = state_action_value[i, j, k, m]
                    policy[i, j, k, m, np.argmax(
                        action_values)] = 1 - 4 * epsilon
    return policy


class ParameterValues(CommonUtils):
    def __init__(self, params_shape, random_init=True):
        self._params_shape = params_shape
        if random_init:
            self.parameter = np.random.uniform(size=params_shape)
        else:
            self.parameter = np.zeros(params_shape)

    def decision(self, state):
        params_values = self.parameter[state]
        arg_max_index = np.flatnonzero(params_values == params_values.max())
        # Break tie randomly
        return np.random.choice(arg_max_index)

    def update_prediction(self, state_action, value):
        self.parameter[state_action] = value

    def change_policy(self, behavior_policy):
        self.parameter = behavior_policy

    def gen_policy(self, epsilon):
        return generate_policy(self.parameter, epsilon)

    def save(self, path, overwrite):
        super().save(path, overwrite)


class HyperParameter(CommonUtils):
    def save(self, dir, overwrite):
        hyper_params_path = gen_path('hyper_parameter', dir)
        super().save(hyper_params_path, overwrite)

    def load(self, dir):
        hyper_params_path = gen_path('hyper_parameter', dir)
        super().load(hyper_params_path)


class AlgoParameter(CommonUtils):
    pass


class Algo(CommonInfo):
    def __init__(self, params, torus_map):
        super().__init__(params, torus_map)
        self.algo_parameters = AlgoParameter()
        self.hyper_parameters = HyperParameter()

    def save(self, dir, overwrite=True):
        self.hyper_parameters.save(dir, overwrite)
        self.algo_parameters.save(dir, overwrite)

    def load(self, dir):
        self.hyper_parameters.load(dir)
        self.algo_parameters.load(dir)

    def initial_state_action(self):
        pass

    def initial_state(self, random=True):
        if not random:
            return np.zeros(4)
        x_size, y_size = self._size
        x_loc = np.random.randint(x_size)
        y_loc = np.random.randint(y_size)
        v = np.random.randint(self._speed_limit, size=2)
        return np.array([x_loc, y_loc] + list(v))

    def initial_action(self, state, random_start=True):
        if not random_start:
            return self.algo_parameters.state_action.decision(state)
        return np.random.randint(5)

    def control(self, state):
        self.algo_parameters.state_action.decision(state)

    def update(self, state_action, reward, cur_drift, step, end):
        pass
