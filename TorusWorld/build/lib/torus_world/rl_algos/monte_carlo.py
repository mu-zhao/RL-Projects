
import numpy as np

from TorusWorld.torus_world.common.algo_editor.algo_utils import(
    Algo, AlgoParameter)


class MonteCarlo(Algo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_action_shape = list(self._size)+[
            self._speed_limit] * 2 + [5]
        random_start = self.hyper_parameters.get('random_init', False)
        self.algo_parameters.state_action = AlgoParameter(
            self.state_action_shape, random_start)
        return_shape = self.state_action_shape + [2]
        self._return = np.zeros(return_shape)
        self._action = None
        self._discounting = self._discount_factor ** np.arange(
            self._step_limit)
        self._episode_record = np.zeros(self._step_limit)
        self._first_visit = {}

    def update(self, state_action, reward, cur_drift, step, end):
        if not end:
            if state_action not in self._first_visit:
                self._first_visit[step] = state_action
            self._episode_record[step] = reward
        else:
            cum_discounted_reward = np.cumsum(
                self._discounting[:step] *
                self._episode_record[:step][::-1])[::-1]
            for step in self._first_visit:
                state_action = self._first_visit[step]
                self._return[state_action][0] += 1
                self._return[state_action][1] += cum_discounted_reward[step]
                mean_return = self._return[state_action][1] / self._return[
                    state_action][0]
                self.algo_parameters.state_action.update_prediction(
                    state_action, mean_return)


class MCES(MonteCarlo):
    """ Monte Carlo with Exploring Starts
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._exploration = self.hyper_parameters['exploring start']

    def initial_state_action(self):
        is_random = np.random.unifrom() < self._exploration
        return self.initial_state(is_random), self.initial_action(is_random)


class OnPolicyMC(MonteCarlo):
    """On policy first visit Monte Carlo
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._exploration = self.hyper_parameters['exploring']

    def control(self, state):
        if np.random.uniform() < self._exploration:
            return np.random.randint(5)
        return super().control(state)
