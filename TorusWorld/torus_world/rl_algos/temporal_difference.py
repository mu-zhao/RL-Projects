"""Temporal difference algorithms. """
import numpy as np

from common.algo_utils import Algo
from common.common_utils import ParameterValues, flatten


class TemporalDifference(Algo):
    """
    """
    def __init__(self, *args, **kwargs):
        """ base TD class
            hyper: random_init, learning_rate
        """
        super().__init__(*args, **kwargs)
        self.state_action_shape = list(self._size)+[
            2 * self._speed_limit + 1] * 2 + [5]
        self.random_initial = self.hyper_parameters.random_init
        self.algo_parameters.state_action = ParameterValues(
            self.state_action_shape, self.random_initial)
        self._alpha = self.hyper_parameters.learning_rate
        self._exploration = self.hyper_parameters.exploration

    def initial_state_action(self):  # non random start
        state_and_action = (super().initial_state(False),
                            super().initial_action(False))
        self._cur_state_action = flatten(state_and_action)
        return state_and_action

    def control(self, state):
        if np.random.uniform() < self._exploration:
            return np.random.randint(5)
        return super().control(state)


class Sarsa(TemporalDifference):
    """ Sarsa algo
        hyper : random_init, learning_rate, exploration

    """
    def update(self, episode_info):
        episode_info.update_episode()
        action = self.control(episode_info.state)
        next_state_action = episode_info.state_action
        # Update prediction
        target = episode_info.cur_reward + self._discount_factor * (
            self.algo_parameters.state_action.reward(next_state_action))
        self.algo_parameters.state_action.update_prediction(
            self._cur_state_action, target, self._alpha)
        episode_info.update_action(action)
        # Update current state_action.
        self._cur_state_action = next_state_action


class QLearning(TemporalDifference):
    def update(self, episode_info):
        episode_info.update_episode()
        # Update prediction
        target = episode_info.cur_reward + self._discount_factor * (
            self.algo_parameters.state_action.max_return(episode_info.state))
        self.algo_parameters.state_action.update_prediction(
            self._cur_state_action, target, self._alpha)
        action = self.control(episode_info.state)
        episode_info.update_action(action)
        # Update current state_action.
        self._cur_state_action = episode_info.state_action


class DoubleLearning(TemporalDifference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.algo_parameters.double_state_actions = [
            ParameterValues(self.state_action_shape, self.random_initial),
            ParameterValues(self.state_action_shape, self.random_initial)]
        # TODO: use better encapsulation.
        self.algo_parameters.state_action.parameter[:] = (
            self.algo_parameters.double_state_actions[0].parameter +
            self.algo_parameters.double_state_actions[1].parameter)

    def update(self, episode_info):
        episode_info.update_episode()
        # Randomly select one state_action table to update
        first_table = np.random.uniform < 0.5
        cur_state = self._cur_state_action[:2]
        # Select best action.
        best_action = self.algo_parameters.double_state_actions[
                first_table].decision(cur_state)
        state_action4other_table = flatten(cur_state, best_action)
        # Select target value from the other table
        target = episode_info.cur_reward + self._discount_factor * (
            self.algo_parameters.double_state_actions[
                not first_table].reward(state_action4other_table))
        # Update table
        self.algo_parameters.double_state_actions[
            first_table].update_prediction(
                self._cur_state_action, target, self._alpha)
        # Update state_action value.
        new_prediction = 0
        for i in range(2):
            new_prediction += (
                self.algo_parameters.double_state_actions[i].reward(
                    self._cur_state_action))
        self.algo_parameters.state_action.update_value(
            self._cur_state_action, new_prediction)
        action = self.control(episode_info.state)
        episode_info.update_action(action)
        # Update current state_action.
        self._cur_state_action = episode_info.state_action


class QSigma(TemporalDifference):
    pass
