from common.algo_utils import Algo
from common.common_utils import ParameterValues

class TemporalDifference(Algo):
    def __init__(self, *args, **kwargs):
        """ base TD class
            hyper: random_init, learning_rate
        """
        super().__init__(*args, **kwargs)
        self.state_action_shape = list(self._size)+[
            2 * self._speed_limit + 1] * 2 + [5]
        random_initial= self.hyper_parameters.random_init
        self.algo_parameters.state_action = ParameterValues(
            self.state_action_shape, random_initial)
        self._alpha = self.hyper_parameters.learning_rate

    def initial_state_action(self):  # non random start
        return super().initial_state(False), super().initial_action(False)


class Sarsa(TemporalDifference):
    """ Sarsa algo
        hyper : random_init, learning_rate, exploration

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._exploration = self.hyper_parameters.exploration
        self.pre
    def update(self, state_action, reward, cur_drift, step, end):
        self.algo_parameters.state_action += self._alpha * (
            reward + 
        )
    def 

class QLearning(TemporalDifference):
    pass 

class ExSarsa(Sarsa):
    pass 

class DoubleLearning(TemporalDifference):
    pass 

