from copy import deepcopy
from collections import defaultdict
import numpy as np
from common.common_utils import ParameterValues

from common.algo_utils import Algo


class MonteCarlo(Algo):
    def __init__(self, *args, **kwargs):
        """ base algo class
            hyper: random_init, trained_episodes
        """
        super().__init__(*args, **kwargs)
        self.state_action_shape = list(self._size)+[
            2 * self._speed_limit + 1] * 2 + [5]
        random_start = self.hyper_parameters.random_init
        self.algo_parameters.state_action = ParameterValues(
            self.state_action_shape, random_start)

    def initial_state_action(self):  # non random start
        return super().initial_state(False), super().initial_action(False)



class FirstVisitMC(MonteCarlo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return_shape = self.state_action_shape + [2]
        self._return = np.zeros(return_shape)
        self._action = None
        self._discounting = self._discount_factor ** np.arange(
            self._step_limit)
        self._episode_record = np.zeros(self._step_limit)
        self._first_visit = {}

    def update(self, state_action, reward, cur_drift, step, end):
        self._episode_record[step] = reward
        if state_action not in self._first_visit:
            self._first_visit[step] = state_action
        if end:
            cum_discounted_reward = np.cumsum(
                self._discounting[:step] *
                self._episode_record[:step][::-1])[::-1]
            for visit in self._first_visit:
                state_action = self._first_visit[visit]
                self._return[state_action][0] += 1
                self._return[state_action][1] += cum_discounted_reward[visit]
                mean_return = self._return[state_action][1] / self._return[
                    state_action][0]
                self.algo_parameters.state_action.update_prediction(
                    state_action, mean_return)
            self._first_visit = {}


class MCES(FirstVisitMC):
    """ Monte Carlo with Exploring Starts
    Hyper: random_init, exploring_start
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._exploration = self.hyper_parameters.exploring_start

    def initial_state_action(self):
        is_random = np.random.unifrom() < self._exploration
        return self.initial_state(is_random), self.initial_action(is_random)


class OnPolicyMC(FirstVisitMC):
    """On policy first visit Monte Carlo
       hyper : random_init, exploring
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._exploration = self.hyper_parameters.exploring


    def control(self, state):
        if np.random.uniform() < self._exploration:
            return np.random.randint(5)
        return super().control(state)

class OffPolicyMC(MonteCarlo):
    """ Off policy Monte Carlo, use weight importance sampling
        Hyper: random_init, (behavior-policy)exploration rate, policy_update_turns
        trained_episodes, weighted(importance_sampling)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._exploration = self.hyper_parameters.exploration
        self._policy_update = self.hyper_parameters.policy_update_turns
        self.algo_parameters.behavior_policy = ParameterValues(
            self.state_action_shape, random_start=False)
        self._weight = ParameterValues(self.state_action_shape,
                                       random_start=False)
        self._episodes = self.hyper_parameters.trained_episodes
        self._weight_factor = 1 - 4 * self._exploration / 5
        self._weighted_is = self.hyper_parameters.weighted
        self._record = []

    def update_behavior_policy(self):
        if self._episodes % self._policy_update == 0:
            self.algo_parameter.behavior_policy.parameter = \
            self.algo_parameters.state_action.parameter

    def update(self, state_action, reward, cur_drift, step, end):
        self.update_behavior_policy()
        if not end:
            self._record.append((state_action, reward))
        else:
            discouted_reward = 0
            weight = 1
            while self._record:
                his_state_action, his_reward = self._record.pop()
                discouted_reward *= self._discount_factor
                discouted_reward += his_reward
                self._weight[his_state_action] += weight if self._weighted_is else 1
                self.algo_parameters.state_action.parameter[his_state_action] += (
                    weight if self._weighted_is else 1 ) * (
                    discouted_reward - self.algo_parameters.parameter[
                    his_state_action]) / self._weight[his_state_action]
                if not self.algo_parameters.is_best_decision(state_action):
                    break 
                weight *= self._weight_factor
            self._record = []
            self._episodes += 1
    
    def control(self, state):
        if np.random.uniform() < self._exploration:
            return np.random.randint(5)
        return self._behavior_policy.decision(state)


class DiscoutingAwareIS(OffPolicyMC):
    """ Instead of updating the state value, we update the state-action values
        Hyper:random_init, (behavior-policy)exploration rate, policy_update_turns
        trained_episodes. 
        reserved attributes: _weight
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._his_state_action = []
        self._his_reward = np.zeros(self._step_limit)
        self._discounting = self._discount_factor ** np.arange(
            self._step_limit)
        self.is_weight = np.full(self._step_limit,self._weight_factor)

    def update(self, state_action, reward, cur_drift, step, end):
        self.update_behavior_policy()
        if not end:
            self._his_reward[step] = reward
            self._his_state_action.append(state_action)
            if not self.algo_parameters.state_action.is_best_decision(
                state_action):
                self.is_weight[step] = 0
        else:
            flat_reward = np.cumsum(self._his_reward[:step])
            for t, his_state_action in enumerate(self._his_state_action):
                is_weights = np.cumprod(self.is_weight[t:step])
                dis_aware_weight = is_weights * self._discounting[:len(
                    is_weights)]
                dis_aware_rewards = dis_aware_weight * flat_reward[t:]
                weight = dis_aware_weight[-1] + sum(dis_aware_weight) * (
                        1 - self._discout_factor) if self._weighted_is else 1
                discounted_reward = sum(dis_aware_rewards) * (
                    1 - self._discout_factor) + dis_aware_rewards[-1]
                self._weight[his_state_action] += weight
                self.algo_parameters.state_action.parameter[
                    his_state_action] += weight * (
                        discounted_reward -
                        self.algo_parameters.state_action[his_state_action]) / (
                        self._weight[his_state_action])    
            self._his_state_action = []

          

class PerDecisionMC(OffPolicyMC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._his_state_action = []
        self._his_reward = np.zeros(self._step_limit)
        self._discounting = self._discount_factor ** np.arange(
            self._step_limit)
        self.is_weight = np.full(self._step_limit,self._weight_factor)

    def update(self, state_action, reward, cur_drift, step, end):
        self.update_behavior_policy()
        if not end:
            self._his_reward[step] = reward
            self._his_state_action.append(state_action)
            if not self.algo_parameters.state_action.is_best_decision(
                state_action):
                self.is_weight[step] = 0
        else:
            for t, his_state_action in enumerate(self._his_state_action):
                is_weights = np.cumprod(self.is_weight[t:step])
                per_decision_reward = sum(is_weights * self._discounting[:len(
                    is_weights)] * self._his_reward[t:step])
                self._weight[his_state_action] += 1 
                self.algo_parameters.state_action.parameter[
                    his_state_action] += (
                        per_decision_reward -
                        self.algo_parameters.state_action[his_state_action]) / (
                        self._weight[his_state_action])    
            self._his_state_action = []
