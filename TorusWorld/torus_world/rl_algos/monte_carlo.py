" Monte Carlo algorithms."

import numpy as np

from common.common_utils import ParameterValues
from common.algo_utils import Algo


class MonteCarlo(Algo):
    def __init__(self, *args, **kwargs):
        """ base algo class, not a functional algo.
            hyper: random_init, trained_episodes
        """
        super().__init__(*args, **kwargs)
        self.state_action_shape = list(self._size)+[
            2 * self._speed_limit + 1] * 2 + [5]
        random_start = self.hyper_parameters.random_init
        self.algo_parameters.state_action = ParameterValues(
            self.state_action_shape, random_start)

    def update(self, ep_info):
        if ep_info.is_training:
            self.prediction(ep_info)
        ep_info.update_episode()
        ep_info.update_action(self.control(ep_info.state))

    def initial_state_action(self):  # non random start
        return super().initial_state(False), super().initial_action(False)


class FirstVisitMC(MonteCarlo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_visit = np.zeros(self.state_action_shape, dtype=int)
        self._action = None
        self._discounting = self._discount_factor ** np.arange(
            self._step_limit)
        self._episode_record = np.zeros(self._step_limit)
        self._first_visit = {}

    def prediction(self, ep_info):
        self._episode_record[ep_info.step] = ep_info.cur_reward
        if ep_info.state_action not in self._first_visit:
            self._first_visit[ep_info.step] = ep_info.state_action
        if ep_info.episode_end:
            cum_discounted_reward = np.cumsum(
                self._discounting[:ep_info.step] *
                self._episode_record[:ep_info.step][::-1])[::-1]
            for step in self._first_visit:
                state_action = self._first_visit[step]
                self._num_visit[state_action] += 1
                # Learning rate for averaging.
                learning_rate = 1 / self._num_visit[state_action]
                # Leaning target.
                target = cum_discounted_reward[step]
                self.algo_parameters.state_action.update_prediction(
                    state_action, target, learning_rate)
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
        Hyper: random_init, (behavior-policy)exploration rate,
        policy_update_turns, trained_episodes, weighted(importance_sampling)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._exploration = self.hyper_parameters.exploration
        self._policy_update = self.hyper_parameters.policy_update_turns
        self.algo_parameters.behavior_policy = ParameterValues(
            self.state_action_shape, random_init=False)
        self._weight = ParameterValues(self.state_action_shape,
                                       random_init=False)
        self._episodes = self.hyper_parameters.trained_episodes
        self._weight_factor = 1
        self._weighted_importance_sampling = self.hyper_parameters.weighted
        if self._weighted_importance_sampling:
            self._weight_factor = 1 - 4 * self._exploration / 5
        self._record = []

    def update_behavior_policy(self):
        if self._episodes % self._policy_update == 0:
            # TODO: use better ways to reperesent and update behavior policy.
            self.algo_parameters.behavior_policy.parameter[:] = (
                self.algo_parameters.state_action.parameter)

    def prediction(self, ep_info):
        self.update_behavior_policy()
        if not ep_info.episode_end:
            self._record.append((ep_info.state_action, ep_info.cur_reward))
        else:
            discouted_reward = 0
            weight = 1
            while self._record:
                his_state_action, his_reward = self._record.pop()
                # Target = R + lambda * Q(S_(t+1), A_(t+1))
                discouted_reward *= self._discount_factor
                discouted_reward += his_reward
                # Learing rate
                self._weight[his_state_action] += weight
                learning_rate = weight / self._weight[his_state_action]

                self.algo_parameters.state_action.update_prediction(
                    his_state_action, discouted_reward, learning_rate)
                if not self.algo_parameters.is_best_decision(
                        ep_info.state_action):
                    break
                weight *= self._weight_factor
            self._record = []
            self._episodes += 1

    def control(self, state):
        if np.random.uniform() < self._exploration:
            return np.random.randint(5)
        return self.algo_parameters.behavior_policy.decision(state)


class FinerOffPolicyMC(OffPolicyMC):
    """ Base class, not functional algo.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._his_state_action = []
        self._his_reward = np.zeros(self._step_limit)
        self._discounting = self._discount_factor ** np.arange(
            self._step_limit)
        self.is_weight = np.full(self._step_limit, self._weight_factor)

    def prediction(self, ep_info):
        self.update_behavior_policy()
        if not ep_info.episode_end:
            self._record(ep_info)
        else:
            self._update_prediction()

    def _record(self, ep_info):
        self._his_reward[ep_info.step] = ep_info.cur_reward
        self._his_state_action.append(ep_info.state_action)
        if not self.algo_parameters.state_action.is_best_decision(
                ep_info.state_action):
            self.is_weight[ep_info.step] = 0

    def _update_prediction(self):
        raise NotImplementedError('Must be implemented by subclasses')


class DiscoutingAwareIS(FinerOffPolicyMC):
    """ Instead of updating the state value, we update the state-action values
        Hyper:random_init, (behavior-policy)exploration rate,
        policy_update_turns, trained_episodes.
        reserved attributes: _weight
    """
    def _update_prediction(self, ep_info):
        flat_reward = np.cumsum(self._his_reward[:ep_info.step])
        for t, his_state_action in enumerate(self._his_state_action):
            is_weights = np.cumprod(self.is_weight[t:ep_info.step])
            dis_aware_weight = is_weights * self._discounting[:len(
                is_weights)]
            dis_aware_rewards = dis_aware_weight * flat_reward[t:]
            if self._weighted_importance_sampling:
                weight = dis_aware_weight[-1] + sum(dis_aware_weight) * (
                    1 - self._discout_factor)
            else:
                weight = 1
            self._weight[his_state_action] += weight
            # Learing rate
            learing_rate = weight / self._weight[his_state_action]
            # Target
            discounted_reward = sum(dis_aware_rewards) * (
                1 - self._discout_factor) + dis_aware_rewards[-1]
            # Update prediction
            self.algo_parameters.state_action.update_prediction(
                his_state_action, discounted_reward, learing_rate)
        self._his_state_action = []


class PerDecisionMC(FinerOffPolicyMC):

    def _update_prediction(self, ep_info):
        for t, his_state_action in enumerate(self._his_state_action):
            is_weights = np.cumprod(self.is_weight[t:ep_info.step])
            # Target
            per_decision_reward = sum(is_weights * self._discounting[:len(
                is_weights)] * self._his_reward[t:ep_info.step])
            # learning rate
            self._weight[his_state_action] += 1
            learning_rate = 1 / self._weight[his_state_action]
            # Update prediction
            self.algo_parameters.state_action.update_prediction(
                his_state_action, per_decision_reward, learning_rate)
        self._his_state_action = []
