import logging

import numpy as np

from common.episode_utils import Episode

logger = logging.getLogger(__name__)


class FlatTorus:
    def __init__(self, world_config):
        self.config = world_config
        self._num_episodes = self.config.params.episodes
        self._cur_episode = Episode(self.config.params, self.config.torus_map)

    def run_episode(self, train=True):
        if train:
            self.config.algo.hyper_parameters.trained_episode += 1
        self._cur_episode.reset(self.config.algo, train)
        while True:
            if self._cur_episode.episode_end:
                return self._cur_episode.reward
            self._cur_episode.update(self.config.algo, train)

    def evaluate_algo(self, num_runs):
        run_return = np.zeros(num_runs)
        for i in range(num_runs):
            run_return[i] = self.run_episode(False)
        return run_return

    def train_evaluation(self, eval_gap=100, num_runs=100):
        self._return_record = []
        while True:
            trained_episode = self.config.algo.hyper_parameters.trained_episode
            if trained_episode > self._num_episodes:
                break
            if trained_episode > 0 and trained_episode % eval_gap == 0:
                eval_result = self.evaluate_algo(num_runs)
                mean, var = eval_result.mean(), eval_result.std()
                print(trained_episode, mean, var)
                self._return_record.append([mean, var])
            self.run_episode()  # training
        return self._return_record

    def add_train_episodes(self, num_episodes):
        self._num_episodes += num_episodes
        self.config.params.episodes += num_episodes

    def save(self):
        self.config.save()

    @property
    def record(self):
        return self._return_record
