import logging

import numpy as np

from episode_utils import Episode

logger = logging.getLogger(__name__)


class FlatTorus:
    def __init__(self, world_config):
        self.config = world_config
        self._num_episodes = self.config.params.episodes
        self._cur_episode = Episode(self.config.params, self.config.torus_map)

    def run_episode(self, train=True):
        self._cur_episode.reset(self._algo, train)
        while True:
            if self._cur_episode.episode_end:
                return self._cur_episode.reward
            self._cur_episode.update(self._algo, train)

    def evaluate_algo(self, num_runs):
        run_return = np.zeros(num_runs)
        for i in range(num_runs):
            run_return[i] = self.run_episode(False)
        return run_return

    def train_evaluation(self, eval_gap=100, num_runs=100):
        self._return_record = np.zeros(self._num_episodes//eval_gap, 2)
        while self._episode < self._num_episodes:
            if self._episode > 0 and self._episode % eval_gap == 0:
                eval_result = self.evaluate_algo(num_runs)
                mean, var = eval_result.mean(), eval_result.std()
                self._return_record[self._episode//eval_gap] = mean, var
            self.run_episode()  # training
        return self._return_record

    def add_train_episodes(self, num_episodes):
        self._num_episodes += num_episodes
        self.config.params.episodes += num_episodes

    def save(self, path):
        self.config.save(path)

    @property
    def record(self):
        return self._return_record
