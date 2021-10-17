import logging
from multiprocessing import Pool
from time import time
import numpy as np

from common.episode_utils import Episode

logger = logging.getLogger(__name__)


class FlatTorus:
    def __init__(self, world_config):
        self.config = world_config
        self._num_episodes = self.config.params.episodes
        self._cur_episode = Episode(self.config.params, self.config.torus_map)

    def run_episode(self, train=True):
        self._cur_episode.reset(self.config.algo, train)
        if train:
            self.config.algo.hyper_parameters.trained_episode += 1
        while True:
            if self._cur_episode.episode_end:
                return self._cur_episode.reward
            self._cur_episode.update(self.config.algo, train)

    def evaluate_algo(self, num_runs):

        def _eval_algo():
            return self.run_episode(False)
        p = Pool()
        res = [p.apply_async(_eval_algo, args=()) for _ in range(num_runs)]
        p.close()
        p.join()
        return np.array([r.get() for r in res])

    def train_evaluation(self, eval_gap=10000, num_runs=10000):
        self.time = time()
        self._return_record = []
        while True:
            trained_episode = self.config.algo.hyper_parameters.trained_episode
            if trained_episode > self._num_episodes:
                break
            if trained_episode > 0 and trained_episode % eval_gap == 0:
                t=time()
                print('train time ', t-self.time)
                eval_result = self.evaluate_algo(num_runs)
                self.time = time()
                print('eval_time ', self.time -t)
                mean, var = eval_result.mean(), eval_result.std()
                self._return_record.append([mean, var])
                print(trained_episode, mean, var)
            self.run_episode()  # training
        return self._return_record

    def add_train_episodes(self, num_episodes):
        self._num_episodes += num_episodes + max(
            self._num_episodes,
            self.config.algo.hyper_parameters.trained_episode)

    def save(self):
        self.config.save()

    @property
    def record(self):
        return self._return_record
