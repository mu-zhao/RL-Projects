import logging
from multiprocessing import Pool
import numpy as np

from common.episode_utils import Episode

logger = logging.getLogger(__name__)


def run_episode(episode, algo, train=True):
    # Reset episode before each run.
    episode.reset(algo, train)
    if train:
        algo.hyper_parameters.trained_episode += 1
    while not episode.episode_end:  
        episode.update(algo)
    return episode.reward


def eval_algo(episode, algo, num_runs):
    res = []
    for _ in range(num_runs):
        res.append(run_episode(episode, algo, False))
    return np.array(res)


def parallel_evaluate_algo(algo, num_runs, config, parallel_proc):
    p = Pool(parallel_proc)
    res = [p.apply_async(run_episode, args=(
           Episode(config.params, config.torus_map), algo, False))
           for _ in range(num_runs)]
    p.close()
    p.join()
    res = [r.get() for r in res]
    return np.array(res)


class FlatTorus:
    def __init__(self, world_config):
        self.config = world_config
        self._num_episodes = self.config.params.episodes
        self._cur_episode = Episode(self.config.params, self.config.torus_map)

    def train_evaluation(self, eval_gap=10000, num_runs=10000, parallel=0):
        self._return_record = []
        while True:
            trained_episode = self.config.algo.hyper_parameters.trained_episode
            if trained_episode > self._num_episodes:
                break
            if trained_episode > 0 and trained_episode % eval_gap == 0:
                if parallel == 0:
                    eval_result = eval_algo(
                        self._cur_episode, self.config.algo, num_runs)
                else:
                    eval_result = parallel_evaluate_algo(
                        self.config.algo, num_runs, self.config, parallel)
                mean, var = eval_result.mean(), eval_result.std()
                self._return_record.append([mean, var])
                print(trained_episode, mean, var)
            run_episode(self._cur_episode, self.config.algo)  # training
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
