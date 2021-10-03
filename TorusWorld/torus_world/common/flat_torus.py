import logging
import os
import json
from TorusWorld.torus_world.common.common_utils import ConfigInfo
from TorusWorld.torus_world.common.episode_utils import Episode
from rl_algos import monte_carlo
import numpy as np

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
        if not train:
          return self._cur_episode.reward
      self._cur_episode.update(self._algo, train)

  def test(self, test_gap=100, num_runs=100):
    self._return_record = np.zeros(self._num_episodes//test_gap)
    while self._episode <self._num_episodes:
      if self._episode > 0 and self._episode % test_gap == 0:  # testing
          total_run_return = 0
          for _ in range(num_runs):
            total_run_return += self.run_episode(False)
          self._return_record[self._episode//test_gap] = total_run_return / num_runs
      self.run_episode() # training
    return self._return_record

  def add_train_episodes(self,num_episodes):
      self._num_episodes += num_episodes
      self.config.params.episodes += num_episodes

  def save(self, path):
    self.config.save(path)

  def read(self, path):
    self.config.read()

  @property
  def record(self):
    return self._return_record





        









    def read(self, dir):
        config_path = f"{dir}/config.json"
        with open(config_path) as f:
            config = json.load(f)
        params = {}
        params_path = f"{dir}/para"
        for d, _, filename in os.walk(params_path):
            parameter = filename.split('.')[0]
            params[parameter] = np.load(f"{params_path}/{filename}")
        logger.info('Successfully read from files!')
        return config, params

    def save(self, path):
        map_id = self.config['map_id']
        method = self.config['method']
        kepisodes = self.config['kepisode']
        base_path = f"{path}/{map_id}/{method}/{kepisodes}"
        while os.path.exists(base_path):
            base_path = f"{base_path}_{np.random.randint(1000)}"
        config_path = f"{base_path}/config.json"
        with open(config_path, 'w') as fs:
            json.dump(self.config, fs)
        for parameter in self.params:
            file_path = f"{base_path}/para/{parameter}.npy"
            np.save(file_path, self.params[parameter])
        logger.info(f"Flat Torus {map_id};{method};{kepisodes} saved!")
