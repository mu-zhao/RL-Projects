"""Utilities of the map class"""
import json
import logging

import numpy as np

from common.commom_utils import CommonUtils

logger = logging.getLogger(__name__)

def _filter(size, end_loc):
  endzone = []
  for pos in end_loc:
    if len(pos) != 2:
      logger.warning(f"loc {pos} not 2d coordinates")
    if not(isinstance(pos[0], int) and isinstance(pos[1], int)) or \
      pos[0] >= size[0] or pos[1] >= size[1]:
      logger.warning(f"loc {pos} not in the world")
    else:
      endzone.append(tuple(pos))
  assert endzone, "valid endzone is empty!"
  return endzone

class TorusMap(CommonUtils):
  def __init__(self, **kwargs):
    if 'path' in kwargs:
      super().__init__(kwargs['path'])
    else:
      self._map_id = kwargs["map_id"]
      self._size = kwargs["size"]
      self._endzone = _filter(kwargs["size"], kwargs["endzone"])
    self._reward = {}
    self._drift = {}
    
  def generate_drift(self, drift_config):
    drift_limit = drift_config[0]
    self._drift_unit = list(range(-drift_limit, drift_limit))
    size = 2 * self.drift_limit + 1
    drift_prob = drift_config[1]
    for i in range(self.size[0]):
      for j in range(self.size[1]):
          if np.random.uniform() < drift_prob:
            local_drift = np.random.uniform(size=size)
            local_drift_dist = local_drift / sum(local_drift)
            self.drift[(i,j)] = list(local_drift_dist)

  def generate_reward(self, reward_config):
    mu, var_limit, reward_prob = reward_config
    assert var_limit > 0, "variance is not positive."
    for i in range(self.size[0]):
      for j in range(self.size[1]):
        if np.random.uniform() < reward_prob:
          self._reward[(i,j)] = [np.random.normal(mu, 1),
                                 np.random.uniform(var_limit)]


  @property
  def size(self):
    return self._size

  @property
  def map_id(self):
    return self._map_id

  @property
  def endzone(self):
    return self._endzone
  
  @endzone.setter
  def endzone(self, new_endzone):
    self._endzone = _filter(self._size, new_endzone)


  def drift_effect(self, loc):
    if loc in self._drift:
      return np.random.choice(self._drift_unit,
                              2, p=self.drift[loc])
    return np.array((0, 0))

  def random_reward(self, loc):
    if loc in self._reward:
      mu, sig = self._reward(loc)
      return np.random.normal(mu, sig)
    return 0

  def is_end(self, loc):
    return loc in self._endzone
    
  def save(self, path):
    self.save_json(path)
    logger.info(f"Flat Torus {self._map_id} Saved!")
