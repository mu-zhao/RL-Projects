"""Utilities of the map class"""
import json
import logging
from types import SimpleNamespace

import numpy as np


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

class Map:
  def __init__(self, map_id, size, endzone):
    self.map_id = map_id
    self.size = size
    self.endzone = _filter(size, endzone)
    
  def generate_drift(self, drift_config):
    self.drift_limit = drift_config[0]
    self.drift = {}
    size = 2 * self.drift_limit + 1
    drift_prob = drift_config[1]
    for i in range(self.size[0]):
      for j in range(self.size[1]):
          if np.random.uniform() < drift_prob:
            local_drift = np.random.uniform(size=size)
            local_drift_dist = local_drift / sum(local_drift)
            self.drift[(i,j)] = list(local_drift_dist)

  def generate_reward(self, reward_config):
    self.reward = {}
    mu, var_limit, reward_prob = reward_config
    assert var_limit > 0, "variance is not positive."
    for i in range(self.size[0]):
      for j in range(self.size[1]):
        if np.random.uniform() < reward_prob:
          self.reward[(i,j)] = [np.random.normal(mu, 1),
                                np.random.uniform(var_limit)]

  def editor():

logger.info(f"Flat Torus {self.map_id} Saved!")
    






   
    
 
    
    path = f"{_BASE_DIR}/{map_id}.json"
    with open(path, 'w') as fs:
        json.dump(map_dict, fs)
    
