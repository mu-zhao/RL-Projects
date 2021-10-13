"""Utilities of the Params class"""
import argparse
import json
import logging

import numpy as np


from common.commom_utils import CommonUtils

logger = logging.getLogger(__name__)

class Params(CommonUtils):
  def __init__(self, **kwargs):
    if 'path' in kwargs:
      super().__init__(kwargs['path'])
    else:
      self._params_id = kwargs['params_id']
      self._episodes = kwargs["episodes"]
      self.discount_factor = kwargs["discout_factor"]
      self._step_limit =kwargs["step_limit"]
      self.time_cost = kwargs["time_cost"]
      self._punitive_cost = kwargs["punitive_cost"]
      self._algo = None

  @property
  def time_cost(self):
    return self._time_cost
  
  @time_cost.setter
  def time_cost(self, time_cost):
    assert time_cost > 0, 'time_cost is not positive'
    self._time_cost = time_cost

  @property
  def step_limit(self):
    return self._step_limit
    
  @property
  def params_id(self):
    return self._params_id

  @property
  def episodes(self):
    return self._episodes

  @episodes.setter
  def episodes(self, num):
    assert self._episodes <= num
    self._episodes = num

  @property
  def discount_factor(self):
    return self._discount_factor
  
  @discount_factor.setter
  def discout_factor(self, discout_factor):
    assert 0 <= self.discount <= 1, 'discount factor not between 0 and 1'
    self._discount_factor = discout_factor


 
  
  def save(self, path):
    self.save(path)
    logger.info(f"Configuration {self._config_id} Saved!")
