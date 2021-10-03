""" This is utilities for algorithms"""

import logging
import json
import os

import numpy as np

from common.common_utils import CommonUtils, CommonInfo, flatten

class StateAction(CommonUtils):
  def __init__(self, state_action_shape, random_start=True, behavior=False):
    self._state_action_shape = state_action_shape
    if behavior:
      self._behavior_prob = np.zeros(state_action_shape) + 0.2
    elif random_start:
      self._state_action_value = np.random.uniform(size=state_action_shape)
    else:
      self._state_action_value = np.zeros(state_action_shape)

  def decision(self, state):
    action_values = self._state_action_value[state]
    arg_max_index = np.flatnonzero( action_values == action_values.max())
    # Break tie randomly
    return np.random.choice(arg_max_index)
  
  def update_prob(self, sa_value, epsilon):
    self._behavior_prob = epsilon
    x, y, vx, vy = self._state_action_shape
    for i in range(x):
      for j in range(y):
        for k in range(vx):
          for l in range(vy):
            action_prob = self._behavior_prob[i,j,k,l]
            self._behavior_prob[i, j, k, l, 
                                np.argmax(action_prob)] = 1 - 4 * epsilon

  def soft_choice(self, state):
    return np.random.choice(5,self._behavior_prob[state])

  
  def update_prediction(self, state_action, value):
    self._state_action_value[state_action] = value

  

    
class Model(CommonUtils):
  def __init__(self):
      pass 


class Algo(CommonInfo):
  def __init__(self, params, torus_map):
    super().__init__(params, torus_map)
    self.state_action_shape = list(self._size)+[self._speed_limit] * 2 + [5]
    self._state_action = StateAction(self.state_action_shape)

  def load(self, path):
    self._state_action.load(path)
    
  def initial_state_action(self):
    pass
  
  def initial_state(self, random =True):
    if not random:
      return np.zeros(4)
    x_size, y_size = self._size
    x_loc = np.random.randint(x_size)
    y_loc = np.random.randint(y_size)
    v = np.random.randint(self._speed_limit,size=2)
    return np.array([x_loc,y_loc]+list(v))

  def initial_action(self, state, random=True):
    if not random:
      return self._state_action.decision(state)
    return np.random.randint(5)

  def control(self, state):
    self._state_action.decision(state)


  def update(self, state_action, reward, cur_drift, step, end):
    pass
  

    



 