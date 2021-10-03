#%%
import json
import logging 
import os
from functools import wraps
import pathlib
from TorusWorld.torus_world.common.map_editor.map_utils import TorusMap
from TorusWorld.torus_world.common.params_editor.params_utils import Params
import numpy as np

_CONTROL_UNIT = np.array([[0,0],[0,1],[1,0],[0,-1],[-1,0]])

ALGO_DICT = {}

_DATA_DIR = ''

_BASE_DIR=""
def is_json(path):
  assert os.path.isfile(path)
  if path.split('.')[-1] == 'json':
    return True
  return False 

def flatten(*args):
  return tuple([t for l in args for t in l ])


def check_dir(func):
  @wraps(func)
  def check_dir_first(dir, *args, **kwargs):
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    return func(dir, *args, **kwargs)
  return check_dir_first



class CommonUtils:
  def __init__(self, path):
    if is_json(path):
      self = self.read_json(path)
  
  def save_json(self, path, json_map):
    with open(path, 'w') as f:
      json.dump(json_map, f)

  def read_json(self, path):
    with open(path) as mf:
      json_dict = json.load(mf)
    return json_dict
  


  @check_dir
  def save_to_np(self, dir, obj_class, obj_info):
    for key in obj_info.__dict__:
      np_path = f"{dir}/{obj_class}/{key}.npy"
      np.save(np_path,getattr(obj_info,key))

  
  def load_file(self, path):
    if is_json(path):
      return self.read_json(path)
    file_name = path.split('/')[-1]
    key = file_name.split('.')[0]
    return key, np.load(path)

  def load(self,path):
    if os.path.isfile(path):
      obj_dict = self.load_file(path)
    else:
      obj_dict = {}
      for file_path in os.listdir(path):
        k, v = self.load_file(file_path)
        obj_dict[k] = v
    return self.make_obj(obj_dict)

  def make_obj(self, obj_dict):
    class _NewClass(self.__class__):
      def __init__(self, class_dict):
        self.__dict__ = class_dict
    return _NewClass(obj_dict)

   

class WorldConfig:

  def __init__(self, world_id, dir=_BASE_DIR):
    self._path = f"{dir}/config/{world_id}.json"
    self._world_id = world_id
    self.config_info = CommonUtils(self._path)
    self.params = Params(f"{dir}/params/{self.config_info['params']}.json")
    self.torus_map = TorusMap(f"{dir}/params/{self.config_info['torus_map']}.json")
    rl_algo = ALGO_DICT[self.config_info['algo_id']](self.params, self.torus_map)
    self.algo = rl_algo.load(self.config_info['algo_path'])
  
  def save(self):
    self.config.save(self._path)

  def read(self, world_id):
    self = WorldConfig(world_id)

  

class CommonInfo():
  def __init__(self, params, torus_map):
    self._size = np.array(torus_map.size)
    self._speed_limit = params.speed_limit
    self._discount_factor = params.discount
    self._step_limit = params.step_limit
    self._time_cost = params.time_cost
    self._punitive_cost = params.punitive_cost 
    self._control_unit = _CONTROL_UNIT
    self._torus_map = torus_map



