""" We use two data types for data storage: json for general info
    and npy for algo paramters.
"""
import json
import logging
from functools import wraps
from pathlib import Path
from typing import final

import numpy as np


logger = logging.getLogger(__name__)

_CONTROL_UNIT = np.array([[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]], dtype=int)

#  path are all Path objects
_json_file_format = {'params': 'params', 'config': 'configs',
                     'torus_map': 'maps'}


def gen_path(key, pdir, file_id, jformat=_json_file_format):
    if key in jformat:
        file_name = Path(f"{jformat[key]}/{key}_{file_id}.json")
        return pdir / file_name
    if key == 'hyper_parameter':
        return pdir / "hyper_parameter.json"
    # key is algo class name, args is world id.
    return pdir / f"{key}/world_{file_id}"


def is_json(path):
    if path.name.split('.')[-1] == 'json':
        return True
    return False


def flatten(*args):
    return tuple([t for ls in args for t in ls])


def change_keys(old_dict, size, to_int):
    new_dict = {}
    for k, v in old_dict.items():
        if to_int:
            key = k[0] * size[0] + k[1]
        else:
            key = (int(k) // size[0], int(k) % size[0])
        new_dict[key] = v
    return new_dict


def naming_rule(path):
    name = path.name.split('.')[0]
    return name.split('_')[-1].isnumeric()


def change_name(path):
    file, dir = path.name, path.parent
    if path.is_file():
        file, file_ex = file.split('.')
    file_name_elem = file.split('_')
    file_name_elem[-1] = str(1 + int(file_name_elem[-1]))
    new_file = '_'.join(file_name_elem)
    if path.is_file():
        new_file += '.' + file_ex
    return dir / new_file


def check_path(action):
    def decorator(func):
        if action == 'read':
            @wraps(func)
            def check_path_first(path, *args):
                assert path.exists(), 'no such file'
                return func(path, *args)
            return check_path_first

        @wraps(func)
        def check_path_first(path, overwrite, *args):
            if not overwrite:
                assert not path.exists()
            path.parent.mkdir(parents=True, exist_ok=True)
            return func(path, overwrite, *args)
        return check_path_first
    return decorator


@check_path('save')
def _save_json(path, overwrite, json_dict):
    with open(path, 'w') as f:
        json.dump(json_dict, f)


@check_path('read')
def _read_json(path):
    with open(path) as mf:
        json_dict = json.load(mf)
    return json_dict


@check_path('save')
def _save_np(file_path, overwrite, np_array):
    np.save(file_path, np_array)


@check_path('read')
def _read_np(path):
    file_name = path.name
    key = file_name.split('.')[0]
    return key, np.load(path, allow_pickle=True)


def save_obj(path, overwrite, obj_dict):
    for key in obj_dict:
        file_name = key+".npy"
        file_path = path / file_name
        np_array = obj_dict[key].parameter
        _save_np(file_path, overwrite, np_array)


def load_obj(path, obj_class):
    obj_dict = {}
    for file_path in path.glob('*.npy'):
        k, v = _read_np(file_path)
        obj_dict[k] = obj_class(np.shape(v))
        obj_dict[k].parameter = v
    return obj_dict


class ParameterValues:
    def __init__(self, params_shape, random_init=True):
        self._params_shape = params_shape
        if random_init:
            self.parameter = np.random.uniform(size=params_shape)
        else:
            self.parameter = np.zeros(params_shape)

    def decision(self, state):
        params_values = self.parameter[state]
        arg_max_index = np.flatnonzero(params_values == params_values.max())
        # Break tie randomly
        return np.random.choice(arg_max_index)

    def is_best_decision(self, state_action):
        state = state_action[:2]
        return np.max(self.parameter[state_action]) <= self.parameter[
            state_action]

    def update_prediction(self, state_action, value):
        self.parameter[state_action] = value

    def change_policy(self, behavior_policy):
        self.parameter = behavior_policy

    def gen_policy(self, epsilon):
        last_axis = len(self._params_shape) - 1
        mask = self.parameter.max(
            axis=last_axis, keepdims=True) == self.parameter
        return np.where(mask, 1 - 4 * epsilon, epsilon)


class CommonUtils:
    def __init__(self, **kwargs):
        if 'path' in kwargs:
            self = self.load(kwargs['path'])

    def save(self, path, overwrite=False):
        if is_json(path):
            _save_json(path, overwrite, self.__dict__)
        else:
            save_obj(path, overwrite, self.__dict__)

    def load(self, path):
        if is_json(path):
            obj_dict = _read_json(path)
        else:
            obj_dict = load_obj(path, ParameterValues)
        self.__dict__.update(obj_dict)


class CommonInfo():
    def __init__(self, params, torus_map):
        self._size = np.array(torus_map.size)
        self._loc_num = self._size[0] * self._size[1]
        self._speed_limit = params.speed_limit
        self._discount_factor = params.discount_factor
        self._step_limit = params.step_limit
        self._time_cost = params.time_cost
        self._punitive_cost = params.punitive_cost
        self._control_unit = _CONTROL_UNIT
        self._torus_map = torus_map
        self._endzone = torus_map.get_endzone()
        self._buffer = min(100, self._step_limit / self._loc_num)
        self._drift_effect = {loc: torus_map.drift_effect(loc, self._buffer)
                              for loc in torus_map.driftzone}
        self._random_reward = {loc: torus_map.random_reward(loc, self._buffer)
                               for loc in torus_map.rewardzone}
