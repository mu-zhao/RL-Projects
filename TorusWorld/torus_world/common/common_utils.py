""" We use two data types for data storage: json for general info
    and npy for algo paramters.
"""
import json
import logging
from functools import wraps
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_CONTROL_UNIT = np.array([[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]])

#  path are all Path objects
_json_file_format = {'params', 'config', 'torus_map', 'hyper_parameter'}


def gen_path(key, dir, *args):
    if key in _json_file_format:
        if args:
            file_name = Path(key) / f"{args}.json"
        else:
            file_name = Path(f"{key}.json")
        return dir / file_name
    return dir / f"{key}_{args}"  # key is algo class name, args is world id.


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
            key = k[0]*size[0] + k[1]
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
            assert naming_rule(path)
            if not overwrite and path.exists():
                logger.warning(f"File {path} already exists!")
                while path.exists():
                    path = change_name(path)
                logger.warning(f"New file name {path.name} created!")
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
    return key, np.load(path)


class CommonUtils:
    def __init__(self, **kwargs):
        if 'path' in kwargs:
            self = self.load(kwargs['path'])

    def save(self, path, overwrite=False):
        if is_json(path):
            _save_json(path, overwrite, self.__dict__)
        else:
            for key in self.__dict__:
                file_name = key+".npy"
                file_path = path / file_name
                np_array = self.__dict__[key]
                _save_np(file_path, overwrite, np_array)

    def load(self, path):
        obj_dict = {}
        if is_json(path):
            obj_dict = _read_json(path)
        else:
            for file_path in path.glob('/*.npy'):
                k, v = _read_np(file_path)
                obj_dict[k] = v
        return self.__dict__.update(obj_dict)


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
