import json 


def save_to_json(obj, path):
  json_map = json.dumps(obj.__dict__)
  with open(path, 'w') as f:
    json.dump(json_map, f)


def read_from_json(obj_class, path):
  class _NewClass(obj_class):
    def __init__(self, json_dict):
      self.__dict__ = json_dict
  with open(path) as mf:
    json_dict = json.load(path)
  return _NewClass(json_dict)