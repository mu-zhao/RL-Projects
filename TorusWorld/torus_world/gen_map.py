from common.config_editor.config_utils import _BASE_DIR
from common.map_editor.map_utils import random_generate_map
from common.common_utils import gen_path

map_id = 2
size = [10, 10]
end_loc = [[3, 5], [9, 3], [4, 2],[8,8],[6,7]]
reward_config = [3, 1, 0.2]
drift_config = [2, 0.2]

map_path = gen_path('torus_map', _BASE_DIR, map_id)
random_generate_map(map_id, size, end_loc, reward_config, drift_config,
                    map_path)
