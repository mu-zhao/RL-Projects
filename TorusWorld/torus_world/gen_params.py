from common.config_editor.config_utils import _BASE_DIR
from common.common_utils import gen_path
from common.params_editor.params_utils import generate_params

params_id = 2
episodes = 1000
discount_factor = 0.98
step_limit = 10000
speed_limit = 3
time_cost = 10
punitive_cost = 10000
params_path = gen_path('params', _BASE_DIR, params_id)

generate_params(params_path, _params_id=params_id, _episodes=episodes,
                _discount_factor=discount_factor, _step_limit=step_limit,
                _time_cost=time_cost,  _punitive_cost=punitive_cost,
                _speed_limit=speed_limit)
