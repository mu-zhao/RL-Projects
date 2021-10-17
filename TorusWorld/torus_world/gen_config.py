
from common.config_editor.config_utils import gen_config

world_id = 4
map_id = 1
params_id = 1
algo_type = 'OnPolicyMC'
hyper_dict = {'exploring': 0.1, 'random_init': False, 'trained_episode': 0}

gen_config(world_id, map_id, params_id, algo_type, hyper_dict)
