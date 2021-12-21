
from common.config_editor.config_utils import gen_config

world_id = 6
map_id = 1
params_id = 2
algo_type = 'OffPolicyMC'
hyper_dict = {'exploration': 0.1, 'random_init': False, 'trained_episodes': 0,
              'policy_update_turns': 100, 'weighted': True}

gen_config(world_id, map_id, params_id, algo_type, hyper_dict)
