#%%
from torus_world.common.map_editor.map_utils import random_generate_map
#%%
map_id = '12'
size = [30, 20]
end_loc = [(3,5),(9,23),(44,12)]
reward_config = [3,1,0.2]
drift_config = [3,0.2]
path = 'TorusWorld/data/maps'
#%%
random_generate_map(map_id, size, end_loc, reward_config, drift_config,path)
