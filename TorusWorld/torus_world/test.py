#%%
from common.params_editor.params_utils import generate_params
from common.map_editor.map_utils import TorusMap, random_generate_map

#%%
map_id = '33'
size = [30, 20]
end_loc = [(3,5),(9,23),(44,12)]
reward_config = [3,1,0.2]
drift_config = [3,0.2]
DIR = '/home/muzhao/my_vscode/RL-Projects/TorusWorld/data'
mpath = f"{DIR}/maps/{map_id}.json"


params_id =4
episodes = 10000
discount_factor =0.95
step_limit = 5000
time_cost =6
punitive_cost = 1000 
ppath= f"{DIR}/params/{params_id}.json"

def main():
  random_generate_map(map_id, size, end_loc, reward_config, drift_config, mpath)
  #generate_params(params_id, episodes, discount_factor, step_limit, time_cost,
  #                  punitive_cost, ppath)

if __name__=='__main__':
    main()