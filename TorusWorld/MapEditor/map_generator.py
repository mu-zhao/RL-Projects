import json
import os 
import logging
import argparse
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

_BASE_DIR = "../data/maps"


def _filter(end_loc,size):
    end_location = set()
    for x, y in end_loc:
        if x >= size[0] or y >= size[1]:
            logger.warning(f"loc ({x}, {y}) not in the world")
        else:
            end_location.add((x,y))
    return end_location


def random_generate_map(self, map_id, size, end_loc,reward_config, 
                        drift_config):
    path = f"_BASE_DIR/{map_id}"
    if os.path.exists(path) == False:
        os.makedirs(path, exist_ok = False)
    end_location =  _filter(end_loc, size)
    map_dict = {}
    map_dict['id'] = map_id
    map_dict['size'] = size
    map_dict['end_loc'] = end_loc
    map_dict['reward_config'] = reward_config
    map_dict['drift_config'] = drift_config
    mu, sig = reward_config
    assert sig > 0, "variance is not positive"
    map_dict['e_reward'] = np.random.normal(mu, sig, size=size)
    map_dict['v_reward'] = np.abs(np.random.normal(mu, sig, size=size))
    map_dict['drift'] = [[0]*size[1] for _ in range(size[0])]
    d_size = 2*drift_config[0] + 1
    for i in range(size[0]):
        for j in range(size[1]):
            if np.random.uniform() < drift_config[1]:
                drift = []
                for _ in range(2):
                    axis_drift = np.random.uniform(size=d_size)
                    axis_drift /= np.sum(axis_drift)
                    drift.append(axis_drift)
                map_dict[i][j]=drift
    with open(path, 'w') as fs:
        json.dump(map_dict, fs)
        

def main():
    parser = argparse.ArgumentParser(description="Arguments for map \
                                      and configration generation")
    parser.add_argument('-S', '--size', nargs=2,
                        type=int, required=True, help="size for the flat \
                        torus world")
    parser.add_argument('-D', '--drift-config', type=float, nargs=2,
                        default=[3,0.15], help='drift limit and frequency')
    parser.add_argument('-R', '--reward-config', type=float, nargs=2,
                        default=[1,1], 
                        help='reward intensity, variance, scarcity')
    parser.add_argument('-E', '--endpoints', action='append', nargs='+',
                        type=int, required=True, help='end coordinates')
    parser.add_argument('--map-id', required=True)

    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        logger.warning("Unknown args ignored:{}".format(unknown_args))
    random_generate_map(args.map_id, args.size, args.end_loc, 
                        args.reward_config, args.drift_config)
     

if __name__ == '__main__':
    main()
