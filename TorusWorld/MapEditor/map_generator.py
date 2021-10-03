import json
import os
import logging
import argparse

import numpy as np

logger = logging.getLogger(__name__)

_BASE_DIR = "../data/maps"


def _filter(end_loc, size):
    end_location = []
    for pos in end_loc:
        if len(pos) != 2:
            logger.warning(f"loc {pos} not coordinates")
        if not(isinstance(pos[0], int) and isinstance(pos[1], int)) or \
           pos[0] >= size[0] or pos[1] >= size[1]:
            logger.warning(f"loc {pos} not in the world")
        else:
            end_location.append(tuple(pos))
    return list(end_location)


def random_generate_map(map_id, size, end_loc, reward_config, drift_config):
    if not os.path.exists(_BASE_DIR):
        os.makedirs(_BASE_DIR, exist_ok=False)
    end_location = _filter(end_loc, size)
    if not end_location:
        logger.warning('Empty end location')
        return
    map_dict = {}
    map_dict['id'] = map_id
    map_dict['size'] = size
    map_dict['end_loc'] = end_location
    map_dict['reward_config'] = reward_config
    map_dict['drift_config'] = drift_config
    map_dict['reward'] = [[False]*size[1] for _ in range(size[0])]
    map_dict['drift'] = [[False]*size[1] for _ in range(size[0])]
    d_size = 2*drift_config[0] + 1
    mu, var_limit, r_p = reward_config
    assert var_limit > 0, "variance is not positive"
    for i in range(size[0]):
        for j in range(size[1]):
            if np.random.uniform() < drift_config[1]:
                drift = []
                for _ in range(2):
                    axis_drift = np.random.uniform(size=d_size)
                    axis_drift /= np.sum(axis_drift)
                    drift.append(list(axis_drift))
                map_dict['drift'][i][j] = drift
            if np.random.uniform() < r_p:
                map_dict['reward'][i][j] = [np.random.normal(mu, 1),
                                            np.random.uniform(var_limit)]
    logger.info('successfully generate map')
    path = f"{_BASE_DIR}/{map_id}.json"
    with open(path, 'w') as fs:
        json.dump(map_dict, fs)
    logger.info(f"Flast Torus {map_id} Saved!")


def main():
    parser = argparse.ArgumentParser(description="Arguments for map \
                                      and configration generation")
    parser.add_argument('-S', '--size', nargs=2,
                        type=int, required=True, help="size for the flat \
                        torus world")
    parser.add_argument('-D', '--drift-config', type=float, nargs=2,
                        default=[3, 0.15], help='drift limit and frequency')
    parser.add_argument('-R', '--reward-config', type=float, nargs=3,
                        default=[1, 1, 0.1],
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
