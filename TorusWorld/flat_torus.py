import logging
import os
import json

import numpy as np

logger = logging.getLogger(__name__)


class FlatTorus:
    def __init__(self, map_dict, config_dict):
        self.config = config_dict
        self.para = {}
        self.size = map_dict['size']
        self.end = set(map_dict['end_loc'])
        self.config['map_id'] = map_dict['map_id']
        self.speed_limit = config_dict['speed_limit']
        self.reward = map_dict['reward']
        self.drift = map_dict['drift']
        # This is just easier on the memory, not necessary.
        self.drift_limit = map_dict['drift_config'][0]
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if self.drift[i][j]:
                    self.drift[i][j] = np.array(self.drift[i][j])

        self.para['cost'] = np.array(self.config['kepisodes'])
        self.control_unit = np.array([[0, 0], [0, 1], [1, 0], [0, -1],
                                     [-1, 0]])
        self.drift_unit = np.arange(-self.drift_limit, self.drift_limit)

    def reset(self):
        self.loc = np.zeros(2, dtype=int)
        self.v = np.zeros(2, dtype=int)
        self.kinetic = 0
        self.cost = 0
        self.discount = 1
        self.step = 0

    def update(self):
        pass

    def decision(self):
        control = self.config['method'].control()
        self.v += self.control_unit[control]
        self.v[self.v > self.speed_limit] = self.speed_limit
        self.v[self.v < -self.speed_limit] = -self.speed_limit
        cur_kinetic = np.sum(self.v**2)
        self.loc += self.v + self.drift_effect
        self.discount *= self.config['discount']
        self.cost += self.random_reward*self.discount - 1 - max(
                        0, cur_kinetic - self.kinetic)
        self.kinetic = cur_kinetic
        self.step += 1

    def reach_end(self, episode, step_limit=100000):
        if self.step < step_limit and self.loc not in self.end:
            return False
        if self.step > step_limit:
            logger.warning(f"episode {episode} didn't reach end")
        if episode % 1000 == 0:
            self.cost_record[episode//1000] = self.cost
        return True

    @property
    def drift_effect(self):
        x, y = self.loc
        if self.drift[x][y]:
            return np.random.choice(self.drift_unit, 2, p=self.drift[x][y])
        return np.array((0, 0))

    @property
    def random_reward(self):
        x, y = self.loc
        if self.reward[x][y]:
            mu, sig = self.reward[x][y]
            return np.random.normal(mu, sig)
        return 0

    def read(self, dir):
        config_path = f"{dir}/config.json"
        with open(config_path) as f:
            config = json.load(f)
        para = {}
        para_path = f"{dir}/para"
        for d, _, filename in os.walk(para_path):
            parameter = filename.split('.')[0]
            para[parameter] = np.load(f"{para_path}/{filename}")
        logger.info('Successfully read from files!')
        return config, para

    def save(self, path):
        map_id = self.config['map_id']
        method = self.config['method']
        episodes = self.config['kepisode']
        base_path = f"{path}/{map_id}/{method}/{episodes}"
        while os.path.exists(base_path):
            base_path = f"{base_path}_{np.random.randint(1000)}"
        config_path = f"{base_path}/config.json"
        with open(config_path, 'w') as fs:
            json.dump(self.config, fs)
        for parameter in self.para:
            file_path = f"{base_path}/para/{parameter}.npy"
            np.save(file_path, self.para[parameter])
        logger.info(f"Flat Torus {map_id};{method};{episodes} saved!")
