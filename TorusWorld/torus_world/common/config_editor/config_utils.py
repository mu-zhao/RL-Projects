from pathlib import Path
from common.map_editor.map_utils import TorusMap
from common.params_editor.params_utils import Params
from rl_algos.monte_carlo import MCES
from common.common_utils import CommonUtils, gen_path

ALGO_DICT = {'MCES': MCES}

_DATA_DIR = 'TorusWorld/data/algos'

_BASE_DIR = '/home/muzhao/my_vscode/RL-Projects/TorusWorld/data'


class WorldConfig:
    def __init__(self, world_id, dir=_BASE_DIR):
        self._dir = Path(dir)
        self._path = self._dir / "config" / f"{world_id}.json"
        self._world_id = world_id
        self.config_info = CommonUtils(self._path)
        self.params_path = gen_path("params", self._dir,
                                    self.config_info['params'])
        self.params = Params(path=self.params_path)
        self.map_path = gen_path('torus_map', self._dir,
                                 self.config_info['torus_map'])
        self.torus_map = TorusMap(path=self.map_path)
        general_algo_dir = self._dir / 'algos'
        algo_type = self.config_info['algo']
        self.algo_dir = gen_path(algo_type, general_algo_dir, self._world_id)
        rl_algo = ALGO_DICT[algo_type](self.params, self.torus_map)
        self.algo = rl_algo.load(self.algo_dir)

    def save(self):
        self.config_info.save(self._path, overwrite=True)
        self.params.save(self.params_path, overwrite=True)
        self.torus_map.save(self.map_path, overwrite=True)
        self.algo.save(self.algo_dir, overwrite=True)

    def load(self, world_id):
        return self.__init__(world_id)
