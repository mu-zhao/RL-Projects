
import pandas as pd
from matplotlib import pyplot as plt

from common.flat_torus import FlatTorus as FT
from common.config_editor.config_utils import WorldConfig


def main():
    world_config = WorldConfig(5)
    flat_torus = FT(world_config)
    flat_torus.add_train_episodes(500)
    record = flat_torus.train_evaluation(num_runs=100,eval_gap=100)
    df = pd.DataFrame(record, columns=['mean', 'var'])
    df.plot()
    plt.show()
    flat_torus.save()

if __name__ == '__main__':
    main()


