from common.flat_torus import FlatTorus as FT
from common.config_editor.config_utils import WorldConfig
from matplotlib import pyplot as plt
import pandas as pd
def main():
    world_config = WorldConfig(4)
    flat_torus = FT(world_config)
    flat_torus.add_train_episodes(200000)
    record = flat_torus.train_evaluation(num_runs=100,eval_gap=10)
    df = pd.DataFrame(record, columns=['mean', 'var'])
    df.plot()
    plt.show()
    flat_torus.save()

if __name__ == '__main__':
    main()
