from common.flat_torus import FlatTorus as FT
from common.config_editor.config_utils import WorldConfig
from matplotlib.pyplot import plot
world_config = WorldConfig(3)
flat_torus = FT(world_config)
flat_torus.add_train_episodes(100000)
record = flat_torus.train_evaluation()
plot(record)
flat_torus.save()