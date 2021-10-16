
import logging
import argparse
from common.params_editor.params_utils import generate_params

logger = logging.getLogger(__name__)

_BASE_DIR = "/home/muzhao/my_vscode/RL-Projects/TorusWorld/torus_world \
    /data/params"


def main():
    parser = argparse.ArgumentParser(description="Arguments for configration \
                                      generation")
    parser.add_argument('-D', '--discount-factor', type=float, required=True,
                        help='discount factor')
    parser.add_argument('-E', '--episodes', type=int, required=True,
                        help='number of thousand of episodes')
    parser.add_argument('-P', '--punitive-cost', type=float, default=1000000,
                        help='cost if the algo exceeds step limit')
    parser.add_argument('-T', '--time-cost', type=float, required=True,
                        help='cost of each step taken')
    parser.add_argument('-S', '--step-limit', type=int, required=True,
                        help='number of maximun step allowed in each episode')
    parser.add_argument('--params-id', required=True)
    parser.add_argument('--parameter-path', default=_BASE_DIR)
    args, unknown_args = parser.parse_known_args()

    if unknown_args:
        logger.warning("Unknown args ignored:{}".format(unknown_args))
    generate_params(args.params_id, args.episodes, args.discount_factor,
                    args.step_limit, args.time_cost, args.punitive_cost,
                    args.path)


if __name__ == '__main__':
    main()
