"""Generate maps for flat torus"""
import argparse
import logging

from map_utils import random_generate_map

_BASE_DIR = "TorusWorld/data/maps"
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Arguments for map \
                                      and configration generation")
    parser.add_argument('-S', '--size', nargs=2,
                        type=int, required=True, help="size for the flat \
                        torus world")
    parser.add_argument('-D', '--drift-config', type=float, nargs=2,
                        default=[3, 0.15], help='drift limit and probability')
    parser.add_argument('-R', '--reward-config', type=float, nargs=3,
                        default=[1, 1, 0.1],
                        help='reward intensity, variance, scarcity')
    parser.add_argument('-E', '--endpoints', action='append', nargs='+',
                        type=int, required=True, help='end coordinates')
    parser.add_argument('--map-id', required=True)
    parser.add_argument('--file-path', default=_BASE_DIR)

    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        logger.warning("Unknown args ignored:{}".format(unknown_args))
    random_generate_map(args.map_id, args.size, args.end_loc,
                        args.reward_config, args.drift_config, args.file_path)


if __name__ == '__main__':
    main()
