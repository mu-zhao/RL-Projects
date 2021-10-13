import json
import logging
import argparse

from config_utils import Config

logger = logging.getLogger(__name__)

_BASE_DIR = "TorusWorld/data/configs"


def _generate_config(config_id, kepisode, discout, step_limit, path):
  config = Config(config_id, kepisode, discout, step_limit)
  logger.info(f"successfully generate config {config_id}")
  config.save(path)


def main():
  parser = argparse.ArgumentParser(description="Arguments for configration \
                                     generation")
  parser.add_argument('-D', '--discount', type=float, required=True, 
                        help='discount factor')
  parser.add_argument('-E', '--kepisode', type=int, required=True, 
                        help='number of thousand of episodes')
  parser.add_argument('-S', '--step-limit', type=int, required=True, 
                        help='number of maximun step allowed in each episode')
  parser.add_argument('--config-id', required=True)
  parser.add_argument('-P', '--config-path', default=_BASE_DIR)
 
  args, unknown_args = parser.parse_known_args()

  if unknown_args:
    logger.warning("Unknown args ignored:{}".format(unknown_args))
  _generate_config(args.config_id, args.kepisode, args.discount,
                   args.step_limit, args.config_path)


if __name__ == '__main__':
  main()
