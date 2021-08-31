import json
import logging
import argparse
from collections import deque

logger = logging.getLogger(__name__)

BASE_DIR = "RaceTrack/data/maps"


class TrackMap:
    def __init__(self, map_id):
        self.track_list = []
        self.start_positions = set()
        self.end_positions = set()
        self.map_id = map_id

    def gen_map(self, pos_list):
        col = set()
        for pose in pos_list:
            if len(pose) == 2:
                col |= {i for i in range(pose[0], pose[1]+1)}
            else:
                self.track_list.append(col)
                col = set()

    def gen_config(self, start_positions, end_positions):
        for pose in start_positions:
            if pose[1] not in self.track_list[pose[0]]:
                logger.warning("point {pose} not in the map region")
            else:
                self.start_positions.add(tuple(pose))
        for pose in end_positions:
            if pose[1] not in self.track_list[pose[0]]:
                logger.warning("point {pose} not in the map region")
            else:
                self.end_positions.add(tuple(pose))

    def validate(self):
        if not self.start_positions:
            logger.warning('start position is empty')
            return False
        if not self.end_positions:
            logger.warning('end position is empty')
            return False
        visited = set()
        queue = deque(self.start_positions)
        while queue:
            point = queue.popleft()
            visited.add(point)
            if point in self.end_positions:
                return True
            for a, b in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                x, y = point[0]+a, point[1]+b
                if y in self.track_list[x]:
                    if (x, y) not in visited:
                        queue.append((x, y))
        return False

    def save(self):
        map_path = f"{BASE_DIR}/map_{self.map_id}"
        config_path = f"{BASE_DIR}/config_{self.map_id}"
        with open(map_path, 'w') as fout:
            json.dump([list(s) for s in self.track_list], fout)
        with open(config_path, 'w') as fout:
            json.dump([list(self.start_positions),
                      list(self.end_positions)], fout)


def main():
    parser = argparse.ArgumentParser(description="Arguments for map \
                                      and configration generation")
    parser.add_argument('-C', '--pos-list', action='append', nargs='+',
                        type=int, required=True, help='track coordinates')
    parser.add_argument('-S', '--start-location', action='append', nargs='+',
                        type=int, required=True, help='start coordinates')
    parser.add_argument('-E', '--end-location', action='append', nargs='+',
                        type=int, required=True, help='end coordinates')
    parser.add_argument('--map-id', required=True)

    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        logger.warning("Unknown args ignored:{}".format(unknown_args))

    track_map = TrackMap(args.map_id)
    track_map.gen_map(args.pos_list)
    track_map.gen_config(args.start_location, args.end_location)

    if track_map.validate():
        track_map.save()
        logger.info('successfully create map and configration')
    else:
        logger.warning('No possible path to end line')


if __name__ == '__main__':
    main()
