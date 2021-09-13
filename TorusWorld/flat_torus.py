class FlatTorus:
    def __init__(self, map_id, world_size, end_locations, drift_limit=2,
                 reward_scarcity=0.1, reward_intensity=1, drift_frequency=0.2):
        self.dim = world_size
        self.end = end_locations
        self.map_id = map_id
        self.drift_limit = drift_limit
        self.reward_scarcity = 0