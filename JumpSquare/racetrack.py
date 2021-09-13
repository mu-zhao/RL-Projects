class RaceTrack:
    class Analytics:
        def __init__(self):
            self.score = 0
            self.records = []
        

    def __init__(self,track_map,config,model) -> None:
        self._map = track_map
        self._model = model 
        self._config = config 

    @property
    def map(self):
        return self._map

    @property
    def model(self):
        return self._model 

    @property
    def config(self):
        return self._config 
    
    def model_train(self,num_episode=1000000):
        score = 0
        for _ in range(num_episode):
            pass 




    

        