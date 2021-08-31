import json
import logging 
import argparse

logger = logging.getLogger(__name__)

BASE_DIR = "RaceTrack/data"

class TrackMap:
    def __init__(self):
        self.track_list = []
        self.start_positions =set()
        self.end_positions = set()
    def gen_map(self,pos_list):
        col=set()
        for pose in pos_list:
            if len(pose) == 2:
                col|={i for i in range(pose[0],pose[1]+1)}
            else:
                self.track_list.append(col)
                col=set()
    def gen_config(self,start_positions,end_positions):
        pass 


        

def main():
    parser = argparse.ArgumentParser(description='Arguments for map and configration generation')
    parser.add_argument('-C','--pos-list', action='append', nargs ='+',type=int, 
                        required=True, help='track coordinates')
    parser.add_argument('-S','--start-location', action='append', nargs='+',type=int, 
                        required = True, help='start coordinates')
    parser.add_argument('-E','--end-location', action='append', nargs='+',type=int, 
                        required = True, help='end coordinates')
    parser.add_argument('--speed-limit',type=int)


