from typing import List
import numpy as np
import settings as s
from .path_utilities import setup_graph_features, print_field, _traverse_shortest_path, _move_to_nearest_coin, _index_to_node
from enum import Enum

EXPLOITATION_RATE = 0.01

class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3 
    WAIT = 4 
    BOMB = 5

def setup_training(self):
    # Set probability of exploration actions to 1
    self.exploration_probability = 1
    setup_graph_features(self, None, load=True, save=False) # Use this to setup the graph features by loading precomputed graph data from the filesystem


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    # setup_graph_features(self, new_game_state["field"], load=False, save=True) # Use this to setup the graph features by computing and saving graph data
    pass

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    # Reduce probability of randomly chosen actions
    if self.exploration_probability - EXPLOITATION_RATE < 0: self.exploration_probability = 0
    else: self.exploration_probability -= EXPLOITATION_RATE


