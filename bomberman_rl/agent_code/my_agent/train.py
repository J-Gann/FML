from typing import List
import numpy as np
import settings as s
from .path_utilities import setup_graph_features, print_field, _traverse_shortest_path, _move_to_nearest_coin
from .learning_utilities import setup_learning_features, update_transitions, train_q_model
from enum import Enum
from sklearn.tree import export_graphviz
from joblib import dump, load
import events as e

EXPLOITATION_RATE = 0.1



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
    setup_learning_features(self, load_model=True)
    self.coins= 0

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if e.COIN_COLLECTED in events: self.coins += 1
    if old_game_state != None and self_action != None: update_transitions(self, old_game_state, self_action, new_game_state, events)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    if last_game_state["round"] % 100 == 0:
        coin_average = self.coins / 100
        self.coins = 0
        print(self.action_value_data["RIGHT"])
        # Reduce probability of randomly chosen actions
        if self.exploration_probability - EXPLOITATION_RATE < 0: self.exploration_probability = 0
        else: self.exploration_probability -= EXPLOITATION_RATE
        print(coin_average, self.exploration_probability)
        train_q_model(self, False)
        for action in Actions:
            export_graphviz(
                self.trees[action.name],
                out_file="./trees/"+action.name+".dot",
                feature_names=["move_up_to_coin", "move_right_to_coin","move_down_to_coin","move_left_to_coin","move_wait_to_coin","move_bomb_to_coin", 
                "move_up_to_bomb", "move_right_to_bomb","move_down_to_bomb","move_left_to_bomb","move_wait_to_bomb","move_bomb_to_bomb"]            
                )