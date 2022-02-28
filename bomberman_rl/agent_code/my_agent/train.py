from typing import List
import numpy as np
import settings as s
from .learning_utilities import setup_learning_features, train_q_model, update_action_value_data
from enum import Enum
from joblib import dump, load
import events as e

class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3 
    WAIT = 4 
    BOMB = 5

def setup_training(self):
    setup_learning_features(self)

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state == None or self_action == None:
        # First step of round
        return
    update_action_value_data(self, old_game_state, self_action, new_game_state, events)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    train_q_model(self, last_game_state, 5)

