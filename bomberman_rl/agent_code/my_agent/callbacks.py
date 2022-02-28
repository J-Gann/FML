import numpy as np
from .learning_utilities import features_from_game_state
from enum import Enum
from joblib import dump, load
import os
import math

EPSILON = 0
MODEL_PATH = "model.joblib"
class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3 
    WAIT = 4 
    BOMB = 5

def setup(self):
    if os.path.isfile(MODEL_PATH): self.trees = load(MODEL_PATH)
    self.EPSILON = EPSILON

def act(self, game_state: dict):
    if game_state["step"] == 1: return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'WAIT']) # DO NOT PLACE BOMB IMMEDIATELY
    # Exploit or explore according to the exploration probability
    if np.random.randint(1,100) / 100 < self.EPSILON: return explore()
    else: return exploit(self, game_state)

def explore():
    choice = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'WAIT', 'BOMB'])
    return choice

def exploit(self, game_state):
    best_prediction = "WAIT"
    best_prediction_value = -math.inf
    for action in Actions:
        action = action.name
        features = np.array(features_from_game_state(self, game_state))
        prediction = self.trees[action].predict(features.reshape(1, -1))
        #print(action, features, prediction)
        if(best_prediction_value < prediction):
            best_prediction_value = prediction
            best_prediction = action
    #print("##############")
    #print("action selected", best_prediction)
    return best_prediction

