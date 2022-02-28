import numpy as np
from .learning_utilities import features_from_game_state
from enum import Enum
from joblib import dump, load
import os

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

    self.last_step = 0
    self.total_steps = 0
    self.n = 0
    self.EPSILON = EPSILON


def act(self, game_state: dict):
    # Exploit or explore according to the exploration probability
    step = game_state["step"]

    self.bomb_timer = -1
    if step < self.last_step:
        self.total_steps += self.last_step
        self.n += 1

        print()
        print(f"steps: {self.last_step}, avg steps: {self.total_steps / self.n:.2f}")
    self.last_step = step

    if np.random.randint(1,100) / 100 < self.EPSILON: return explore()
    else: return exploit(self, game_state)

def explore():
    choice = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'WAIT', 'BOMB'])
    #print(choice, 2 * "\n")
    return choice

def exploit(self, game_state):
    best_prediction = "WAIT"
    best_prediction_value = 0
    for action in Actions:
        action = action.name
        features = np.array(features_from_game_state(self, game_state)[0])
        prediction = self.trees[action].predict(features.reshape(1, -1))
        #print(action, features, prediction, 2 * "\n")
        if(best_prediction_value < prediction):
            best_prediction_value = prediction
            best_prediction = action
    return best_prediction

