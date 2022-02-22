import numpy as np
from .learning_utilities import features_from_game_state
from enum import Enum

class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3 
    WAIT = 4 
    BOMB = 5

def setup(self):
    # Initialize to force exploitation. This gets overwritten by the training setup to 1 in case of a training session
    self.exploration_probability = 0

def act(self, game_state: dict):

    # return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'WAIT'])

    best_prediction = "WAIT"
    best_prediction_value = 0
    for action in Actions:
        action = action.name
        features = features_from_game_state(self, game_state, action)
        prediction = self.trees[action].predict(features.reshape(-1, 1))
        if features == [1]: print(action, prediction)
        if(best_prediction_value < prediction):
            best_prediction_value = prediction
            best_prediction = action
    
    return best_prediction

    # Exploit or explore according to the exploration probability
    #if np.random.randint(0, 1) < self.exploration_probability: return explore(self, game_state)
    #else: return exploit(self, game_state)


#def explore(self, game_state):
#    return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'WAIT'])

#def exploit(self, game_state):

#    return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'WAIT']) #TODO: Use prediction

