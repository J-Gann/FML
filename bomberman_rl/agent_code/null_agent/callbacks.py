import numpy as np
from agent_code.null_agent.features.feature import FeatureCollector
from enum import Enum
from joblib import load
import os
import math

# set agent to only exploit (gets overwritten to 1 during training in learning_utilities)
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
    """Author: Jonas Gann"""
    self.past_moves = []    # Maintain a list of past agent moves for usage as features
    # Create a feature extractor which is able to generate features from game states
    self.feature_collector = FeatureCollector.create_current_version()
    if os.path.isfile(MODEL_PATH):  # Load a stored model if one exists
        self.trees = load(MODEL_PATH)
    # Set agent to only explore (can be overwritten if agent is used for training)
    self.EPSILON = EPSILON


def act(self, game_state: dict):
    """Author: Jonas Gann"""
    if game_state["step"] == 1:
        self.past_moves = []    # Reset the list of past moves each new round
    # Randomly explore or exploit according to the probability of epsilon
    if np.random.randint(1, 100) / 100 < self.EPSILON:
        return explore(self)
    else:
        return exploit(self, game_state)


def explore(self):
    """Author: Jonas Gann"""
    choice = np.random.choice(
        list(Actions), p=[0.18, 0.18, 0.18, 0.18, 0.18, 0.1])   # Randomly perform an action
    self.past_moves.append(choice)  # Remember the performed action
    return choice.name


def exploit(self, game_state):
    """Author: Jonas Gann"""
    best_prediction = "WAIT"
    best_prediction_value = -math.inf
    # Generate features from the current game state
    features = self.feature_collector.compute_feature(game_state, self)
    # self.feature_collector.print_feature_summary(features)    # Log useful debugging information about the features
    for action in Actions:  # Compute the predicted reward for each action and execute the one with the largest predicted value
        action = action.name
        prediction = self.trees[action].predict(features.reshape(1, -1))
        if best_prediction_value < prediction:
            best_prediction_value = prediction
            best_prediction = action
    # Remember the performed action
    self.past_moves.append(Actions[best_prediction])
    return best_prediction
