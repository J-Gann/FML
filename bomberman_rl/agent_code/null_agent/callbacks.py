import numpy as np

from agent_code.null_agent.features.feature import FeatureCollector, MoveNextToNearestBox, MoveToNearestCoin
from enum import Enum
from joblib import load
import os
import math

EPSILON = 0 # set agent to only exploit (gets overwritten to 1 during training in learning_utilities)
MODEL_PATH = "model.joblib"


class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    WAIT = 4
    BOMB = 5


def setup(self):
    self.past_moves = []
    self.feature_collector = FeatureCollector.create_current_version()

    if os.path.isfile(MODEL_PATH):
        self.trees = load(MODEL_PATH)
    self.EPSILON = EPSILON


def act(self, game_state: dict):
    if game_state["step"] == 1:
        self.past_moves = []
    if np.random.randint(1, 100) / 100 < self.EPSILON:
        return explore(self)
    else:
        return exploit(self, game_state)


def explore(self):
    choice = np.random.choice(list(Actions), p=[0.18, 0.18, 0.18, 0.18, 0.18, 0.1])
    self.past_moves.append(choice)
    return choice.name


def exploit(self, game_state):

    best_prediction = "WAIT"
    best_prediction_value = -math.inf

    features = self.feature_collector.compute_feature(game_state, self)
    #self.feature_collector.print_feature_summary(features)

    for action in Actions:
        action = action.name
        prediction = self.trees[action].predict(features.reshape(1, -1))
        if best_prediction_value < prediction:
            best_prediction_value = prediction
            best_prediction = action

    self.past_moves.append(Actions[best_prediction])
    #print("Performed action: ", best_prediction)

    return best_prediction
