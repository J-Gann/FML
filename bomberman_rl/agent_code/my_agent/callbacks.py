import numpy as np

from agent_code.my_agent.features.feature import (
    AgentInBlastZone,
    BoxesInBlastRange,
    CouldEscapeOwnBomb,
    EnemiesInBlastRange,
    FeatureCollector,
    MoveNextToNearestBox,
    MoveOutOfBlastZone,
    MoveToNearestCoin,
    MoveToNearestEnemy,
    PastMoves,
    PossibleActions,
)
from agent_code.my_agent.features.movement_graph import MovementGraph
from .learning_utilities import features_from_game_state
from enum import Enum
from joblib import dump, load
import os
import math
from .path_utilities import FeatureExtraction, Actions

# isn't this also set in learning utils?
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
    self.past_moves = []
    self.feature_collector = FeatureCollector(
        MoveToNearestCoin(),
        MoveOutOfBlastZone(),
        MoveNextToNearestBox(),
        MoveToNearestEnemy(),
        EnemiesInBlastRange(),
        PastMoves(),
        BoxesInBlastRange(),
        AgentInBlastZone(),
        PossibleActions(),
        CouldEscapeOwnBomb(),
    )

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
    choice = np.random.choice(["RIGHT", "LEFT", "UP", "DOWN", "WAIT", "BOMB"])
    self.past_moves.append(choice)
    return choice


def exploit(self, game_state):

    best_prediction = "WAIT"
    best_prediction_value = -math.inf
    for action in Actions:
        action = action.name
        features = self.feature_collector.compute_feature(game_state, MovementGraph(game_state))
        prediction = self.trees[action].predict(features.reshape(1, -1))
        if best_prediction_value < prediction:
            best_prediction_value = prediction
            best_prediction = action

    self.past_moves.append(best_prediction)
    return best_prediction
