import numpy as np
from .learning_utilities import features_from_game_state
from .features.feature import (
    AgentInBlastZone,
    AgentPosition,
    BombDirection,
    BombDropPossible,
    BoxesInBlastRange,
    CoinDirections,
    CouldEscapeOwnBomb,
    CrateDirection,
    ExplosionDirections,
    FeatureCollector,
    MoveIntoDeath,
    MoveNextToNearestBox,
    MoveOutOfBlastZone,
    MoveToNearestCoin,
    OpponentDirections,
    PossibleActions,
    Walls,
)
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
    if os.path.isfile(MODEL_PATH):
        self.trees = load(MODEL_PATH)
    self.EPSILON = EPSILON


def act(self, game_state: dict):
    if game_state["step"] == 1:
        choice = np.random.choice(["RIGHT", "LEFT", "UP", "DOWN"])  # DO NOT PLACE BOMB IMMEDIATELY
        return choice
    # Exploit or explore according to the exploration probability
    if np.random.randint(1, 100) / 100 < self.EPSILON:
        return explore()
    else:
        return exploit(self, game_state)


def explore():
    choice = np.random.choice(["RIGHT", "LEFT", "UP", "DOWN", "WAIT", "BOMB"])
    return choice


def exploit(self, game_state):
    feature_collector = FeatureCollector(
        AgentPosition(),
        BombDropPossible(),
        ExplosionDirections(),
        CoinDirections(),
        OpponentDirections(),
        Walls(),
        CrateDirection(),
        BombDirection(),
        MoveToNearestCoin(),
        MoveOutOfBlastZone(),
        MoveNextToNearestBox(),
        BoxesInBlastRange(),
        AgentInBlastZone(),
        PossibleActions(),
        # MoveIntoDeath(),
        CouldEscapeOwnBomb(),
    )

    best_prediction = "WAIT"
    best_prediction_value = -math.inf
    for action in Actions:
        action = action.name
        features = feature_collector.compute_feature(game_state)
        feature_collector.print_feature_summary(features)
        prediction = self.trees[action].predict(features.reshape(1, -1))
        if best_prediction_value < prediction:
            best_prediction_value = prediction
            best_prediction = action
    return best_prediction
