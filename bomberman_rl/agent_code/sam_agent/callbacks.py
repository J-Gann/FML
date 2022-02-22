import dis
import os
import pickle
import random

import numpy as np
from enum import Enum, auto

from abc import ABCMeta, abstractmethod

# TODO: actually import these settings
# pythons import system is silly
# from ...settings import BOMB_POWER, BOMB_TIMER
# from bomberman_rl.settings import BOMB_TIMER

BOMB_POWER = 3
EXPLOSION_TIMER = 2

### feature extraction parameter ###

K_NEAREST_CRATES = 1
K_NEAREST_COINS = 1
K_NEAREST_EXPLOSIONS = 1
K_NEAREST_BOMBS = 1

####################################

NEWLINE = "\n"

positions = np.empty((17, 17, 2))

for i in range(17):
    for j in range(17):
        positions[i][j][0] = i
        positions[i][j][1] = j


class Actions:
    UP = "UP"
    RIGHT = "RIGHT"
    DOWN = "DOWN"
    LEFT = "LEFT"
    WAIT = "WAIT"
    BOMB = "BOMB"


def setup(self):
    pass


def act(self, game_state: dict) -> str:
    state_to_features(game_state)
    return Actions.WAIT


def explain_feature_vector(v):
    def bool_to_str(b):
        if int(b) == 1:
            return "yes"
        else:
            return "no"

    print(
        f"""
        position: {v[0]}, {v[1]}
        wall to:
            top:   {bool_to_str(v[2])}
            left:  {bool_to_str(v[3])}
            down:  {bool_to_str(v[4])}
            right: {bool_to_str(v[5])}
        opponent distance:
            one:    {v[6]} {v[7]}
            two:    {v[8]} {v[9]}
            three:  {v[10]} {v[11]}
        closest crate direction: {v[12]}, {v[13]}
        closest coin direction: {v[14]}, {v[15]}
        closest bomb direction: {v[16], v[17]}
        closest explosion direction: {v[18], v[19]}
        bomb possible: {bool_to_str(v[20])}
    """
    )


class Feature(metaclass=ABCMeta):
    @abstractmethod
    def compute_feature(self, game_state):
        pass

    @abstractmethod
    def explain_feature(self, feature_vector):
        pass


feature_shape = None


def state_to_features(game_state: dict):

    features_collectors = [
        Position(),
        Walls(),
        OpponentDirections(),
        CrateDirections(),
        CoinDirections(),
        BombDirections(),
        ExplosionDirections(),
        BombPossible(),
    ]

    features = [fc.compute_feature(game_state).flatten() for fc in features_collectors]

    feature_shape = list(map(len, features))

    last_index = 0
    for f, fc, idx in zip(features, features_collectors, feature_shape):
        print(fc.explain_feature(f[last_index:idx]))
    print(2 * "\n")


def format_position(v):
    "Formats a 2d vector"
    return f"{int(v[0]):2d}, {int(v[0]):2d}"


def bool_to_str(b):
    """takes bool or int, returns yes or no"""
    if int(b) == 1:
        return "yes"
    else:
        return "no"


def get_position(game_state):
    return game_state["self"][-1]


class TwoDimensionalFeature(Feature):
    def explain_feature(self, feature_vector):
        return f"{self.description}: {format_position(feature_vector)}"


class Position(TwoDimensionalFeature):
    def __init__(self):
        super().__init__()
        self.description = "current position"

    def compute_feature(self, game_state):
        return np.array(get_position(game_state))


class BombPossible(Feature):
    def compute_feature(self, game_state):
        bomb_possible = game_state["self"][-2]
        return np.array([int(bomb_possible)])

    def explain_feature(self, feature_vector):
        return f"bomb is possible: {bool_to_str(feature_vector[0])}"


class ExplosionDirections(TwoDimensionalFeature):
    def __init__(self):
        super().__init__()
        self.description = "direction of nearest explosion"

    def compute_feature(self, game_state):
        explosion_map = game_state["explosion_map"].T
        position = get_position(game_state)

        explosion_positions = positions[explosion_map == 1]
        explosion_positions = k_closest(position, explosion_positions, k=K_NEAREST_EXPLOSIONS)
        explosion_direction = explosion_positions - position
        return explosion_direction


class CoinDirections(TwoDimensionalFeature):
    def __init__(self):
        super().__init__()
        self.description = "direction of nearest coin"

    def compute_feature(self, game_state):
        position = get_position(game_state)
        coins = np.array(game_state["coins"])
        closest_coins = k_closest(position, coins, k=K_NEAREST_COINS)
        coin_directions = closest_coins - position
        return coin_directions


class OpponentDirections(Feature):
    def compute_feature(self, game_state):
        position = get_position(game_state)
        others = game_state["others"]

        others_positions = np.array([other[3] for other in others])
        others_positions = pad_matrix(others_positions, 3)
        others_directions = others_positions - position
        return others_directions

    def explain_feature(self, feature_vector):
        positions = [format_position(feature_vector[i : i + 2]) for i in range(3)]
        return f"distances of opponents: {NEWLINE}    {(NEWLINE + '    ').join(positions)}"


class Walls(Feature):
    def compute_feature(self, game_state):
        x, y = get_position(game_state)
        field = game_state["field"]
        # indicates whether top, left, down, right there is a wall next to agent
        walls = (np.array([field[x, y - 1], field[x - 1, y], field[x, y + 1], field[x + 1, y]]) == -1).astype(np.int)
        return walls

    def explain_feature(self, feature_vector):
        is_wall_str = map(bool_to_str, feature_vector)
        directions = ["top", "left", "down", "right"]

        return f"""wall to: {NEWLINE}{NEWLINE.join([f"    {dir}: {is_wall}" for dir, is_wall in zip(directions, is_wall_str)])}"""


class CrateDirections(TwoDimensionalFeature):
    def __init__(self):
        super().__init__()
        self.description = "direction of closest crate"

    def compute_feature(self, game_state):
        position = get_position(game_state)
        field = game_state["field"]

        nearest_crates = k_closest(position, crate_positions(field), k=K_NEAREST_CRATES)
        crates_direction = nearest_crates - position
        return crates_direction


class BombDirections(TwoDimensionalFeature):
    def __init__(self):
        super().__init__()
        self.description = "direction of closest bomb"

    def compute_feature(self, game_state):
        """
        Bombs are only relevant in a radius of 7 boxes.
        Since it extends 3 in each direction and explodes after three steps.
        So the player can reach the explotion zone in a 7 boxes radius.

        Vector has dimension 4 x 2 and is filled with zeros.
        """
        position = get_position(game_state)
        bombs = game_state["bombs"]

        if len(bombs) == 0:
            return np.zeros((K_NEAREST_BOMBS, 2))

        bomb_positions = np.array([bomb[0] for bomb in bombs])
        bomb_steps = np.array([bomb[1] for bomb in bombs])
        distances = manhattan_distance(position, bomb_positions)

        within_range = (distances - bomb_steps - 1 - BOMB_POWER - EXPLOSION_TIMER) < 1
        bomb_positions = bomb_positions[within_range]
        bomb_positions = k_closest(position, bomb_positions, k=K_NEAREST_BOMBS, pad=False)

        bomb_direction = bomb_positions - position
        return pad_matrix(bomb_direction, K_NEAREST_BOMBS)


def k_closest(position, objects, k=5, pad=True):
    """If pad=True returns size k x 2 (pads with zeros). Uses manhattan distance"""
    if objects.shape[0] == 0:
        return np.zeros((k, 2))

    if pad:
        objects = pad_matrix(objects, k)

    distances = manhattan_distance(position, objects)

    return objects[np.argsort(distances)][:k, :]


def crate_positions(field):
    return positions[field == 1]


def manhattan_distance(position, objects):
    """objects with shape D x 2"""
    return np.abs(objects - position).sum(axis=1)


def pad_matrix(mat, size):
    """Pads a N x 2 matrix to size x 2"""
    N = mat.shape[0]
    zero_rows_to_add = max(0, size - N)
    return np.concatenate([mat, np.zeros((zero_rows_to_add, 2))])
