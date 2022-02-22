import dis
import os
import pickle
import random

import numpy as np
from enum import Enum, auto

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
    name, points, alive, position = game_state["self"]

    if not alive:
        return Actions.WAIT

    print(state_to_features(game_state).shape)

    return Actions.WAIT


def state_to_features(game_state: dict):
    round = game_state["round"]
    step = game_state["step"]
    name, points, alive, position = game_state["self"]

    position = np.array(position)

    field = game_state["field"]
    nearest_crates = k_closest(position, crate_positions(field), k=K_NEAREST_CRATES)
    crates_direction = nearest_crates - position

    x, y = position
    # indicates whether top, left, down, right there is a wall next to agent
    walls = np.array([field[x, y - 1], field[x - 1, y], field[x, y + 1], field[x + 1, y]]) == -1

    others = game_state["others"]
    others_positions = np.array([other[3] for other in others])
    others_positions = pad_matrix(others_positions, 3)
    others_directions = others_positions - position

    bombs = game_state["bombs"]
    close_bombs = bomb_vector(position, bombs)
    bomb_direction = close_bombs - position

    coins = np.array(game_state["coins"])
    closest_coins = k_closest(position, coins, k=K_NEAREST_COINS)
    coin_directions = closest_coins - position

    explosion_map = game_state["explosion_map"].T
    explosion_positions = positions[explosion_map == 1]
    explosion_positions = k_closest(position, explosion_positions, k=K_NEAREST_EXPLOSIONS)
    explosion_direction = explosion_positions - position

    return np.concatenate(
        [
            position,
            walls,
            others_directions.flatten(),
            crates_direction.flatten(),
            coin_directions.flatten(),
            bomb_direction.flatten(),
            explosion_direction.flatten(),
        ]
    )


def bomb_vector(position, bombs):
    """
    Bombs are only relevant in a radius of 7 boxes.
    Since it extends 3 in each direction and explodes after three steps.
    So the player can reach the explotion zone in a 7 boxes radius.

    Vector has dimension 4 x 2 and is filled with zeros.
    """

    if len(bombs) == 0:
        return np.zeros((K_NEAREST_BOMBS, 2))

    bomb_positions = np.array([bomb[0] for bomb in bombs])
    bomb_steps = np.array([bomb[1] for bomb in bombs])
    distances = manhattan_distance(position, bomb_positions)

    within_range = (distances - bomb_steps - 1 - BOMB_POWER - EXPLOSION_TIMER) < 1
    bomb_positions = bomb_positions[within_range]
    bomb_positions = k_closest(position, bomb_positions, k=K_NEAREST_BOMBS)
    return bomb_positions


def k_closest(position, objects, k=5):
    """Always returns size k x 2 (pads with zeros). Uses manhattan distance"""
    if objects.shape[0] == 0:
        return np.zeros((k, 2))

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
