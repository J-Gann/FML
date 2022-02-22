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
    round = game_state["round"]
    step = game_state["step"]
    name, points, alive, position = game_state["self"]

    if not alive:
        return Actions.WAIT

    position = np.array(position)

    field = game_state["field"]
    nearest_crates = k_closest(position, crate_positions(field), k=3)
    crates_direction = nearest_crates - position

    others = game_state["others"]
    others_positions = np.array([other[3] for other in others])
    others_directions = others_positions - position

    bombs = game_state["bombs"]
    close_bombs = bomb_vector(position, bombs)
    bomb_direction = close_bombs - position

    coins = np.array(game_state["coins"])
    closest_coins = k_closest(position, coins, k=3)
    coin_directions = closest_coins - position

    explosion_map = game_state["explosion_map"].T
    explosion_positions = positions[explosion_map == 1]
    explosion_positions = k_closest(position, explosion_positions, k=4)
    explosion_direction = explosion_positions - position

    return Actions.WAIT


def state_to_features(game_state: dict):
    round = game_state["round"]
    step = game_state["step"]
    name, points, alive, position = game_state["self"]

    if not alive:
        return Actions.WAIT

    position = np.array(position)

    field = game_state["field"]
    nearest_crates = k_closest(position, crate_positions(field), k=3)
    crates_direction = nearest_crates - position

    others = game_state["others"]
    others_positions = np.array([other[3] for other in others])
    others_directions = others_positions - position

    bombs = game_state["bombs"]
    close_bombs = bomb_vector(position, bombs)
    bomb_direction = close_bombs - position

    coins = np.array(game_state["coins"])
    closest_coins = k_closest(position, coins, k=3)
    coin_directions = closest_coins - position

    explosion_map = game_state["explosion_map"].T
    explosion_positions = positions[explosion_map == 1]
    explosion_positions = k_closest(position, explosion_positions, k=4)
    explosion_direction = explosion_positions - position

    return np.concatenate(
        [
            position,
            others_directions.flatten(),
            crates_direction.flatten(),
            coin_directions.flatten(),
            explosion_direction.flatten(),
        ]
    )
    return


def bomb_vector(position, bombs):
    """
    Bombs are only relevant in a radius of 7 boxes.
    Since it extends 3 in each direction and explodes after three steps.
    So the player can reach the explotion zone in a 7 boxes radius.

    Vector has dimension 4 x 2 and is filled with zeros.
    """

    if len(bombs) == 0:
        return np.zeros((4, 2))

    bomb_positions = np.array([bomb[0] for bomb in bombs])
    bomb_steps = np.array([bomb[1] for bomb in bombs])
    distances = manhattan_distance(position, bomb_positions)

    within_range = (distances - bomb_steps - 1 - BOMB_POWER - EXPLOSION_TIMER) < 1
    bomb_positions = bomb_positions[within_range]
    bomb_positions = np.concatenate([bomb_positions, np.zeros((4 - bomb_positions.shape[0], 2))])

    return bomb_positions


def k_closest(position, objects, k=5):
    """Uses manhattan distance"""
    if objects.shape[0] == 0:
        return objects

    distances = manhattan_distance(position, objects)

    return objects[np.argsort(distances)][:k, :]


def crate_positions(field):
    return positions[field == 1]


def manhattan_distance(position, objects):
    """objects with shape D x 2"""
    return np.abs(objects - position).sum(axis=1)
