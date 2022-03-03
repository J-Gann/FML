from abc import ABCMeta, abstractmethod

import numpy as np
from enum import Enum, auto

from feature_utils import (
    crate_positions,
    format_boolean,
    format_position,
    k_closest,
    manhattan_distance,
    pad_matrix,
    positions,
)

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


# needed because of a problem with format strings and \n
NEWLINE = "\n"


class Feature(metaclass=ABCMeta):
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    @abstractmethod
    def compute_feature(self, game_state: dict) -> np.array:
        pass

    def format_feature(self, feature_vector: np.array) -> str:
        return str(feature_vector)

    def explain_feature(self, feature_vector: np.array) -> str:
        return f"{self.description}: {self.format_feature(feature_vector)}"


def get_position(game_state: dict) -> np.array:
    return game_state["self"][-1]


class TwoDimensionalFeature(Feature):
    def format_feature(self, feature_vector: np.array) -> str:
        return format_position(feature_vector)


class BooleanFeature(Feature):
    def format_feature(self, feature_vector: np.array) -> str:
        return format_boolean(feature_vector)


class Position(TwoDimensionalFeature):
    def __init__(self) -> None:
        super().__init__("agent_position", "current position")

    def compute_feature(self, game_state: dict) -> np.array:
        return np.array(get_position(game_state))


class BombPossible(BooleanFeature):
    def __init__(self) -> None:
        super().__init__("bomb_possible", "dropping bomb is possible")

    def compute_feature(self, game_state: dict) -> np.array:
        bomb_possible = game_state["self"][-2]
        return np.array([int(bomb_possible)])


class ExplosionDirections(TwoDimensionalFeature):
    def __init__(self) -> None:
        super().__init__("explosion_direction", "direction of nearest explosion")

    def compute_feature(self, game_state: dict) -> np.array:
        explosion_map = game_state["explosion_map"].T
        position = get_position(game_state)

        explosion_positions = positions[explosion_map == 1]
        explosion_positions = k_closest(position, explosion_positions, k=K_NEAREST_EXPLOSIONS)
        explosion_direction = explosion_positions - position
        return explosion_direction


class CoinDirections(TwoDimensionalFeature):
    def __init__(self) -> None:
        super().__init__("coin_direction", "direction of nearest coin")

    def compute_feature(self, game_state: dict) -> np.array:
        position = get_position(game_state)
        coins = np.array(game_state["coins"])
        closest_coins = k_closest(position, coins, k=K_NEAREST_COINS)
        coin_directions = closest_coins - position
        return coin_directions


class OpponentDirections(Feature):
    def __init__(self) -> None:
        super().__init__("opponent_direction", "distances of opponents")

    def compute_feature(self, game_state: dict) -> np.array:
        position = get_position(game_state)
        others = game_state["others"]

        others_positions = np.array([other[3] for other in others])
        others_positions = pad_matrix(others_positions, 3)
        others_directions = others_positions - position
        return others_directions

    def format_feature(self, feature_vector: np.array) -> str:
        positions = [format_position(feature_vector[i : i + 2]) for i in range(3)]
        sep = "\n    "
        return sep + sep.join(positions)


class Walls(Feature):
    def __init__(self) -> None:
        super().__init__("walls", "wall to")

    def compute_feature(self, game_state: dict) -> np.array:
        x, y = get_position(game_state)
        field = game_state["field"]
        # indicates whether top, left, down, right there is a wall next to agent
        walls = (np.array([field[x, y - 1], field[x - 1, y], field[x, y + 1], field[x + 1, y]]) == -1).astype(np.int)
        return walls

    def explain_feature(self, feature_vector):
        sep = "\n    "
        return sep + sep.join(
            [
                f"{dir}: {is_wall}"
                for dir, is_wall in zip(["top", "left", "down", "right"], map(format_boolean, feature_vector))
            ]
        )


# TODO: more than one crate
class CrateDirection(TwoDimensionalFeature):
    def __init__(self) -> None:
        super().__init__("crate_direction", "direction of closest crate")

    def compute_feature(self, game_state: dict) -> np.array:
        position = get_position(game_state)
        field = game_state["field"]

        nearest_crates = k_closest(position, crate_positions(field), k=K_NEAREST_CRATES)
        crates_direction = nearest_crates - position
        return crates_direction


# TODO: more than one bomb
class BombDirection(TwoDimensionalFeature):
    def __init__(self) -> None:
        super().__init__("bomb_direction", "direction of closest bomb")

    def compute_feature(self, game_state: dict) -> np.array:
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
