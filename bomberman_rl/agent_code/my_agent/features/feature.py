from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import numpy as np
from enum import Enum
import copy
from .movement_graph import MovementGraph, to_node
from .actions import Actions

BOMB_POWER = 3
BOMB_TIMER = 4
EXPLOSION_TIMER = 2

import numpy as np
from enum import Enum, auto

from .feature_utils import (
    crate_positions,
    format_boolean,
    format_position,
    k_closest,
    manhattan_distance,
    pad_matrix,
    positions,
    ROWS,
    COLS,
    action_new_index,
    get_agent_position,
    camel_to_snake_case,
)

### feature extraction parameter ###

K_NEAREST_CRATES = 1
K_NEAREST_COINS = 1
K_NEAREST_EXPLOSIONS = 1
K_NEAREST_BOMBS = 1

####################################


class Feature(metaclass=ABCMeta):
    def name(self):
        return camel_to_snake_case(type(self).__name__)

    def description(self):
        return self.name().replace("_", " ")

    @abstractmethod
    def dim(self) -> int:
        pass

    @abstractmethod
    def compute_feature(self, game_state: dict, movement_graph: MovementGraph) -> np.array:
        pass

    def format_feature(self, feature_vector: np.array) -> str:
        return str(feature_vector)

    def explain_feature(self, feature_vector: np.array) -> str:
        return f"{self.description()}: {self.format_feature(feature_vector)}"


class TwoDimensionalFeature(Feature):
    def dim(self) -> int:
        return 2

    def format_feature(self, feature_vector: np.array) -> str:
        return format_position(feature_vector)


class BooleanFeature(Feature):
    def dim(self) -> int:
        return 1

    def format_feature(self, feature_vector: np.array) -> str:
        return format_boolean(feature_vector)


class ActionFeature(Feature):
    def dim(self) -> int:
        return 1

    def format_feature(self, feature_vector: np.array) -> str:
        assert len(feature_vector.shape) == 1 and feature_vector.shape[0] == 1, "must be scalar"

        return list(Actions)[int(feature_vector[0])].name


class AgentPosition(TwoDimensionalFeature):
    def compute_feature(self, game_state: dict, movement_graph: MovementGraph) -> np.array:
        return np.array(get_agent_position(game_state))


class BombDropPossible(BooleanFeature):
    def compute_feature(self, game_state: dict, movement_graph: MovementGraph) -> np.array:
        bomb_possible = game_state["self"][-2]
        return np.array([int(bomb_possible)])


# TODO: more than one field
class ExplosionDirections(TwoDimensionalFeature):
    def compute_feature(self, game_state: dict, movement_graph: MovementGraph) -> np.array:
        explosion_map = game_state["explosion_map"].T
        position = get_agent_position(game_state)

        explosion_positions = positions[explosion_map == 1]
        explosion_positions = k_closest(position, explosion_positions, k=K_NEAREST_EXPLOSIONS)
        explosion_direction = explosion_positions - position
        return explosion_direction


class CoinDirections(TwoDimensionalFeature):
    def compute_feature(self, game_state: dict, movement_graph: MovementGraph) -> np.array:
        position = get_agent_position(game_state)
        coins = np.array(game_state["coins"])
        closest_coins = k_closest(position, coins, k=K_NEAREST_COINS)
        coin_directions = closest_coins - position
        return coin_directions


class OpponentDirections(Feature):
    def __init__(self, number_of_opponents=3) -> None:
        super().__init__()
        self.number_of_opponents = number_of_opponents

    def dim(self) -> int:
        return 2 * self.number_of_opponents

    def compute_feature(self, game_state: dict, movement_graph: MovementGraph) -> np.array:
        position = get_agent_position(game_state)
        others = game_state["others"]

        others_positions = np.array([other[3] for other in others])
        others_positions = pad_matrix(others_positions, self.number_of_opponents)
        others_directions = others_positions - position
        return others_directions

    def format_feature(self, feature_vector: np.array) -> str:
        positions = [format_position(feature_vector[i : i + 2]) for i in range(3)]
        sep = "\n    "
        return sep + sep.join(positions)


class Walls(Feature):
    def dim(self) -> int:
        return 4

    def compute_feature(self, game_state: dict, movement_graph: MovementGraph) -> np.array:
        x, y = get_agent_position(game_state)
        field = game_state["field"]
        # indicates whether top, left, down, right there is a wall next to agent
        walls = (np.array([field[x, y - 1], field[x - 1, y], field[x, y + 1], field[x + 1, y]]) == -1).astype(np.int)
        return walls

    def format_feature(self, feature_vector):
        sep = "\n    "
        return sep + sep.join(
            [
                f"{dir}: {is_wall}"
                for dir, is_wall in zip(["top", "left", "down", "right"], map(format_boolean, feature_vector))
            ]
        )


# TODO: more than one crate
class CrateDirection(TwoDimensionalFeature):
    def compute_feature(self, game_state: dict, movement_graph: MovementGraph) -> np.array:
        position = get_agent_position(game_state)
        field = game_state["field"]

        nearest_crates = k_closest(position, crate_positions(field), k=K_NEAREST_CRATES)
        crates_direction = nearest_crates - position
        return crates_direction


# TODO: more than one bomb
class BombDirection(TwoDimensionalFeature):
    def compute_feature(self, game_state: dict, movement_graph: MovementGraph) -> np.array:
        """
        Bombs are only relevant in a radius of 7 boxes.
        Since it extends 3 in each direction and explodes after three steps.
        So the player can reach the explotion zone in a 7 boxes radius.

        Vector has dimension 4 x 2 and is filled with zeros.
        """
        position = get_agent_position(game_state)
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


class MoveToNearestCoin(ActionFeature):
    def compute_feature(self, game_state: dict, movement_graph: MovementGraph) -> np.array:
        return np.array([movement_graph.next_step_to_nearest_index(game_state["coins"]).value])


def move_out_of_blast_zone(game_state: dict, movement_graph: MovementGraph):
    agent_position = get_agent_position(game_state)
    free_indices = []
    blast_indices = movement_graph.blast_indices()

    for x in range(COLS):
        for y in range(ROWS):
            index = (x, y)
            obstructed = movement_graph.index_obstructed(index)
            in_blast_zone = index in blast_indices
            if not obstructed and not in_blast_zone:
                free_indices.append(index)
    if agent_position in blast_indices:
        return np.array([movement_graph.next_step_to_nearest_index(free_indices).value])
    else:
        return np.array([Actions.NONE.value])


class MoveOutOfBlastZone(ActionFeature):
    def compute_feature(self, game_state: dict, movement_graph: MovementGraph) -> np.array:
        return move_out_of_blast_zone(game_state, movement_graph)


class MoveNextToNearestBox(ActionFeature):
    def compute_feature(self, game_state: dict, movement_graph: MovementGraph) -> np.array:
        box_neighbors = []
        indices = np.argwhere(game_state["field"] == 1)
        tuple_indices = [(index[0], index[1]) for index in indices]
        # find all neighbors of boxes the agent can move to (the box itsel is always out of range for the agent)
        for x, y in tuple_indices:
            neighbors = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
            for nx, ny in neighbors:
                if not movement_graph.index_obstructed((nx, ny)):
                    box_neighbors.append((nx, ny))
        if get_agent_position(game_state) in box_neighbors:
            return np.array([Actions.NONE.value])
        return np.array([movement_graph.next_step_to_nearest_index(box_neighbors).value])


class BoxesInBlastRange(Feature):
    def dim(self) -> int:
        return 1

    def compute_feature(self, game_state: dict, movement_graph: MovementGraph) -> np.array:
        sum = 0
        x, y = get_agent_position(game_state)
        field = game_state["field"]

        for i in range(4):
            if 0 < x + i < COLS and field[x + i, y] == 1:
                sum += 1
            if COLS > x - i > 0 and field[x - i, y] == 1:
                sum += 1
            if 0 < y + i < ROWS and field[x, y + i] == 1:
                sum += 1
            if ROWS > y - i > 0 and field[x, y - i] == 1:
                sum += 1
        return np.array([sum])


class AgentInBlastZone(BooleanFeature):
    def compute_feature(self, game_state: dict, movement_graph: MovementGraph) -> np.array:
        in_blast_zone = to_node(get_agent_position(game_state)) in movement_graph.blast_nodes()
        return np.array([int(in_blast_zone)])


class PossibleActions(Feature):
    def dim(self) -> int:
        return len(Actions) - 1

    def compute_feature(self, game_state: dict, movement_graph: MovementGraph) -> np.array:
        agent_position = get_agent_position(game_state)
        res = []
        for action in Actions:
            if action == Actions.NONE:
                pass
            elif action == Actions.WAIT:
                res.append(1)
            elif action == Actions.BOMB:
                res.append(int(game_state["self"][2]))
            else:
                new_index = action_new_index(agent_position, action)
                obstructed = movement_graph.index_obstructed(new_index)
                res.append(int(not obstructed))
        return np.array(res)


class MoveIntoDeath(Feature):
    def dim(self) -> int:
        return len(Actions) - 1

    def compute_feature(self, game_state: dict, movement_graph: MovementGraph) -> np.array:
        res = []
        for action in Actions:
            if action == Actions.NONE:
                pass
            elif action == Actions.WAIT or action == Actions.BOMB:
                agent_index_blast_next_step = False
                for bomb in self.game_state["bombs"]:
                    (x, y) = bomb[0]
                    time_till_explosion = bomb[1]
                    blast_indices = []
                    for i in range(4):
                        if 0 < x + i < COLS:
                            blast_indices.append((x + i, y))
                        if COLS > x - i > 0:
                            blast_indices.append((x - i, y))
                        if 0 < y + i < ROWS:
                            blast_indices.append((x, y + i))
                        if ROWS > y - i > 0:
                            blast_indices.append((x, y - i))
                    if self.agent_index in blast_indices and time_till_explosion == 0:
                        agent_index_blast_next_step = True
                res.append(int(agent_index_blast_next_step))
            else:
                # FIXME: this should not unpack action
                (nx, ny) = action
                next_node_in_active_blast = not self.explosion_map[nx][ny] == 0
                next_node_in_active_blast_next_step = False
                for bomb in self.game_state["bombs"]:
                    (x, y) = bomb[0]
                    time_till_explosion = bomb[1]
                    blast_indices = []
                    for i in range(4):
                        if 0 < x + i < COLS:
                            blast_indices.append((x + i, y))
                        if COLS > x - i > 0:
                            blast_indices.append((x - i, y))
                        if 0 < y + i < ROWS:
                            blast_indices.append((x, y + i))
                        if ROWS > y - i > 0:
                            blast_indices.append((x, y - i))
                    if (nx, ny) in blast_indices and time_till_explosion == 0:
                        next_node_in_active_blast_next_step = True
                res.append(int(next_node_in_active_blast or next_node_in_active_blast_next_step))
        return np.array(res)


class CouldEscapeOwnBomb(BooleanFeature):
    def compute_feature(self, game_state: dict, movement_graph: MovementGraph) -> np.array:
        old_bomb_indices = copy.deepcopy(movement_graph.bomb_indices)
        movement_graph.bomb_indices.append(get_agent_position(game_state))

        res = move_out_of_blast_zone(game_state, movement_graph) != Actions.NONE

        movement_graph.bomb_indices = old_bomb_indices
        self.bomb_indices = old_bomb_indices

        return np.array([int(res)])


example_game_state = {
    "round": 1,
    "step": 1,
    "field": np.array(
        [
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, -1],
            [-1, 0, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 0, -1],
            [-1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, -1],
            [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 0, -1, 1, -1, 1, -1],
            [-1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, -1],
            [-1, 0, -1, 1, -1, 0, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            [-1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, -1],
            [-1, 1, -1, 0, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            [-1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, -1],
            [-1, 1, -1, 0, -1, 0, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            [-1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, -1],
            [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            [-1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, -1],
            [-1, 0, -1, 1, -1, 1, -1, 0, -1, 1, -1, 1, -1, 1, -1, 0, -1],
            [-1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        ]
    ),
    "self": ("rule_based_agent", 0, True, (15, 15)),
    "others": [],
    "bombs": [],
    "coins": [],
    "user_input": "BOMB",
    "explosion_map": np.array(
        [
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ]
    ),
}


class FeatureCollector(Feature):
    def __init__(self, *features: List[Feature]):
        self.features: List[Feature] = features

    def dim(self) -> int:
        return sum(f.dim() for f in self.features)

    def compute_feature(self, game_state: dict, movement_graph: MovementGraph = None) -> np.array:
        movement_graph = MovementGraph(game_state)
        vecs = [f.compute_feature(game_state, movement_graph).flatten() for f in self.features]

        return np.concatenate(vecs)

    def explain_feature(self, feature_vector: np.array) -> str:
        explainations = []

        index = 0
        for f in self.features:
            v = feature_vector[index : index + f.dim()]
            explainations.append(f.explain_feature(v))
            index += f.dim()

        return "\n".join(explainations)

    def print_feature_summary(self, feature_vector):
        print(self.explain_feature(feature_vector))
