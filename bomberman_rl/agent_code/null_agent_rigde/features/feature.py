from .feature_utils import (
    format_boolean,
    format_position,
    get_enemy_positions,
    ROWS,
    COLS,
    action_new_index,
    get_agent_position,
    camel_to_snake_case,
)
from abc import ABCMeta, abstractmethod
from typing import List
import numpy as np
import copy
from .movement_graph import MovementGraph, to_node
from .actions import Actions
import settings as s


class Feature(metaclass=ABCMeta):
    """Author: Samuel Melm"""

    def name(self):
        return camel_to_snake_case(type(self).__name__)

    def description(self):
        return self.name().replace("_", " ")

    @abstractmethod
    def dim(self) -> int:
        pass

    @abstractmethod
    def compute_feature(self, game_state: dict, self_obj) -> np.array:
        pass

    def format_feature(self, feature_vector: np.array) -> str:
        return str(feature_vector)

    def explain_feature(self, feature_vector: np.array) -> str:
        return f"{self.description()}: {self.format_feature(feature_vector)}"


class TwoDimensionalFeature(Feature):
    """Author: Samuel Melm"""

    def dim(self) -> int:
        return 2

    def format_feature(self, feature_vector: np.array) -> str:
        return format_position(feature_vector)


class BooleanFeature(Feature):
    """Author: Samuel Melm"""

    def dim(self) -> int:
        return 1

    def format_feature(self, feature_vector: np.array) -> str:
        return format_boolean(feature_vector)


class ActionFeature(Feature):
    """Author: Samuel Melm"""

    def dim(self) -> int:
        return len(Actions) - 1  # since we dont consider NONE

    def format_feature(self, feature_vector: np.array) -> str:
        return Actions.from_one_hot(feature_vector).name


class AgentPosition(TwoDimensionalFeature):
    """Author: Samuel Melm"""

    def compute_feature(self, game_state: dict, self_obj) -> np.array:
        return np.array(get_agent_position(game_state))


class BombDropPossible(BooleanFeature):
    """Author: Samuel Melm"""

    def compute_feature(self, game_state: dict, self_obj) -> np.array:
        bomb_possible = game_state["self"][-2]
        return np.array([int(bomb_possible)])


class MoveToNearestCoin(ActionFeature):
    """Author: Samuel Melm"""

    def compute_feature(self, game_state: dict, self_obj) -> np.array:
        return self_obj.movement_graph.next_step_to_nearest_index(game_state["coins"]).as_one_hot()


def move_out_of_blast_zone(game_state: dict, movement_graph: MovementGraph) -> Actions:
    """Author: Jonas Gann"""
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
        return movement_graph.next_step_to_nearest_index(free_indices)
    else:
        return Actions.NONE


class MoveOutOfBlastZone(ActionFeature):
    def compute_feature(self, game_state: dict, self_obj) -> np.array:
        return move_out_of_blast_zone(game_state, self_obj.movement_graph).as_one_hot()


class MoveNextToNearestBox(ActionFeature):
    """Author: Jonas Gann"""

    def compute_feature(self, game_state: dict, self_obj) -> np.array:
        box_neighbors = []
        indices = np.argwhere(game_state["field"] == 1)
        tuple_indices = [(index[0], index[1]) for index in indices]
        # find all neighbors of boxes the agent can move to (the box itsel is always out of range for the agent)
        for x, y in tuple_indices:
            neighbors = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
            for nx, ny in neighbors:
                if not self_obj.movement_graph.index_obstructed((nx, ny)):
                    box_neighbors.append((nx, ny))
        if get_agent_position(game_state) in box_neighbors:
            return Actions.WAIT.as_one_hot()
        return self_obj.movement_graph.next_step_to_nearest_index(box_neighbors).as_one_hot()


class MoveToNearestEnemy(ActionFeature):
    """Author: Jonas Gann"""

    def compute_feature(self, game_state: dict, self_obj) -> np.array:
        enemy_indices = get_enemy_positions(game_state)

        enemy_neighbors = []
        tuple_indices = [(index[0], index[1]) for index in enemy_indices]
        # find all neighbors of enemyes the agent can move to (the enemy itsel is always out of range for the agent)
        for x, y in tuple_indices:
            neighbors = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
            for nx, ny in neighbors:
                if not self_obj.movement_graph.index_obstructed((nx, ny)):
                    enemy_neighbors.append((nx, ny))
        if get_agent_position(game_state) in enemy_neighbors:
            return Actions.WAIT.as_one_hot()
        return self_obj.movement_graph.next_step_to_nearest_index(enemy_neighbors).as_one_hot()


class EnemiesInBlastRange(Feature):
    """Author: Jonas Gann"""

    def dim(self) -> int:
        return 1

    def compute_feature(self, game_state: dict, self_obj) -> np.array:
        sum = 0
        x, y = get_agent_position(game_state)
        field = game_state["field"]
        enemy_indices = get_enemy_positions(game_state)

        for i in range(4):
            if field[x + i, y] == -1:
                break
            if 0 < x + i < s.COLS and (x + i, y) in enemy_indices:
                sum += 1
        for i in range(4):
            if field[x - i, y] == -1:
                break
            if s.COLS > x - i > 0 and (x - i, y) in enemy_indices:
                sum += 1
        for i in range(4):
            if field[x, y + i] == -1:
                break
            if 0 < y + i < s.ROWS and (x, y + i) in enemy_indices:
                sum += 1
        for i in range(4):
            if field[x, y - i] == -1:
                break
            if s.ROWS > y - i > 0 and (x, y - i) in enemy_indices:
                sum += 1
        # print("boxes_in_blast",sum)
        return np.array([sum])


class PastMoves(Feature):
    """Author: Jonas Gann"""

    def __init__(self, n=4) -> None:
        super().__init__()
        self.n = n

    def dim(self) -> int:
        return self.n

    def compute_feature(self, game_state: dict, self_obj) -> np.array:
        if len(self_obj.past_moves) < self.n:
            return np.array([-1 for i in range(self.n)])
        return np.array([action.value for action in self_obj.past_moves[-self.n:]])


class BoxesInBlastRange(Feature):
    """Author: Jonas Gann"""

    def dim(self) -> int:
        return 1

    def compute_feature(self, game_state: dict, self_obj) -> np.array:
        sum = 0
        x, y = get_agent_position(game_state)
        field = game_state["field"]

        for i in range(4):
            if field[x + i, y] == -1:
                break
            if 0 < x + i < s.COLS and field[x + i, y] == 1:
                sum += 1
        for i in range(4):
            if field[x - i, y] == -1:
                break
            if s.COLS > x - i > 0 and field[x - i, y] == 1:
                sum += 1
        for i in range(4):
            if field[x, y + i] == -1:
                break
            if 0 < y + i < s.ROWS and field[x, y + i] == 1:
                sum += 1
        for i in range(4):
            if field[x, y - i] == -1:
                break
            if s.ROWS > y - i > 0 and field[x, y - i] == 1:
                sum += 1
        return np.array([sum])


class AgentInBlastZone(BooleanFeature):
    """Author: Jonas Gann"""

    def compute_feature(self, game_state: dict, self_obj) -> np.array:
        in_blast_zone = to_node(get_agent_position(
            game_state)) in self_obj.movement_graph.blast_nodes()
        return np.array([int(in_blast_zone)])


class PossibleActions(Feature):
    """Author: Jonas Gann"""

    def dim(self) -> int:
        return len(Actions) - 1  # since we do not consider WAIT or NONE

    def compute_feature(self, game_state: dict, self_obj) -> np.array:
        agent_position = get_agent_position(game_state)
        res = []
        for action in Actions:
            if action in [Actions.NONE]:
                pass
            elif action == Actions.BOMB:
                res.append(int(game_state["self"][2]))
            else:
                new_index = action_new_index(agent_position, action)
                obstructed = self_obj.movement_graph.index_obstructed(
                    new_index)
                res.append(int(not obstructed))
        return np.array(res)


class CouldEscapeOwnBomb(BooleanFeature):
    """Author: Jonas Gann"""

    def compute_feature(self, game_state: dict, self_obj) -> np.array:
        old_bomb_indices = copy.deepcopy(self_obj.movement_graph.bomb_indices)
        self_obj.movement_graph.bomb_indices.append(
            get_agent_position(game_state))

        res = move_out_of_blast_zone(
            game_state, self_obj.movement_graph) != Actions.NONE

        self_obj.movement_graph.bomb_indices = old_bomb_indices
        self.bomb_indices = old_bomb_indices

        return np.array([int(res)])


class AgentFieldNeighbors(Feature):
    """Author: Jonas Gann"""

    def compute_feature(self, game_state: dict, self_obj) -> np.array:
        neighbors = []
        x, y = get_agent_position(game_state)
        for i in range(2):
            if 0 < x + i < COLS:
                neighbors.append(game_state["field"][x + i][y])
            else:
                neighbors.append(-2)
            if COLS > x - i > 0:
                neighbors.append(game_state["field"][x - i][y])
            else:
                neighbors.append(-2)
            if 0 < y + i < ROWS:
                neighbors.append(game_state["field"][x][y + i])
            else:
                neighbors.append(-2)
            if ROWS > y - i > 0:
                neighbors.append(game_state["field"][x][y - i])
            else:
                neighbors.append(-2)
        return np.array(neighbors)

    def dim(self) -> int:
        return 8


class AgentExplosionNeighbors(Feature):
    """Author: Jonas Gann"""

    def compute_feature(self, game_state: dict, self_obj) -> np.array:
        neighbors = []
        x, y = get_agent_position(game_state)
        for i in range(2):
            if 0 < x + i < COLS:
                neighbors.append(game_state["explosion_map"][x + i][y])
            else:
                neighbors.append(-1)
            if COLS > x - i > 0:
                neighbors.append(game_state["explosion_map"][x - i][y])
            else:
                neighbors.append(-1)
            if 0 < y + i < ROWS:
                neighbors.append(game_state["explosion_map"][x][y + i])
            else:
                neighbors.append(-1)
            if ROWS > y - i > 0:
                neighbors.append(game_state["explosion_map"][x][y - i])
            else:
                neighbors.append(-1)
        return np.array(neighbors)

    def dim(self) -> int:
        return 8

# If the enemy can move in only one or two directions and our agent can find a
# path to the enemy, he should place a bomb next to him


class NearestEnemyPossibleMoves(Feature):
    """Author: Jonas Gann"""

    def compute_feature(self, game_state: dict, self_obj) -> np.array:
        enemy_indices = get_enemy_positions(game_state)
        nearest_index = self_obj.movement_graph.nearest_index(enemy_indices)
        if nearest_index != None:
            x, y = nearest_index
            enemy_movement_range = 0
            if self_obj.movement_graph.index_obstructed((x + 1, y)):
                enemy_movement_range += 1
            if self_obj.movement_graph.index_obstructed((x - 1, y)):
                enemy_movement_range += 1
            if self_obj.movement_graph.index_obstructed((x, y + 1)):
                enemy_movement_range += 1
            if self_obj.movement_graph.index_obstructed((x, y - 1)):
                enemy_movement_range += 1
            return np.array([enemy_movement_range])
        else:
            return np.array([-1])

    def dim(self) -> int:
        return 1


class CoinDistance(Feature):
    """Author: Jonas Gann"""

    def compute_feature(self, game_state: dict, self_obj) -> np.array:
        coin_indices = game_state["coins"]
        nearest_index = self_obj.movement_graph.nearest_index(coin_indices)
        if nearest_index == None:
            return np.array([-1])
        distance = self_obj.movement_graph.nearest_distance_index(
            nearest_index)
        if distance != None:
            return np.array([distance])
        else:
            return np.array([-1])

    def dim(self) -> int:
        return 1


class BoxDistance(Feature):
    """Author: Jonas Gann"""

    def compute_feature(self, game_state: dict, self_obj) -> np.array:
        box_neighbors = []
        indices = np.argwhere(game_state["field"] == 1)
        tuple_indices = [(index[0], index[1]) for index in indices]
        # find all neighbors of boxes the agent can move to (the box itsel is always out of range for the agent)
        for x, y in tuple_indices:
            neighbors = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
            for nx, ny in neighbors:
                if not self_obj.movement_graph.index_obstructed((nx, ny)):
                    box_neighbors.append((nx, ny))
        if get_agent_position(game_state) in box_neighbors:
            return np.array([0])
        nearest_index = self_obj.movement_graph.nearest_index(box_neighbors)
        if nearest_index == None:
            return np.array([-1])
        distance = self_obj.movement_graph.nearest_distance_index(
            nearest_index)
        if distance != None:
            return np.array([distance])
        else:
            return np.array([-1])

    def dim(self) -> int:
        return 1


class EnemyDistance(Feature):
    """Author: Jonas Gann"""

    def compute_feature(self, game_state: dict, self_obj) -> np.array:
        enemy_indices = get_enemy_positions(game_state)
        enemy_neighbors = []
        tuple_indices = [(index[0], index[1]) for index in enemy_indices]
        # find all neighbors of enemyes the agent can move to (the enemy itsel is always out of range for the agent)
        for x, y in tuple_indices:
            neighbors = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
            for nx, ny in neighbors:
                if not self_obj.movement_graph.index_obstructed((nx, ny)):
                    enemy_neighbors.append((nx, ny))
        if get_agent_position(game_state) in enemy_neighbors:
            return np.array([0])
        nearest_index = self_obj.movement_graph.nearest_index(enemy_neighbors)
        if nearest_index == None:
            return np.array([-1])
        distance = self_obj.movement_graph.nearest_distance_index(
            nearest_index)
        if distance != None:
            return np.array([distance])
        else:
            return np.array([-1])

    def dim(self) -> int:
        return 1


class SafetyDistance(Feature):
    """Author: Jonas Gann"""

    def compute_feature(self, game_state: dict, self_obj) -> np.array:
        agent_position = get_agent_position(game_state)
        free_indices = []
        blast_indices = self_obj.movement_graph.blast_indices()

        for x in range(COLS):
            for y in range(ROWS):
                index = (x, y)
                obstructed = self_obj.movement_graph.index_obstructed(index)
                in_blast_zone = index in blast_indices
                if not obstructed and not in_blast_zone:
                    free_indices.append(index)
        if agent_position in blast_indices:
            nearest_index = self_obj.movement_graph.nearest_index(free_indices)
            if nearest_index == None:
                return np.array([-1])
            distance = self_obj.movement_graph.nearest_distance_index(
                nearest_index)
            if distance != None:
                return np.array([distance])
            else:
                return np.array([-1])
        else:
            return np.array([-1])

    def dim(self) -> int:
        return 1


class FeatureCollector(Feature):
    """Author: Samuel Melm"""

    def __init__(self, *features: List[Feature]):
        self.features: List[Feature] = features

    def dim(self) -> int:
        return sum(f.dim() for f in self.features)

    def compute_feature(self, game_state: dict, self_obj) -> np.array:
        self_obj.movement_graph = MovementGraph(game_state)

        vecs = [f.compute_feature(game_state, self_obj).flatten()
                for f in self.features]

        return np.concatenate(vecs)

    def explain_feature(self, feature_vector: np.array) -> str:
        explainations = []

        index = 0
        for f in self.features:
            v = feature_vector[index: index + f.dim()]
            explainations.append(f.explain_feature(v))
            index += f.dim()

        return "\n".join(explainations)

    def single_feature_from_vector(self, feature_vector, feature_class):
        index = 0

        for f in self.features:
            if isinstance(f, feature_class):
                return feature_vector[index: index + f.dim()]
            index += f.dim()

        raise ValueError(f"no entry in feature collector for {feature_class}")

    def print_feature_summary(self, feature_vector):
        print()
        print("{:*^30}".format(" feature summary "))
        print()
        print(feature_vector)
        print()
        print(self.explain_feature(feature_vector))
        print()
        print("{:*^30}".format(" end of feature summary "))
        print()

    @classmethod
    def create_current_version(cls):
        return FeatureCollector(
            MoveToNearestCoin(),
            MoveOutOfBlastZone(),
            MoveNextToNearestBox(),
            AgentInBlastZone(),
            PossibleActions(),
            BoxesInBlastRange(),
            CouldEscapeOwnBomb(),
            MoveToNearestEnemy(),
            PastMoves(),
            EnemiesInBlastRange(),
            AgentFieldNeighbors(),
            AgentExplosionNeighbors(),
            NearestEnemyPossibleMoves(),
            SafetyDistance(),
            EnemyDistance(),
            BoxDistance(),
            CoinDistance()
        )
