from enum import Enum
from typing import List, Tuple
import numpy as np
import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import numpy as np
from enum import Enum
import copy

COLS, ROWS = 17, 17


class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    WAIT = 4
    BOMB = 5
    NONE = 6

    def as_string(self):
        self.name

    def as_one_hot(self):
        return [int(self.value == index) for index in range(6)]


def bomb_indices(game_state: dict) -> List[Tuple]:
    return [(bomb[0][0], bomb[0][1]) for bomb in game_state["bombs"]]


def to_node(index):
    return index[1] * COLS + index[0]


def to_index(node):
    return (node % COLS, math.floor(node / COLS))


class MovementGraph:
    def __init__(self, game_state: dict) -> None:
        self.field = game_state["field"]
        self.explosion_map = game_state["explosion_map"]
        self.bomb_indices = bomb_indices(game_state)
        self.agent_index = game_state["self"][3]
        self.agent_node = to_node(self.agent_index)

        row = []
        col = []
        data = []
        for x in range(COLS):
            for y in range(ROWS):
                neighbors = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
                for nx, ny in neighbors:
                    if self.is_within_field(nx, ny):
                        node = to_node((x, y))
                        neighbor = to_node((nx, ny))
                        if not self.node_obstructed(node) and not self.node_obstructed(
                            neighbor
                        ):
                            row.append(node)
                            col.append(neighbor)
                            data.append(1)
        self.matrix = csr_matrix((data, (row, col)))

    def is_within_field(self, x: int, y: int) -> bool:
        return 0 <= x < COLS and 0 <= y < ROWS

    def node_obstructed(self, node):
        # Check if the field is obstructed at the node by wither a wall, a boy or an explosion
        x, y = to_index(node)
        return self.index_obstructed((x, y))

    def next_step_to_nearest_node(self, nodes):
        # Find the nearest reachable node in from the nodes array originating from the agent position and return the next move along the shortest path
        nodes = self.remove_obstructed_nodes(nodes)
        nodes = self.remove_nodes_out_of_range(nodes)
        if len(nodes) == 0:
            return Actions.NONE
        distances, predecessors, sources = dijkstra(
            csgraph=self.matrix,
            directed=True,
            indices=nodes,
            return_predecessors=True,
            unweighted=True,
            min_only=True,
        )
        if not self._node_in_movement_range(self.agent_node):
            # Agent node is currently seperated from all other nodes and is therefore not contained in the movement_graph
            return Actions.NONE
        source = sources[self.agent_node]
        if source != -9999:  # A path to one of the nodes exists
            distance = distances[self.agent_node]
            next_node = predecessors[self.agent_node]
            cx, cy = to_index(next_node)
            ax, ay = self.agent_index
            if cx - ax > 0:
                return Actions.RIGHT
            elif cx - ax < 0:
                return Actions.LEFT
            elif cy - ay > 0:
                return Actions.DOWN
            elif cy - ay < 0:
                return Actions.UP
        else:
            return Actions.NONE

    def index_obstructed(self, index):
        # Check if the field is obstructed at the index by wither a wall, a boy, an explosion or an out of range error
        x, y = index
        in_range = 0 <= x < COLS and 0 <= y < ROWS
        is_wall = self.field[x, y] == -1
        is_box = self.field[x, y] == 1
        is_explosion = self.explosion_map[x, y] != 0
        is_bomb = (x, y) in self.bomb_indices and self.agent_index != (x, y)
        return is_wall or is_box or is_explosion or not in_range or is_bomb

    def _node_in_movement_range(self, node):
        # It can happen that NOT obstructed nodes exist which are not reachable through any edge.
        # These nodes are not added to the movement_graph during creation of the adjacency list.
        # Therefore not filtering them out leads to out of range errors.
        return node < self.matrix.shape[0]

    def _index_in_movement_range(self, index):
        node = to_node(index)
        return self._node_in_movement_range(node)

    def remove_obstructed_nodes(self, nodes):
        free_nodes = []
        for node in nodes:
            if not self.node_obstructed(node):
                free_nodes.append(node)
        return free_nodes

    def remove_nodes_out_of_range(self, nodes):
        # It can happen that NOT obstructed nodes exist which are not reachable through any edge.
        # These nodes are not added to the movement_graph during creation of the adjacency list.
        # Therefore not filtering them out leads to out of range errors.
        free_nodes = []
        for node in nodes:
            if self._node_in_movement_range(node):
                free_nodes.append(node)
        return free_nodes

    def next_step_to_nearest_index(self, indices):
        nodes = [to_node(index) for index in indices]
        return self.next_step_to_nearest_node(nodes)

    def blast_indices(self):
        blast_indices = []
        for x, y in self.bomb_indices:
            for i in range(4):
                if 0 < x + i < COLS:
                    blast_indices.append((x + i, y))
                if COLS > x - i > 0:
                    blast_indices.append((x - i, y))
                if 0 < y + i < ROWS:
                    blast_indices.append((x, y + i))
                if ROWS > y - i > 0:
                    blast_indices.append((x, y - i))
        return blast_indices

    def blast_nodes(self):
        blast_indices = self.blast_indices()
        return [to_node(index) for index in blast_indices]


class FeatureExtraction:
    def __init__(self, game_state):
        self.game_state = game_state
        self.field = self.game_state["field"]
        self.agent_node = to_node(self.game_state["self"][3])
        self.agent_index = self.game_state["self"][3]
        self.coins = game_state["coins"]
        self.bombs = game_state["bombs"]
        self.bomb_indices = bomb_indices(game_state)
        self.movement_graph = MovementGraph(game_state)
        self.explosion_map = game_state["explosion_map"]

    def FEATURE_move_to_nearest_coin(self):
        return self.movement_graph.next_step_to_nearest_index(self.coins)

    def FEATURE_move_out_of_blast_zone(self):
        free_indices = []
        blast_indices = self.movement_graph.blast_indices()
        for x in range(COLS):
            for y in range(ROWS):
                index = (x, y)
                obstructed = self.movement_graph.index_obstructed(index)
                in_blast_zone = index in blast_indices
                if not obstructed and not in_blast_zone:
                    free_indices.append(index)
        if self.agent_index in blast_indices:
            return self._next_step_to_nearest_index(free_indices)
        else:
            return Actions.NONE

    def FEATURE_move_next_to_nearest_box(self):
        box_neighbors = []
        indices = np.argwhere(self.field == 1)
        tuple_indices = [(index[0], index[1]) for index in indices]
        # find all neighbors of boxes the agent can move to (the box itsel is always out of range for the agent)
        for x, y in tuple_indices:
            neighbors = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
            for nx, ny in neighbors:
                if not self.movement_graph.index_obstructed((nx, ny)):
                    box_neighbors.append((nx, ny))
        if self.agent_index in box_neighbors:
            return Actions.NONE
        step = self.movement_graph.next_step_to_nearest_index(box_neighbors)
        return step

    def FEATURE_boxes_in_agent_blast_range(self):
        sum = 0
        x, y = self.agent_index
        for i in range(4):
            if 0 < x + i < COLS and self.field[x + i, y] == 1:
                sum += 1
            if COLS > x - i > 0 and self.field[x - i, y] == 1:
                sum += 1
            if 0 < y + i < ROWS and self.field[x, y + i] == 1:
                sum += 1
            if ROWS > y - i > 0 and self.field[x, y - i] == 1:
                sum += 1
        return [sum]

    def FEATURE_in_blast_zone(self):
        blast_nodes = self.movement_graph.blast_nodes()
        if self.agent_node in blast_nodes:
            return [1]
        else:
            return [0]

    def _action_new_index(self, action):
        # return the node an action puts the agent on
        if action == Actions.UP:
            return (self.agent_index[0], self.agent_index[1] - 1)
        elif action == Actions.DOWN:
            return (self.agent_index[0], self.agent_index[1] + 1)
        elif action == Actions.LEFT:
            return (self.agent_index[0] - 1, self.agent_index[1])
        elif action == Actions.RIGHT:
            return (self.agent_index[0] + 1, self.agent_index[1])
        elif action == Actions.WAIT:
            return self.agent_index
        elif action == Actions.BOMB:
            return self.agent_index

    def _action_new_node(self, action):
        index = self._action_new_index(action)
        return to_node(index)

    def FEATURE_actions_possible(self):
        res = []
        for action in Actions:
            if action == Actions.NONE:
                pass
            elif action == Actions.WAIT:
                res.append(1)
            elif action == Actions.BOMB:
                res.append(int(self.game_state["self"][2]))
            else:
                new_index = self._action_new_index(action)
                obstructed = self.movement_graph.index_obstructed(new_index)
                res.append(int(not obstructed))
        return res

    def _action_possible(self, action):
        if action == Actions.NONE:
            pass
        elif action == Actions.WAIT:
            return True
        elif action == Actions.BOMB:
            return self.game_state["self"][2]
        else:
            new_index = self._action_new_index(action)
            obstructed = self._index_obstructed(new_index)
            return not obstructed

    def FEATURE_move_into_death(self):
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
                (nx, ny) = self._action_new_index(action)
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
                res.append(
                    int(
                        next_node_in_active_blast or next_node_in_active_blast_next_step
                    )
                )
        return res

    def FEATURE_could_escape_own_bomb(self):
        old_bomb_indices = copy.deepcopy(self.bomb_indices)
        self.bomb_indices.append(self.agent_index)
        res = self.FEATURE_move_out_of_blast_zone() != Actions.NONE
        self.bomb_indices = old_bomb_indices
        return [res]


fe = FeatureExtraction(
    {
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
)


print(fe.FEATURE_move_to_nearest_coin())
print(fe.FEATURE_move_out_of_blast_zone())
print(fe.FEATURE_move_next_to_nearest_box())
print(fe.FEATURE_boxes_in_agent_blast_range())
print(fe.FEATURE_in_blast_zone())
print(fe.FEATURE_actions_possible())
print(fe.FEATURE_move_into_death())
print(fe.FEATURE_could_escape_own_bomb())
